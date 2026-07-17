#!/usr/bin/env bash
set -euo pipefail

BETA="/opt/hermes-qwen-pool/router-beta-20260620-nyx-reasoning"
PROD="/opt/hermes-qwen-pool/router"
SOURCE="/home/charles/services/hermes-qwen-pool/router"
PROD_CONFIG="/etc/hermes-qwen-pool/router.yaml"
STAGED_CONFIG="/tmp/router-production-health-reload.yaml"
BACKUP_ROOT="/opt/hermes-qwen-pool/backups/router-20260620-160628"
BACKUP_CONFIG="/etc/hermes-qwen-pool/router.yaml.bak-20260620-160628"
PYTHON="/opt/hermes-qwen-pool/.venv/bin/python"

rollback() {
  echo "Deployment failed; restoring router backup" >&2
  sudo systemctl stop hermes-router.service || true
  sudo cp -a "$BACKUP_ROOT/router/." "$PROD/"
  sudo cp -a "$BACKUP_ROOT/source-router/." "$SOURCE/"
  sudo cp -a "$BACKUP_CONFIG" "$PROD_CONFIG"
  sudo systemctl start hermes-router.service
}
active=""
queued=""
for _ in {1..1200}; do
  snapshot="$(curl -fsS http://127.0.0.1:4000/api/status)"
  active="$(jq '[.backends[].active_requests] | add' <<<"$snapshot")"
  queued="$(jq '.queued_requests' <<<"$snapshot")"
  [[ "$active" == "0" && "$queued" == "0" ]] && break
  sleep 0.5
done
if [[ "$active" != "0" || "$queued" != "0" ]]; then
  echo "Production did not become idle; refusing deployment" >&2
  exit 2
fi

sudo systemctl stop hermes-router.service
trap 'rollback' ERR

for file in hermes_router.py reasoning.py README.md config.yaml.example test_reasoning.py test_health_reload.py test_router_smoke.py; do
  sudo install -o root -g root -m 0644 "$BETA/$file" "$PROD/$file"
  sudo install -o charles -g charles -m 0644 "$BETA/$file" "$SOURCE/$file"
done
sudo mkdir -p "$PROD/scripts" "$SOURCE/scripts"
for file in "$BETA"/scripts/*; do
  [[ -f "$file" ]] || continue
  sudo install -o root -g root -m 0755 "$file" "$PROD/scripts/$(basename "$file")"
  sudo install -o charles -g charles -m 0755 "$file" "$SOURCE/scripts/$(basename "$file")"
done
sudo install -o root -g root -m 0644 "$STAGED_CONFIG" "$PROD_CONFIG"

cd "$PROD"
sudo "$PYTHON" -m py_compile hermes_router.py reasoning.py test_health_reload.py
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -q -p no:cacheprovider test_reasoning.py test_health_reload.py
"$PYTHON" hermes_router.py --config "$PROD_CONFIG" --check-config

sudo systemctl start hermes-router.service
for _ in {1..100}; do
  curl -fsS http://127.0.0.1:4000/v1/models >/dev/null 2>&1 && break
  sleep 0.2
done
curl -fsS http://127.0.0.1:4000/v1/models >/dev/null
trap - ERR
echo "Production semantic health/reload deployment passed"
