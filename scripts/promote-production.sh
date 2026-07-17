#!/usr/bin/env bash
set -euo pipefail

BETA="/opt/hermes-qwen-pool/router-beta-20260620-nyx-reasoning"
PROD="/opt/hermes-qwen-pool/router"
SOURCE="/home/charles/services/hermes-qwen-pool/router"
PROD_CONFIG="/etc/hermes-qwen-pool/router.yaml"
STAGED_CONFIG="/tmp/router-production-reasoning.yaml"
BACKUP_ROOT="/opt/hermes-qwen-pool/backups/router-20260620-123804"
BACKUP_CONFIG="/etc/hermes-qwen-pool/router.yaml.bak-20260620-123804"
PYTHON="/opt/hermes-qwen-pool/.venv/bin/python"

rollback_startup() {
  echo "Production startup failed; restoring backup" >&2
  sudo systemctl stop hermes-router.service || true
  sudo cp -a "$BACKUP_ROOT/router/." "$PROD/"
  sudo cp -a "$BACKUP_ROOT/source-router/." "$SOURCE/"
  sudo cp -a "$BACKUP_CONFIG" "$PROD_CONFIG"
  sudo systemctl start hermes-router.service
}

active=""
for _ in {1..1200}; do
  active="$(curl -fsS http://127.0.0.1:4000/api/status | jq '[.backends[].active_requests] | add')"
  [[ "$active" == "0" ]] && break
  sleep 0.5
done
if [[ "$active" != "0" ]]; then
  echo "Production did not become idle within 10 minutes; refusing restart" >&2
  exit 2
fi

sudo systemctl stop hermes-router.service

for file in hermes_router.py reasoning.py README.md config.yaml.example test_reasoning.py test_router_smoke.py; do
  sudo install -o root -g root -m 0644 "$BETA/$file" "$PROD/$file"
done
sudo mkdir -p "$PROD/scripts"
for file in "$BETA"/scripts/*; do
  [[ -f "$file" ]] || continue
  sudo install -o root -g root -m 0755 "$file" "$PROD/scripts/$(basename "$file")"
done
sudo install -o root -g root -m 0644 "$STAGED_CONFIG" "$PROD_CONFIG"

sudo mkdir -p "$SOURCE/scripts"
for file in hermes_router.py reasoning.py README.md config.yaml.example test_reasoning.py test_router_smoke.py; do
  sudo install -o charles -g charles -m 0644 "$BETA/$file" "$SOURCE/$file"
done
for file in "$BETA"/scripts/*; do
  [[ -f "$file" ]] || continue
  sudo install -o charles -g charles -m 0755 "$file" "$SOURCE/scripts/$(basename "$file")"
done

cd "$PROD"
sudo "$PYTHON" -m py_compile hermes_router.py reasoning.py
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -q -p no:cacheprovider test_reasoning.py
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -c 'from hermes_router import load_config, create_app; c=load_config("/etc/hermes-qwen-pool/router.yaml"); assert c.listen_port == 4000; create_app(c); print("production config/app load ok")'

if ! sudo systemctl start hermes-router.service; then
  rollback_startup
  exit 1
fi

for _ in {1..50}; do
  curl -fsS http://127.0.0.1:4000/v1/models >/dev/null 2>&1 && break
  sleep 0.2
done
if ! curl -fsS http://127.0.0.1:4000/v1/models >/dev/null; then
  rollback_startup
  exit 1
fi

echo "Production promotion and startup gate passed"
