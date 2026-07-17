#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${NYX_PYTHON:-/opt/hermes-qwen-pool/.venv/bin/python}"
ROUTER_PID=""

cleanup() {
  if [[ -n "$ROUTER_PID" ]]; then
    kill "$ROUTER_PID" 2>/dev/null || true
    wait "$ROUTER_PID" 2>/dev/null || true
  fi
  "$ROOT/scripts/start-fake-reasoning-upstreams.sh" stop
}
trap cleanup EXIT

if ss -ltn | grep -q ':4001 '; then
  echo "Port 4001 is already in use" >&2
  exit 1
fi

cd "$ROOT"
"$PYTHON" -m py_compile hermes_router.py reasoning.py scripts/fake_reasoning_upstream.py
"$PYTHON" -m pytest -q test_reasoning.py
"$PYTHON" -c 'from hermes_router import load_config, create_app; c=load_config("router-beta-fake.yaml"); assert c.listen_port == 4001; create_app(c); print("beta config/app load ok")'

./scripts/start-fake-reasoning-upstreams.sh
"$PYTHON" hermes_router.py --config router-beta-fake.yaml >/tmp/nyx-router-beta-fake.log 2>&1 &
ROUTER_PID="$!"

for _ in {1..100}; do
  curl -fsS http://127.0.0.1:4001/v1/models >/dev/null 2>&1 && break
  sleep 0.1
done
curl -fsS http://127.0.0.1:4001/v1/models >/dev/null

"$PYTHON" test_router_smoke.py --base http://127.0.0.1:4001
NYX_ROUTER_URL=http://127.0.0.1:4001 ./scripts/test-reasoning-routing.sh

echo "Fake beta validation passed"
