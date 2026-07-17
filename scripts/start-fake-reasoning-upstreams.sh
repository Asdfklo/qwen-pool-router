#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${NYX_PYTHON:-/opt/hermes-qwen-pool/.venv/bin/python}"
PID_DIR="${NYX_FAKE_PID_DIR:-/tmp/nyx-fake-reasoning-upstreams}"

stop() {
  if [[ -d "$PID_DIR" ]]; then
    for pid_file in "$PID_DIR"/*.pid; do
      [[ -f "$pid_file" ]] || continue
      kill "$(cat "$pid_file")" 2>/dev/null || true
    done
    rm -rf "$PID_DIR"
  fi
}

if [[ "${1:-start}" == "stop" ]]; then
  stop
  exit 0
fi

stop
mkdir -p "$PID_DIR"
for spec in "18081:FAKE_UPSTREAM_OFF" "18082:FAKE_UPSTREAM_MEDIUM" "18083:FAKE_UPSTREAM_HIGH"; do
  port="${spec%%:*}"
  label="${spec#*:}"
  nohup "$PYTHON" "$ROOT/scripts/fake_reasoning_upstream.py" --port "$port" --label "$label" \
    >"$PID_DIR/$port.log" 2>&1 &
  echo "$!" >"$PID_DIR/$port.pid"
done

for port in 18081 18082 18083; do
  for _ in {1..50}; do
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null && break
    sleep 0.1
  done
  curl -fsS "http://127.0.0.1:$port/health" >/dev/null
done

echo "Fake reasoning upstreams ready on 18081, 18082, and 18083"
