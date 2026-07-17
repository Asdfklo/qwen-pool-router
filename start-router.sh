#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-${HERMES_ROUTER_CONFIG:-router/config.yaml.example}}"
python3 router/hermes_router.py --config "$CONFIG"
