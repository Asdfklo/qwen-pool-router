#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR=/opt/hermes-qwen-pool
CONFIG_DIR=/etc/hermes-qwen-pool

sudo mkdir -p "$INSTALL_DIR" "$CONFIG_DIR"
sudo cp -R "$SRC_DIR"/router "$SRC_DIR"/node "$SRC_DIR"/requirements.txt "$INSTALL_DIR"/
sudo python3 -m venv "$INSTALL_DIR/.venv"
sudo "$INSTALL_DIR/.venv/bin/python" -m pip install -U pip
sudo "$INSTALL_DIR/.venv/bin/python" -m pip install -r "$INSTALL_DIR/requirements.txt"

if [[ ! -f "$CONFIG_DIR/router.yaml" ]]; then
  sudo cp "$SRC_DIR/router/config.yaml.example" "$CONFIG_DIR/router.yaml"
fi
if [[ ! -f "$CONFIG_DIR/router.env" ]]; then
  sudo cp "$SRC_DIR/.env.example" "$CONFIG_DIR/router.env"
fi

sudo cp "$SRC_DIR/router/hermes-router.service" /etc/systemd/system/hermes-router.service
sudo systemctl daemon-reload

if command -v ufw >/dev/null 2>&1; then
  sudo ufw allow 4000/tcp comment "Hermes Qwen pool router" || true
fi

echo "Router installed."
echo "Edit: sudo nano $CONFIG_DIR/router.yaml"
echo "Start: sudo systemctl enable --now hermes-router"
echo "Status: systemctl status hermes-router"
echo "Logs: journalctl -u hermes-router -f"
echo "Health: curl http://192.168.68.99:4000/health"
echo "Models: curl http://192.168.68.99:4000/v1/models"
echo "Hermes/Open WebUI base URL: http://192.168.68.99:4000/v1"
