#!/usr/bin/env bash
set -euo pipefail

BACKUP_ROOT="/opt/hermes-qwen-pool/backups/router-20260620-123804"
BACKUP_CONFIG="/etc/hermes-qwen-pool/router.yaml.bak-20260620-123804"

sudo systemctl stop hermes-router.service
sudo cp -a "$BACKUP_ROOT/router/." /opt/hermes-qwen-pool/router/
sudo cp -a "$BACKUP_ROOT/source-router/." /home/charles/services/hermes-qwen-pool/router/
sudo cp -a "$BACKUP_CONFIG" /etc/hermes-qwen-pool/router.yaml
sudo systemctl start hermes-router.service
sleep 2
systemctl is-active hermes-router.service
curl -fsS http://127.0.0.1:4000/v1/models | jq .
