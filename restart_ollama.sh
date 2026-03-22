#!/bin/bash
set -e

OVERRIDE_DIR=/etc/systemd/system/ollama.service.d
OVERRIDE_FILE=$OVERRIDE_DIR/override.conf

mkdir -p "$OVERRIDE_DIR"

cat > "$OVERRIDE_FILE" << 'EOF'
[Service]
Environment="OLLAMA_NUM_PARALLEL=2"
EOF

echo "Written: $OVERRIDE_FILE"
cat "$OVERRIDE_FILE"

systemctl daemon-reload
systemctl restart ollama
sleep 2
systemctl status ollama --no-pager | head -20
