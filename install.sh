#!/usr/bin/env bash
# ============================================================
# TetraMem-XL v6.5 — One-Click Installer
# BCC Lattice Honeycomb + PCNN Neural Pulse Memory System
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/sunormesky-max/Tetrahedron-Memory-Hive-System/main/install.sh | bash
#
#   # With custom password:
#   curl -sSL https://raw.githubusercontent.com/sunormesky-max/Tetrahedron-Memory-Hive-System/main/install.sh | bash -s -- --password MySecret123
#
#   # With custom port:
#   curl -sSL ... | bash -s -- --port 9000 --password MySecret123
#
# Environment variables (all optional):
#   TETRAMEM_PASSWORD   — UI login password (default: auto-generated)
#   TETRAMEM_PORT       — API port (default: 8000)
#   TETRAMEM_UI_PORT    — Nginx UI port (default: 8082)
#   TETRAMEM_DIR        — Install directory (default: /opt/tetramem)
#   TETRAMEM_NO_NGINX   — Set to "1" to skip nginx setup
# ============================================================

set -euo pipefail

# --- Parse args ---
TETRAMEM_PASSWORD="${TETRAMEM_PASSWORD:-}"
TETRAMEM_PORT="${TETRAMEM_PORT:-8000}"
TETRAMEM_UI_PORT="${TETRAMEM_UI_PORT:-8082}"
TETRAMEM_DIR="${TETRAMEM_DIR:-/opt/tetramem}"
TETRAMEM_NO_NGINX="${TETRAMEM_NO_NGINX:-0}"
TETRAMEM_REPO="https://github.com/sunormesky-max/Tetrahedron-Memory-Hive-System.git"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --password) TETRAMEM_PASSWORD="$2"; shift 2 ;;
    --port) TETRAMEM_PORT="$2"; shift 2 ;;
    --ui-port) TETRAMEM_UI_PORT="$2"; shift 2 ;;
    --dir) TETRAMEM_DIR="$2"; shift 2 ;;
    --no-nginx) TETRAMEM_NO_NGINX=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# --- Colors ---
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${CYAN}[TetraMem]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# --- Preflight ---
log "TetraMem-XL v6.5 Installer"
log "============================"

[[ "$(id -u)" -ne 0 ]] && fail "Please run as root (or with sudo)"

# Generate password if not set
if [[ -z "$TETRAMEM_PASSWORD" ]]; then
  TETRAMEM_PASSWORD="TM$(openssl rand -hex 6 2>/dev/null || cat /dev/urandom | tr -dc 'a-zA-Z0-9' | head -c 12)"
  warn "No password set. Auto-generated: $TETRAMEM_PASSWORD"
fi

# --- Step 1: Dependencies ---
log "Step 1/7: Installing system dependencies..."
apt-get update -qq 2>/dev/null || yum install -y -q python3 python3-pip git 2>/dev/null || true
command -v python3 >/dev/null 2>&1 || { apt-get install -y -qq python3 python3-pip 2>/dev/null || yum install -y -q python3 python3-pip 2>/dev/null; }
command -v git >/dev/null 2>&1 || { apt-get install -y -qq git 2>/dev/null || yum install -y -q git 2>/dev/null; }
command -v python3 >/dev/null 2>&1 || fail "Python3 not found and could not be installed"
python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" || fail "Python 3.8+ required"
ok "System dependencies ready"

# --- Step 2: Clone ---
log "Step 2/7: Downloading TetraMem-XL..."
if [[ -d "$TETRAMEM_DIR/.git" ]]; then
  log "Directory exists, pulling latest..."
  git -C "$TETRAMEM_DIR" pull --ff-only 2>/dev/null || warn "Git pull failed, using existing code"
else
  rm -rf "$TETRAMEM_DIR" 2>/dev/null || true
  git clone --depth 1 "$TETRAMEM_REPO" "$TETRAMEM_DIR"
fi
ok "Code ready at $TETRAMEM_DIR"

# --- Step 3: Python deps ---
log "Step 3/7: Installing Python dependencies..."
pip3 install --quiet --upgrade pip 2>/dev/null || true
pip3 install --quiet numpy fastapi uvicorn pydantic 2>/dev/null || pip3 install numpy fastapi uvicorn pydantic
ok "Python dependencies installed"

# --- Step 4: Data directory ---
log "Step 4/7: Initializing data directory..."
mkdir -p "$TETRAMEM_DIR/tetramem_data_v2"
if [[ ! -f "$TETRAMEM_DIR/tetramem_data_v2/mesh_index.json" ]]; then
  echo '{"metadata":{"version":"6.5.0","created":"'$(date -Iseconds)'"},"tetrahedra":{}}' > "$TETRAMEM_DIR/tetramem_data_v2/mesh_index.json"
fi
ok "Data directory ready"

# --- Step 5: Configure ---
log "Step 5/7: Configuring..."
export TETRAMEM_STORAGE="$TETRAMEM_DIR/tetramem_data_v2"
export TETRAMEM_UI_PASSWORD="$TETRAMEM_PASSWORD"

# Fix static directory
mkdir -p "$TETRAMEM_DIR/static"
if [[ -f "$TETRAMEM_DIR/ui/index.html" ]]; then
  cp "$TETRAMEM_DIR/ui/index.html" "$TETRAMEM_DIR/static/index.html"
  sed -i "s/CHANGE_ME/$TETRAMEM_PASSWORD/g" "$TETRAMEM_DIR/static/index.html"
fi
if [[ -f "$TETRAMEM_DIR/ui/dashboard.html" ]]; then
  cp "$TETRAMEM_DIR/ui/dashboard.html" "$TETRAMEM_DIR/static/dashboard.html"
fi
ok "Configuration complete"

# --- Step 6: Start API ---
log "Step 6/7: Starting API server..."

# Kill existing
pkill -f "uvicorn start_api_v2:app" 2>/dev/null || true
sleep 2

# Create systemd service
cat > /etc/systemd/system/tetramem.service << EOF
[Unit]
Description=TetraMem-XL v6.5 Memory API
After=network.target

[Service]
Type=simple
WorkingDirectory=$TETRAMEM_DIR
Environment=TETRAMEM_STORAGE=$TETRAMEM_DIR/tetramem_data_v2
Environment=TETRAMEM_UI_PASSWORD=$TETRAMEM_PASSWORD
ExecStart=$(which python3) -m uvicorn start_api_v2:app --host 127.0.0.1 --port $TETRAMEM_PORT
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable tetramem >/dev/null 2>&1 || true
systemctl start tetramem || fail "Failed to start TetraMem service"

# Wait for startup
log "Waiting for API to initialize (first run loads lattice, ~120s)..."
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$TETRAMEM_PORT/api/v1/health" >/dev/null 2>&1; then
    break
  fi
  sleep 3
  printf "."
done
echo ""

if ! curl -sf "http://127.0.0.1:$TETRAMEM_PORT/api/v1/health" >/dev/null 2>&1; then
  fail "API did not start within 180s. Check: journalctl -u tetramem"
fi
ok "API running on port $TETRAMEM_PORT"

# --- Step 7: Nginx (optional) ---
if [[ "$TETRAMEM_NO_NGINX" != "1" ]]; then
  log "Step 7/7: Configuring nginx..."
  if command -v nginx >/dev/null 2>&1; then
    cat > /etc/nginx/sites-available/tetramem << EOF
server {
    listen $TETRAMEM_UI_PORT;
    server_name _;

    location /ui/ {
        alias $TETRAMEM_DIR/static/;
        index index.html;
        try_files \$uri \$uri/ /ui/index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:$TETRAMEM_PORT/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_read_timeout 300s;
    }

    location /libs/ {
        alias $TETRAMEM_DIR/static/;
    }

    location = / {
        return 302 /ui/;
    }
}
EOF
    ln -sf /etc/nginx/sites-available/tetramem /etc/nginx/sites-enabled/tetramem 2>/dev/null || true
    nginx -t 2>/dev/null && systemctl reload nginx 2>/dev/null || warn "Nginx config test failed, skipping"
    ok "Nginx configured on port $TETRAMEM_UI_PORT"
  else
    warn "nginx not found. Install it or set TETRAMEM_NO_NGINX=1"
    warn "API is still accessible directly on port $TETRAMEM_PORT"
  fi
else
  log "Step 7/7: Skipping nginx (--no-nginx)"
fi

# --- Done ---
PUBLIC_IP="$(curl -sf ifconfig.me 2>/dev/null || curl -sf icanhazip.com 2>/dev/null || echo '<SERVER_IP>')"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  TetraMem-XL v6.5 Installation Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  API:  ${CYAN}http://$PUBLIC_IP:$TETRAMEM_PORT/api/v1/health${NC}"
echo -e "  UI:   ${CYAN}http://$PUBLIC_IP:$TETRAMEM_UI_PORT/ui/${NC}"
echo ""
echo -e "  Username: ${YELLOW}tetramem${NC}"
echo -e "  Password: ${YELLOW}$TETRAMEM_PASSWORD${NC}"
echo ""
echo -e "  Data:    $TETRAMEM_DIR/tetramem_data_v2/"
echo -e "  Logs:    journalctl -u tetramem -f"
echo -e "  Restart: systemctl restart tetramem"
echo -e "  Stop:    systemctl stop tetramem"
echo ""
echo -e "  Standalone dashboard:"
echo -e "  ${CYAN}http://$PUBLIC_IP:$TETRAMEM_UI_PORT/ui/dashboard.html${NC}"
echo -e "  (or open ui/dashboard.html locally, enter API address above)"
echo ""
echo -e "${CYAN}Quick API test:${NC}"
echo "  curl http://127.0.0.1:$TETRAMEM_PORT/api/v1/health"
echo "  curl -X POST http://127.0.0.1:$TETRAMEM_PORT/api/v1/store -H 'Content-Type: application/json' -d '{\"content\":\"Hello TetraMem\",\"labels\":[\"test\"]}'"
echo ""
