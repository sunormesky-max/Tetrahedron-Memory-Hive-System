#!/bin/bash
set -e

VERSION="2.1.0"
TETRAMEM_STORAGE="${TETRAMEM_STORAGE:-/data/tetramem}"
RAY_WORKERS="${RAY_WORKERS:-4}"
API_PORT="${API_PORT:-8000}"
API_WORKERS="${API_WORKERS:-4}"
MONITOR_PORT="${MONITOR_PORT:-9090}"

echo "=== TetraMem-XL v${VERSION} Production Deployment ==="
echo "Storage: ${TETRAMEM_STORAGE}"
echo "Ray Workers: ${RAY_WORKERS}"
echo "API: 0.0.0.0:${API_PORT} (${API_WORKERS} workers)"
echo ""

if [ -f .env ]; then
  export $(cat .env | xargs)
fi

mkdir -p "${TETRAMEM_STORAGE}"

echo "[1/5] Running production tests..."
python -m pytest tests/ -q --ignore=tests/test_stress_1m.py -k "not TestRouterEndpoints" \
  --tb=short -x || {
    echo "ERROR: Production tests failed. Aborting deployment."
    exit 1
  }

echo "[2/5] Starting Ray cluster..."
if command -v ray &> /dev/null; then
  ray up cluster.yaml --yes 2>/dev/null || {
    echo "Ray cluster config not found, starting local..."
    ray start --head --num-cpus=${RAY_WORKERS} --port=6379 || true
  }
else
  echo "Ray not installed, using local mode."
fi

echo "[3/5] Starting API server..."
nohup uvicorn start_api_persisted:app \
  --host 0.0.0.0 \
  --port ${API_PORT} \
  --workers ${API_WORKERS} \
  > "${TETRAMEM_STORAGE}/api.log" 2>&1 &
API_PID=$!
echo "API PID: ${API_PID}"

sleep 3

echo "[4/5] Health check..."
HEALTH=$(curl -s http://localhost:${API_PORT}/api/v1/health 2>/dev/null || echo "failed")
if echo "${HEALTH}" | grep -q "ok"; then
  echo "Health check passed."
else
  echo "WARNING: Health check returned: ${HEALTH}"
fi

TOPOLOGY=$(curl -s http://localhost:${API_PORT}/api/v1/health/topology 2>/dev/null || echo "failed")
echo "Topology: ${TOPOLOGY}" | head -c 200
echo ""

echo "[5/5] Starting monitoring..."
if command -v prometheus &> /dev/null; then
  nohup prometheus --config.file=prometheus.yml \
    --storage.tsdb.path="${TETRAMEM_STORAGE}/prometheus" \
    --web.listen-address="0.0.0.0:${MONITOR_PORT}" \
    > "${TETRAMEM_STORAGE}/prometheus.log" 2>&1 &
  echo "Prometheus PID: $!"
fi

echo ""
echo "=== Deployment Complete ==="
echo "API:        http://localhost:${API_PORT}"
echo "API Docs:   http://localhost:${API_PORT}/docs"
echo "Health:     http://localhost:${API_PORT}/api/v1/health"
echo "Topology:   http://localhost:${API_PORT}/api/v1/health/topology"
echo "Metrics:    http://localhost:${API_PORT}/metrics"
echo "Storage:    ${TETRAMEM_STORAGE}"
echo "Log:        ${TETRAMEM_STORAGE}/api.log"
echo ""
echo "To stop: kill ${API_PID}"
