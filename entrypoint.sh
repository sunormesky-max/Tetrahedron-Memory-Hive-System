#!/bin/bash
set -e

if [ -f ./static/index.html ]; then
    sed -i "s/CHANGE_ME/${TETRAMEM_PASSWORD:-CHANGE_ME}/g" ./static/index.html
fi

exec python -m uvicorn start_api_v2:app --host 0.0.0.0 --port 8000
