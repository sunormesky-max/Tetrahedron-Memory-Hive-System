@echo off
cd /d C:\龙虾\Tetrahedron-Memory-Hive-System
python -c "from start_api_v3 import init_state, app; init_state(); import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000)"
