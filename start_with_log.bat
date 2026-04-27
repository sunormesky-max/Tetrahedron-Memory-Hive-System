@echo off
cd /d C:\龙虾\Tetrahedron-Memory-Hive-System
python -m uvicorn start_api_v3:app --host 127.0.0.1 --port 8000 >> C:\龙虾\server.log 2>&1
