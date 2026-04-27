@echo off
cd /d C:\龙虾\Tetrahedron-Memory-Hive-System
python -m uvicorn start_api_v3:app --host 127.0.0.1 --port 8000 --timeout-keep-alive 300 --limit-concurrency 10
