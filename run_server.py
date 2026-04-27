"""Run TetraMem server with preloaded state"""
import sys
sys.path.insert(0, r'C:\龙虾\Tetrahedron-Memory-Hive-System')

from start_api_v3 import init_state, app
import uvicorn

# Initialize (blocking)
init_state()

# Start server (blocking)
uvicorn.run(app, host="127.0.0.1", port=8000)
