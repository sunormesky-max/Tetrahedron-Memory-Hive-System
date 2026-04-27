"""
TetraMem-XL API v3.0 - Simplified with pre-initialization
"""
import os, time, hashlib, threading, json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import uvicorn

from tetrahedron_memory.honeycomb_neural_field import HoneycombNeuralField

STORAGE_DIR = os.environ.get("TETRAMEM_STORAGE", "./tetramem_data_v2")

_field: HoneycombNeuralField = None
_start_time = time.time()

def init_state():
    global _field
    print("[TetraMem v3.0] Initializing honeycomb...")
    _field = HoneycombNeuralField(resolution=7, spacing=1.0)
    _field.initialize()
    
    json_file = Path(STORAGE_DIR) / "mesh_index.json"
    if json_file.exists():
        print(f"[TetraMem v3.0] Loading memories from {json_file}...")
        data = json.loads(json_file.read_text(encoding="utf-8"))
        items = data.get("tetrahedra", [])
        for item in items:
            content = item.get("content", "")
            if not content:
                continue
            _field.store(
                content=content,
                labels=item.get("labels", []),
                weight=item.get("weight", 1.0),
                metadata=item.get("metadata"),
            )
        print(f"[TetraMem v3.0] Loaded {len(items)} memories")
    else:
        print("[TetraMem v3.0] Fresh start - no data file")
    
    _field.start_pulse_engine()
    stats = _field.stats()
    print(f"[TetraMem v3.0] Ready: {stats['total_nodes']} nodes, {stats['occupied_nodes']} occupied, pulse engine running")

# Pre-initialize before creating app
print("[TetraMem v3.0] Starting pre-initialization...")
init_state()
print("[TetraMem v3.0] Pre-initialization complete")

app = FastAPI(title="TetraMem-XL v3", version="3.0.0")

# ... rest of the API endpoints would go here ...
# For now, just run uvicorn directly
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
