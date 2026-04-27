"""Preload memories then start server"""
import json
from pathlib import Path
from tetrahedron_memory.honeycomb_neural_field import HoneycombNeuralField

# Initialize and preload
print("Loading memories...")
_field = HoneycombNeuralField(resolution=9, spacing=1.0)
_field.initialize()

STORAGE_DIR = "./tetramem_data_v2"
json_file = Path(STORAGE_DIR) / "mesh_index.json"
if json_file.exists():
    data = json.loads(json_file.read_text(encoding="utf-8"))
    for item in data.get("tetrahedra", []):
        content = item.get("content", "")
        if not content:
            continue
        _field.store(
            content=content,
            labels=item.get("labels", []),
            weight=item.get("weight", 1.0),
            metadata=item.get("metadata"),
        )
    print(f"Migrated {len(data.get('tetrahedra', []))} memories to honeycomb")

_field.start_pulse_engine()
stats = _field.stats()
print(f"Honeycomb: {stats['total_nodes']} nodes, {stats['face_edges']} face edges, pulse engine running")

# Now start uvicorn
import uvicorn
uvicorn.run("start_api_v3:app", host="127.0.0.1", port=8000, lifespan="off")
