"""
TetraMem-XL API v3.0 — Honeycomb Neural Field
Drop-in replacement for start_api_v2.py — same endpoints, new engine underneath.
"""
import os, time, hashlib, threading, json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from tetrahedron_memory.honeycomb_neural_field import HoneycombNeuralField

STORAGE_DIR = os.environ.get("TETRAMEM_STORAGE", "./tetramem_data_v2")

_field: HoneycombNeuralField = None
_state_lock = threading.RLock()
_start_time = time.time()


def init_state():
    global _field
    _field = HoneycombNeuralField(resolution=5, spacing=1.0)
    _field.initialize()

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
        print(f"[TetraMem v3.0] Migrated {len(data.get('tetrahedra', []))} memories to honeycomb")
    else:
        print("[TetraMem v3.0] Fresh start")

    _field.start_pulse_engine()
    stats = _field.stats()
    print(f"[TetraMem v3.0] Honeycomb: {stats['total_nodes']} nodes, {stats['face_edges']} face edges, pulse engine running")


@asynccontextmanager
async def lifespan(application):
    init_state()
    yield
    _field.stop_pulse_engine()
    print("[TetraMem v3.0] Shutdown complete, pulse engine stopped")


app = FastAPI(title="TetraMem-XL v3", version="3.0.0", lifespan=lifespan)


class StoreReq(BaseModel):
    content: str
    labels: Optional[List[str]] = None
    weight: float = Field(default=1.0, ge=0.1, le=10.0)
    metadata: Optional[Dict[str, Any]] = None


class QueryReq(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=100)
    labels: Optional[List[str]] = None
    exclude_dreams: bool = False
    exclude_low_confidence: bool = True


class AssociateReq(BaseModel):
    tetra_id: str
    max_depth: int = Field(default=2, ge=1, le=5)


class TimelineReq(BaseModel):
    direction: str = "newest"
    limit: int = Field(default=20, ge=1, le=100)
    labels: Optional[List[str]] = None
    min_weight: float = Field(default=0.0, ge=0.0)


@app.post("/api/v1/store")
def store(req: StoreReq):
    try:
        with _state_lock:
            tid = _field.store(req.content, labels=req.labels, weight=req.weight, metadata=req.metadata)
        return {"id": tid}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/query")
def query(req: QueryReq):
    try:
        with _state_lock:
            results = _field.query(req.query, k=req.k, labels=req.labels)
        return {"results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/associate")
def associate(req: AssociateReq):
    try:
        with _state_lock:
            results = _field.associate(req.tetra_id, max_depth=req.max_depth)
        return {"associations": results}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/dream")
def dream():
    with _state_lock:
        status = _field.pulse_status()
    return {"result": {"phase": "continuous", "pulse_count": status["pulse_count"], "bridge_count": status["bridge_count"]}}


@app.post("/api/v1/self-organize")
def self_organize():
    with _state_lock:
        _field._check_convergence_bridges()
        _field._global_decay()
    return {"stats": _field.stats()}


@app.post("/api/v1/closed-loop")
def closed_loop():
    with _state_lock:
        _field._check_convergence_bridges()
        _field._global_decay()
    return {"result": {"phase": "pulse_active", "stats": _field.stats()}}


@app.get("/api/v1/stats")
def stats():
    with _state_lock:
        return _field.stats()


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": "3.0.0", "uptime_seconds": time.time() - _start_time}


@app.get("/api/v1/tetrahedra")
def list_tetrahedra():
    with _state_lock:
        items = _field.list_occupied()
    return {"tetrahedra": items, "count": len(items)}


@app.get("/api/v1/tetrahedra/{tetra_id}")
def get_tetra(tetra_id: str):
    with _state_lock:
        node = _field.get_node(tetra_id)
    if node is None:
        raise HTTPException(404, "Not found")
    return node


@app.delete("/api/v1/tetrahedra/{tetra_id}")
def delete_tetra(tetra_id: str):
    with _state_lock:
        node = _field._nodes.get(tetra_id)
        if node is None:
            raise HTTPException(404, "Not found")
        node.content = None
        node.labels = []
        node.weight = 0.0
        node.activation = 0.0
        node.base_activation = 0.01
        node.metadata = {}
    return {"status": "ok", "deleted": tetra_id}


@app.get("/api/v1/topology-graph")
def topology_graph():
    with _state_lock:
        occupied_ids = {nid for nid, n in _field._nodes.items() if n.is_occupied}
        nearby_ids = set(occupied_ids)
        for oid in list(occupied_ids)[:50]:
            node = _field._nodes.get(oid)
            if node:
                nearby_ids.update(node.face_neighbors)
                nearby_ids.update(node.edge_neighbors[:6])

        nodes = []
        for nid in nearby_ids:
            node = _field._nodes.get(nid)
            if node is None:
                continue
            nodes.append({
                "id": nid,
                "centroid": node.position.tolist(),
                "weight": node.weight,
                "labels": list(node.labels),
                "is_dream": "__pulse_bridge__" in node.labels,
                "activation": node.activation,
                "occupied": node.is_occupied,
                "pulse": node.pulse_accumulator,
            })

        nid_set = {n["id"] for n in nodes}
        edges = []
        for n1, n2, etype in _field._edges:
            if n1 in nid_set and n2 in nid_set:
                edges.append({"source": n1, "target": n2, "type": etype})

        return {"nodes": nodes, "edges": edges}


@app.get("/api/v1/lattice-info")
def lattice_info():
    with _state_lock:
        memories = []
        for nid, node in _field._nodes.items():
            if node.is_occupied:
                memories.append({
                    "id": nid,
                    "pos": node.position.tolist(),
                    "content": node.content,
                    "labels": list(node.labels),
                    "weight": node.weight,
                    "activation": node.activation,
                    "pulse": node.pulse_accumulator,
                    "is_dream": "__pulse_bridge__" in node.labels,
                })
        return {
            "resolution": _field._resolution,
            "spacing": _field._spacing,
            "total_nodes": len(_field._nodes),
            "memories": memories,
        }


@app.get("/api/v1/pulse-snapshot")
def pulse_snapshot():
    with _state_lock:
        pulses = []
        for nid, node in _field._nodes.items():
            if node.pulse_accumulator > 0.05 or (node.is_occupied and node.activation > 0.5):
                pulses.append({
                    "id": nid,
                    "centroid": node.position.tolist(),
                    "strength": node.pulse_accumulator if not node.is_occupied else node.activation,
                    "occupied": node.is_occupied,
                })
        return {"pulses": pulses, "count": len(pulses)}


@app.post("/api/v1/timeline")
def timeline(req: TimelineReq):
    with _state_lock:
        items = _field.browse_timeline(
            direction=req.direction,
            limit=req.limit,
            label_filter=req.labels,
            min_weight=req.min_weight,
        )
    return {"items": items, "count": len(items)}


@app.get("/api/v1/pulse-status")
def pulse_status():
    with _state_lock:
        return _field.pulse_status()


@app.post("/api/v1/export")
@app.get("/api/v1/export")
def export():
    with _state_lock:
        items = _field.list_occupied()
    lines = ["# TetraMem-XL v3.0 Memory Export\n"]
    lines.append(f"Total memories: {len(items)}\n")
    for item in items:
        labels_str = ", ".join(item.get("labels", [])) or "-"
        lines.append(f"## [{item['id'][:8]}] (w={item['weight']:.2f} a={item.get('activation',0):.2f}) [{labels_str}]")
        lines.append(item.get("content", ""))
        lines.append("")
    text = "\n".join(lines)
    out = Path(STORAGE_DIR) / "tetramem_export.md"
    out.write_text(text, encoding="utf-8")
    return {"status": "ok", "path": str(out), "size": len(text)}


@app.post("/api/v1/abstract-reorganize")
def abstract_reorganize():
    return {"result": {"phase": "pulse_handled", "stats": _field.stats()}}


@app.post("/api/v1/navigate")
def navigate():
    return {"path": []}


@app.post("/api/v1/seed-by-label")
def seed_by_label():
    return {"id": None}


@app.get("/api/v1/topology-health")
def topology_health():
    return {"result": _field.stats()}


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True))


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("TETRAMEM_HOST", "127.0.0.1")
    port = int(os.environ.get("TETRAMEM_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
