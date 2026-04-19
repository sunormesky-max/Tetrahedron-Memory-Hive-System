"""
TetraMem-XL API v5.3 �?Tetrahedral Cell Decomposition + Honeycomb Structural Analysis + Enhanced Memory Placement
"""
import os, time, hashlib, threading, json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from tetrahedron_memory.honeycomb_neural_field import HoneycombNeuralField
from tetrahedron_memory.phase_transition_honeycomb import HoneycombPhaseTransition

STORAGE_DIR = os.environ.get("TETRAMEM_STORAGE", "./tetramem_data_v2")

_field: HoneycombNeuralField = None
_phase_detector: HoneycombPhaseTransition = None
_state_lock = threading.RLock()
_start_time = time.time()


def init_state():
    global _field, _phase_detector
    _field = HoneycombNeuralField(resolution=5, spacing=1.0)
    _field.initialize()

    json_file = Path(STORAGE_DIR) / "mesh_index.json"
    if json_file.exists():
        data = json.loads(json_file.read_text(encoding="utf-8"))
        loaded = data.get("tetrahedra", [])
        for item in loaded:
            content = item.get("content", "")
            if not content:
                continue
            _field.store(
                content=content,
                labels=item.get("labels", []),
                weight=item.get("weight", 1.0),
                metadata=item.get("metadata"),
            )
        persist_meta = data.get("metadata", {})
        persist_time = persist_meta.get("persist_time", 0)
        from datetime import datetime as _dt
        pt_str = _dt.fromtimestamp(persist_time).strftime("%Y-%m-%d %H:%M") if persist_time else "unknown"
        print(f"[TetraMem v5.3] Migrated {len(loaded)} memories from persist file (saved at {pt_str})")
    else:
        print("[TetraMem v5.3] Fresh start")

    _phase_detector = HoneycombPhaseTransition()
    _field.start_pulse_engine()
    stats = _field.stats()
    print(f"[TetraMem v5.3] Honeycomb: {stats['total_nodes']} nodes, {stats['face_edges']} face edges, PCNN pulse engine running")


@asynccontextmanager
async def lifespan(application):
    init_state()
    yield
    _field.stop_pulse_engine()
    print("[TetraMem v5.3] Shutdown complete, PCNN pulse engine stopped")


app = FastAPI(title="TetraMem-XL v5.3", version="5.3.0", lifespan=lifespan)


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
        _sync_persist()
        return {"id": tid}
    except Exception as e:
        raise HTTPException(500, str(e))


import threading as _threading

def _sync_persist():
    try:
        with _state_lock:
            field = _field
        if field is None:
            return
        nodes = field.list_occupied()
        import json as _json
        export = {"tetrahedra": [
            {"id": n["id"], "content": n.get("content", ""),
             "labels": n.get("labels", []), "weight": n.get("weight", 1.0),
             "metadata": n.get("metadata", {}),
             "centroid": n.get("centroid", n.get("position", [0, 0, 0]))}
            for n in nodes
        ], "metadata": {"persist_time": time.time()}}
        path = Path(STORAGE_DIR) / "mesh_index.json"
        path.write_text(_json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[TetraMem] sync persist error: {e}")

def _auto_persist_loop():
    while True:
        _threading.Event().wait(timeout=30)
        _sync_persist()

_threading.Thread(target=_auto_persist_loop, daemon=True, name="auto-persist").start()


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
        result = _field.run_dream_cycle()
    return {"result": result}





@app.get("/api/v1/phase-transition/status")
def phase_status():
    try:
        with _state_lock:
            detector = _phase_detector
        if detector is None:
            return {"status": "not_initialized"}
        gt, tensions = detector.compute_global_tension(_field)
        return {
            "status": "ok",
            "global_tension": round(gt, 3),
            "nodes_with_tension": len(tensions),
            "trend": detector.get_tension_trend(),
            "total_transitions": detector._transition_count,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/phase-transition/trigger")
def phase_trigger():
    try:
        with _state_lock:
            detector = HoneycombPhaseTransition(tension_threshold=0.0, cooldown_seconds=0)
            global_tension, tensions = detector.compute_global_tension(_field)
            clusters = detector.identify_tension_clusters(tensions, _field)
        if not clusters:
            return {"status": "no_tension", "global_tension": global_tension}
        result = detector.execute_transition(_field, tensions, clusters)
        return {"status": "transition_complete", "result": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/tension")
def tension_map():
    try:
        with _state_lock:
            field = _field
        nodes = field.list_occupied()
        if not nodes:
            return {"tensions": {}, "global": 0, "trend": "no_data"}
        tensions = {}
        for n in nodes:
            nid = n.get("id", "")
            w = n.get("weight", 0)
            labels = n.get("labels", [])
            act = n.get("activation", 0)
            neighbors = n.get("face_neighbors", 0)
            if not isinstance(neighbors, int):
                neighbors = len(neighbors) if neighbors else 0
            tension = max(0, (1.0 - neighbors / 7.0) * w + act * 0.5)
            if "__dream__" in labels:
                tension *= 0.3
            tensions[nid] = tension
        global_tension = sum(tensions.values())
        top = sorted(tensions.items(), key=lambda x: -x[1])[:20]

        with _state_lock:
            detailed = field.get_tension_map(top_n=10)
        return {
            "global_tension": round(global_tension, 3),
            "top_tension_nodes": [{"id": tid[:12], "tension": round(t, 3)} for tid, t in top],
            "detailed_tension": detailed,
            "total_scored": len(tensions),
        }
    except Exception as e:
        raise HTTPException(500, str(e))

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
    return {"status": "ok", "version": "5.3.0", "uptime_seconds": time.time() - _start_time}


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
def navigate(request: dict = None):
    req = request or {}
    source = req.get("source_id", "")
    target = req.get("target_id", "")
    if not source or not target:
        return {"path": [], "error": "source_id and target_id required"}
    with _state_lock:
        return _field.agent_navigate(source, target)


@app.post("/api/v1/seed-by-label")
def seed_by_label(request: dict = None):
    req = request or {}
    labels = req.get("labels", [])
    if not labels:
        return {"id": None}
    with _state_lock:
        results = _field.query(" ".join(labels), k=1, labels=labels)
        if results:
            return {"id": results[0]["id"]}
        return {"id": None}


@app.get("/api/v1/dream/status")
def dream_status():
    with _state_lock:
        return _field.dream_status()


@app.get("/api/v1/dream/history")
def dream_history(n: int = 10):
    with _state_lock:
        return {"history": _field.dream_history(n)}


@app.post("/api/v1/agent/context")
def agent_context(request: dict = None):
    req = request or {}
    topic = req.get("topic", "")
    max_mem = req.get("max_memories", 15)
    if not topic:
        raise HTTPException(400, "topic is required")
    with _state_lock:
        return _field.agent_get_context(topic, max_mem)


@app.post("/api/v1/agent/reasoning")
def agent_reasoning(request: dict = None):
    req = request or {}
    source_id = req.get("source_id", "")
    target_query = req.get("target_query", "")
    if not source_id or not target_query:
        return {"chain": [], "error": "source_id and target_query are required"}
    with _state_lock:
        return _field.agent_reasoning_chain(source_id, target_query, req.get("max_hops", 5))


@app.post("/api/v1/agent/suggest")
def agent_suggest(request: dict = None):
    req = request or {}
    context = req.get("context", "")
    with _state_lock:
        return _field.agent_suggest(context)


@app.get("/api/v1/topology-health")
def topology_health():
    return {"result": _field.stats()}


@app.get("/api/v1/pcnn/states")
def pcnn_states():
    with _state_lock:
        return {"states": _field.get_pcnn_node_states()}


@app.get("/api/v1/pcnn/tension-map")
def pcnn_tension():
    with _state_lock:
        return {"tension_map": _field.get_tension_map()}


@app.get("/api/v1/pcnn/hebbian")
def hebbian_paths():
    with _state_lock:
        ps = _field.pulse_status()
        return {
            "hebbian": ps.get("hebbian", {}),
            "top_paths": ps.get("hebbian_top_paths", []),
        }


@app.get("/api/v1/pcnn/config")
def pcnn_config():
    from tetrahedron_memory.honeycomb_neural_field import PCNNConfig
    return {
        "face_decay": PCNNConfig.FACE_DECAY,
        "edge_decay": round(PCNNConfig.FACE_DECAY * PCNNConfig.EDGE_DECAY_FACTOR, 3),
        "beta": PCNNConfig.BETA,
        "alpha_feed": PCNNConfig.ALPHA_FEED,
        "alpha_link": PCNNConfig.ALPHA_LINK,
        "alpha_threshold": PCNNConfig.ALPHA_THRESHOLD,
        "max_hops_exploratory": PCNNConfig.MAX_HOPS_EXPLORATORY,
        "max_hops_reinforcing": PCNNConfig.MAX_HOPS_REINFORCING,
        "max_hops_tension": PCNNConfig.MAX_HOPS_TENSION,
        "self_check_max_hops": PCNNConfig.SELF_CHECK_MAX_HOPS,
        "bridge_threshold": PCNNConfig.BRIDGE_THRESHOLD,
        "pulse_type_probabilities": {t.value: p for t, p in PCNNConfig.PULSE_TYPE_PROBABILITIES.items()},
    }


@app.get("/api/v1/self-check/status")
def self_check_status():
    with _state_lock:
        return _field.self_check_status()


@app.post("/api/v1/self-check/run")
def self_check_run():
    with _state_lock:
        return _field.run_self_check()


@app.get("/api/v1/self-check/history")
def self_check_history(n: int = 10):
    with _state_lock:
        return {"history": _field.self_check_history(n)}


@app.get("/api/v1/duplicates")
def detect_duplicates():
    with _state_lock:
        dupes = _field.detect_duplicates()
    return {"duplicates": dupes, "count": len(dupes)}


@app.get("/api/v1/isolated")
def detect_isolated():
    with _state_lock:
        isolated = _field.detect_isolated()
    return {"isolated": isolated, "count": len(isolated)}


@app.get("/api/v1/lattice-integrity/check")
def lattice_integrity_check():
    with _state_lock:
        return _field.run_lattice_check()


@app.get("/api/v1/lattice-integrity/status")
def lattice_integrity_status():
    with _state_lock:
        return _field.lattice_check_status()


@app.get("/api/v1/lattice-integrity/history")
def lattice_integrity_history(n: int = 10):
    with _state_lock:
        return {"history": _field.lattice_check_history(n)}


@app.get("/api/v1/crystallized/status")
def crystallized_status():
    with _state_lock:
        return _field.crystallized_status()


@app.post("/api/v1/cascade/trigger")
def cascade_trigger(request: dict = None):
    req = request or {}
    strength = req.get("strength", 0.5)
    source_id = req.get("source_id")
    with _state_lock:
        return _field.trigger_cascade(source_id=source_id, strength=strength)


@app.post("/api/v1/structure-pulse/trigger")
def structure_pulse_trigger(request: dict = None):
    req = request or {}
    source_id = req.get("source_id")
    with _state_lock:
        return _field.trigger_structure_pulse(source_id=source_id)


@app.post("/api/v1/crystallized/force")
def force_crystallize():
    with _state_lock:
        return _field.force_crystallize()


@app.post("/api/v1/self-organize/run")
def self_organize_run():
    with _state_lock:
        return _field.run_self_organize()


@app.get("/api/v1/self-organize/status")
def self_organize_status():
    with _state_lock:
        return _field.self_organize_status()


@app.get("/api/v1/self-organize/history")
def self_organize_history(n: int = 10):
    with _state_lock:
        return {"history": _field.self_organize_history(n)}


@app.get("/api/v1/clusters")
def get_clusters():
    with _state_lock:
        return {"clusters": _field.get_clusters()}


@app.get("/api/v1/shortcuts")
def get_shortcuts(n: int = 20):
    with _state_lock:
        return {"shortcuts": _field.get_shortcuts(n), "count": len(_field.get_shortcuts(n))}


@app.get("/api/v1/honeycomb/analysis")
def honeycomb_analysis():
    with _state_lock:
        return _field.honeycomb_analysis()


@app.get("/api/v1/honeycomb/cells")
def honeycomb_cells(n: int = 20, sort_by: str = "quality"):
    with _state_lock:
        return {"cells": _field.get_tetrahedral_cells(n, sort_by), "count": n}


@app.get("/api/v1/honeycomb/cells/{node_id}")
def honeycomb_cells_for_node(node_id: str):
    with _state_lock:
        cells = _field.get_cell_for_node(node_id)
        return {"node_id": node_id, "cells": cells, "count": len(cells)}


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True))


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("TETRAMEM_HOST", "127.0.0.1")
    port = int(os.environ.get("TETRAMEM_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
