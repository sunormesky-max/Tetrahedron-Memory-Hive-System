"""
TetraMem-XL API v6.0 — Agent-Driven Memory System + Feedback Loop + Session Management + SSE Events
"""
import os, time, hashlib, threading, json, asyncio
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from tetrahedron_memory.honeycomb_neural_field import HoneycombNeuralField
from tetrahedron_memory.phase_transition_honeycomb import HoneycombPhaseTransition

STORAGE_DIR = os.environ.get("TETRAMEM_STORAGE", "./tetramem_data_v2")

_field: HoneycombNeuralField = None
_phase_detector: HoneycombPhaseTransition = None
_state_lock = threading.RLock()
_start_time = time.time()

_event_subscribers: List[asyncio.Queue] = []
_event_subscriber_lock = threading.Lock()


def _emit_event(event_type: str, data: Dict[str, Any]):
    msg = json.dumps({"event": event_type, "data": data, "timestamp": time.time()}, ensure_ascii=False)
    with _event_subscriber_lock:
        dead = []
        for i, q in enumerate(_event_subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(i)
            except Exception:
                dead.append(i)
        for i in reversed(dead):
            _event_subscribers.pop(i)


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
            meta = item.get("metadata") or {}
            ct = meta.pop("creation_time", None)
            ct_override = float(ct) if ct else None
            _field.store(
                content=content,
                labels=item.get("labels", []),
                weight=item.get("weight", 1.0),
                metadata=meta if meta else None,
                creation_time_override=ct_override,
            )
        persist_meta = data.get("metadata", {})
        persist_time = persist_meta.get("persist_time", 0)
        from datetime import datetime as _dt
        pt_str = _dt.fromtimestamp(persist_time).strftime("%Y-%m-%d %H:%M") if persist_time else "unknown"
        print(f"[TetraMem v6.0] Migrated {len(loaded)} memories from persist file (saved at {pt_str})")
    else:
        print("[TetraMem v6.0] Fresh start")

    _phase_detector = HoneycombPhaseTransition()
    _field.start_pulse_engine()
    _field._phase_transition = _phase_detector
    stats = _field.stats()
    print(f"[TetraMem v6.0] Honeycomb: {stats['total_nodes']} nodes, {stats['face_edges']} face edges, PCNN pulse engine running")


@asynccontextmanager
async def lifespan(application):
    init_state()
    yield
    try:
        _sync_persist()
        print("[TetraMem v6.5] Final persist completed")
    except Exception as e:
        print(f"[TetraMem v6.5] Final persist failed: {e}")
    _field.stop_pulse_engine()
    print("[TetraMem v6.5] Shutdown complete, PCNN pulse engine stopped")


app = FastAPI(title="TetraMem-XL v6.5", version="6.5.0", lifespan=lifespan)


def _resolve_node(field, node_id: str) -> str:
    if node_id in field._nodes:
        return node_id
    for nid in field._nodes:
        if nid.startswith(node_id):
            return nid
    return node_id


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
    limit: int = Field(default=20, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
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
             "metadata": {**n.get("metadata", {}),
                          "creation_time": n.get("creation_time", 0.0)},
             "centroid": n.get("centroid", n.get("position", [0, 0, 0]))}
            for n in nodes
        ], "metadata": {"persist_time": time.time()}}
        path = Path(STORAGE_DIR) / "mesh_index.json"
        path.write_text(_json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[TetraMem] persisted {len(nodes)} memories to mesh_index.json")
    except Exception as e:
        print(f"[TetraMem] sync persist error: {e}")

def _auto_persist_loop():
    while True:
        _threading.Event().wait(timeout=30)
        _sync_persist()

_threading.Thread(target=_auto_persist_loop, daemon=True, name="auto-persist").start()

import atexit
atexit.register(_sync_persist)


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
        result = _field.run_self_organize()
        _sync_persist()
    return result


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
    return {"status": "ok", "version": "6.5.0", "uptime_seconds": time.time() - _start_time}


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
        items, total = _field.browse_timeline(
            direction=req.direction,
            limit=req.limit,
            offset=req.offset,
            label_filter=req.labels,
            min_weight=req.min_weight,
        )
    return {"items": items, "count": len(items), "total": total}


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


@app.post("/api/v1/feedback/record")
def feedback_record(request: dict = None):
    req = request or {}
    action = req.get("action", "")
    context_id = req.get("context_id", "")
    outcome = req.get("outcome", "neutral")
    confidence = req.get("confidence", 0.5)
    reasoning = req.get("reasoning", "")
    metadata = req.get("metadata")
    if not action or not context_id:
        raise HTTPException(400, "action and context_id are required")
    with _state_lock:
        result = _field.feedback_record(action, context_id, outcome, confidence, reasoning, metadata)
    _emit_event("feedback_recorded", {"action": action, "outcome": outcome})
    return result


@app.post("/api/v1/feedback/learn")
def feedback_learn(request: dict = None):
    req = request or {}
    action = req.get("action", "")
    source_id = req.get("source_id", "")
    target_id = req.get("target_id", "")
    success = req.get("success", True)
    confidence = req.get("confidence", 0.5)
    if not action or not source_id or not target_id:
        raise HTTPException(400, "action, source_id, and target_id are required")
    with _state_lock:
        result = _field.feedback_learn(action, source_id, target_id, success, confidence)
    if success:
        _emit_event("feedback_learned", {"action": action, "source": source_id[:12], "target": target_id[:12]})
    return result


@app.get("/api/v1/feedback/stats")
def feedback_stats():
    with _state_lock:
        return _field.feedback_stats()


@app.get("/api/v1/feedback/insights")
def feedback_insights():
    with _state_lock:
        return {"insights": _field.feedback_insights()}


@app.post("/api/v1/session/create")
def session_create(request: dict = None):
    req = request or {}
    agent_id = req.get("agent_id", "default")
    metadata = req.get("metadata")
    with _state_lock:
        session_id = _field.session_create(agent_id, metadata)
    _emit_event("session_created", {"session_id": session_id, "agent_id": agent_id})
    return {"session_id": session_id}


@app.post("/api/v1/session/{session_id}/add")
def session_add(session_id: str, request: dict = None):
    req = request or {}
    role = req.get("role", "user")
    content = req.get("content", "")
    metadata = req.get("metadata")
    with _state_lock:
        return _field.session_add(session_id, role, content, metadata)


@app.get("/api/v1/session/{session_id}/recall")
def session_recall(session_id: str, n: int = 20):
    with _state_lock:
        return _field.session_recall(session_id, n)


@app.post("/api/v1/session/{session_id}/consolidate")
def session_consolidate(session_id: str):
    with _state_lock:
        result = _field.session_consolidate(session_id)
    _emit_event("session_consolidated", {"session_id": session_id})
    return result


@app.get("/api/v1/session/list")
def session_list():
    with _state_lock:
        return {"sessions": _field.session_list()}


@app.get("/api/v1/session/{session_id}")
def session_get(session_id: str):
    with _state_lock:
        result = _field.session_get(session_id)
    if result is None:
        raise HTTPException(404, "Session not found")
    return result


@app.get("/api/v1/events")
async def events(topics: str = ""):
    topic_set = set(topics.split(",")) if topics else set()
    queue = asyncio.Queue(maxsize=100)
    with _event_subscriber_lock:
        _event_subscribers.append(queue)

    async def event_generator():
        heartbeat_count = 0
        try:
            while True:
                try:
                    msg = queue.get_nowait()
                    if topic_set:
                        try:
                            evt = json.loads(msg)
                            if evt.get("event") not in topic_set:
                                continue
                        except Exception:
                            pass
                    yield f"data: {msg}\n\n"
                except asyncio.QueueEmpty:
                    pass
                await asyncio.sleep(1)
                heartbeat_count += 1
                if heartbeat_count >= 30:
                    heartbeat_count = 0
                    yield f"data: {json.dumps({'event': 'heartbeat', 'timestamp': time.time()})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            with _event_subscriber_lock:
                try:
                    _event_subscribers.remove(queue)
                except ValueError:
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/v1/search")
def search_endpoint(request: dict = None):
    req = request or {}
    query_text = req.get("query", "")
    limit = req.get("limit", 10)
    if not query_text:
        return {"results": []}
    with _state_lock:
        results = _field.query(query_text, k=limit)
    mapped = []
    for r in results:
        mapped.append({
            "text": r.get("content", ""),
            "content": r.get("content", ""),
            "score": round(r.get("distance", 0.0), 6),
            "source": "tetramem",
            "metadata": {"id": r.get("id", ""), "weight": r.get("weight", 0), "labels": r.get("labels", [])},
        })
    return {"results": mapped}


@app.post("/api/v1/read")
def read_endpoint(request: dict = None):
    req = request or {}
    path = req.get("path", "")
    if not path:
        return {"content": "", "text": ""}
    with _state_lock:
        node = _field.get_node(path)
    if node is None:
        with _state_lock:
            results = _field.query(path, k=1)
            if results:
                r = results[0]
                return {"content": r.get("content", ""), "text": r.get("content", ""), "metadata": {"id": r.get("id", "")}}
        return {"content": "", "text": ""}
    return {"content": node.get("content", ""), "text": node.get("content", ""), "metadata": node.get("metadata", {})}


@app.get("/api/v1/status")
def status_endpoint():
    with _state_lock:
        stats = _field.stats()
    return {
        "status": "ok",
        "backend": "tetramem",
        "version": "6.5.0",
        "total_memories": stats.get("occupied_nodes", 0),
        "pulse_engine_running": stats.get("pulse_engine_running", False),
        "uptime_seconds": time.time() - _start_time,
    }


@app.post("/api/v1/sync")
def sync_endpoint(request: dict = None):
    _sync_persist()
    with _state_lock:
        count = _field.stats().get("occupied_nodes", 0)
    return {"ok": True, "synced": count, "errors": 0}


@app.get("/api/v1/capabilities/embeddings")
def capabilities_embeddings():
    return {"available": False, "model": None, "dimensions": 0, "engine": "topological"}


@app.get("/api/v1/capabilities/vectors")
def capabilities_vectors():
    return {"available": False, "engine": None, "indexSize": 0, "alternative": "topological_bfs"}


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
        result = _field.run_self_check()
    _sync_persist()
    return result


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
        result = _field.run_self_organize()
    _sync_persist()
    return result


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


@app.get("/api/v1/reflection-field/status")
def reflection_field_status():
    with _state_lock:
        if _field._reflection_field is None:
            return {"status": "not_initialized"}
        return _field._reflection_field.stats()


@app.post("/api/v1/reflection-field/run")
def reflection_field_run():
    with _state_lock:
        if _field._reflection_field is None:
            _field._reflection_field = __import__("tetrahedron_memory.honeycomb_neural_field", fromlist=["SpatialReflectionField"]).SpatialReflectionField()
        return _field._reflection_field.run_reflection_cycle(_field)


@app.get("/api/v1/reflection-field/energy/{node_id}")
def reflection_field_energy(node_id: str):
    with _state_lock:
        nid = _resolve_node(_field, node_id)
        if _field._reflection_field is None:
            return {"node_id": nid, "energy": 0.5}
        energy = _field._reflection_field.get_node_energy(nid)
        quality = _field._reflection_field.get_spatial_quality(_field, nid)
        return {"node_id": nid, "energy": round(energy, 4), "spatial_quality": round(quality, 4)}


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True))


@app.get("/api/v1/spatial/quality/{node_id}")
def spatial_quality(node_id: str):
    with _state_lock:
        nid = _resolve_node(_field, node_id)
        gq = _field._compute_node_geometric_quality(nid)
        div = _field._compute_geometric_topo_divergence(nid)
        rf_energy = _field._reflection_field.get_node_energy(nid) if _field._reflection_field else 0.5
        return {"node_id": nid, "geometric_quality": gq, "geo_topo_divergence": round(div, 4), "field_energy": rf_energy}


@app.get("/api/v1/spatial/crystallographic-direction")
def crystallographic_direction_test():
    with _state_lock:
        occupied = [(nid, n) for nid, n in _field._nodes.items() if n.is_occupied]
        if len(occupied) < 2:
            return {"error": "need at least 2 nodes"}
        a, b = occupied[0][1], occupied[1][1]
        factor = _field._bcc_direction_factor(a.position, b.position)
        dist = float(np.linalg.norm(a.position - b.position))
        return {"factor": round(factor, 4), "distance": round(dist, 4), "sample_nodes": [occupied[0][0][:8], occupied[1][0][:8]]}


@app.get("/api/v1/spatial/autocorrelation")
def spatial_autocorrelation():
    with _state_lock:
        morans_i = _field.compute_spatial_autocorrelation()
        return {
            "morans_i": round(morans_i, 4),
            "interpretation": "clustered" if morans_i > 0.1 else ("dispersed" if morans_i < -0.1 else "random"),
            "history_len": len(_field._autocorrelation_history),
        }


@app.get("/api/v1/spatial/bcc-cell-coherence/{node_id}")
def bcc_cell_coherence(node_id: str):
    with _state_lock:
        nid = _resolve_node(_field, node_id)
        coherence = _field._bcc_cell_coherence(nid)
        cellmates = _field._get_bcc_cellmates(nid)
        occ_cellmates = sum(1 for cmid in cellmates if _field._nodes.get(cmid) and _field._nodes[cmid].is_occupied)
        return {
            "node_id": nid,
            "bcc_cell_coherence": round(coherence, 4),
            "total_cellmates": len(cellmates),
            "occupied_cellmates": occ_cellmates,
        }


@app.get("/api/v1/spatial/vacancy-map")
def vacancy_map(top_n: int = 20):
    with _state_lock:
        vacancies = []
        for nid, node in _field._nodes.items():
            if not node.is_occupied:
                attraction = _field._vacancy_attraction(nid)
                if attraction > 0:
                    vacancies.append({"node_id": nid[:8], "attraction": round(attraction, 4)})
        vacancies.sort(key=lambda x: -x["attraction"])
        return {"top_vacancies": vacancies[:top_n], "total_with_attraction": len(vacancies)}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("TETRAMEM_HOST", "127.0.0.1")
    port = int(os.environ.get("TETRAMEM_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
