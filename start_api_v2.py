from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tetrahedron_memory.honeycomb_neural_field import HoneycombNeuralField
from tetrahedron_memory.auth import AuthManager
from tetrahedron_memory.input_validation import InputValidator
from tetrahedron_memory.enterprise import VersionControl, QuotaManager, BackupManager
from tetrahedron_memory.audit_log import AuditLog
from tetrahedron_memory.persistence_engine import PersistenceEngine
from tetrahedron_memory.system_ops import SystemOperationManager
from tetrahedron_memory.agent_loop import AgentMemoryLoop
from tetrahedron_memory.insight_aggregator import InsightAggregator
from tetrahedron_memory.phase_transition_honeycomb import HoneycombPhaseTransition
from tetrahedron_memory.observability import SimpleMetrics
from tetrahedron_memory.pcnn_types import PCNNConfig


log = logging.getLogger("tetramem.api")

_field: Optional[HoneycombNeuralField] = None
_state_lock: Optional[threading.RLock] = None
_persistence: Optional[PersistenceEngine] = None
_proactive_engine_stop: Optional[threading.Event] = None
_loading_complete: bool = False
_start_time: float = 0.0
_system_ops: Optional[SystemOperationManager] = None
_agent_hb_lock: Optional[threading.Lock] = None
_agent_heartbeats: Dict[str, float] = {}
_insight_aggregator: Optional[InsightAggregator] = None
_emergence_monitor = None
_phase_detector: Optional[HoneycombPhaseTransition] = None
_agent_loop: Optional[AgentMemoryLoop] = None
_event_subscribers: List = []
_event_subscriber_lock: Optional[threading.Lock] = None
STORAGE_DIR: str = ""
auth_manager: Optional[AuthManager] = None
quota_manager: Optional[QuotaManager] = None
audit_log: Optional[AuditLog] = None
backup_manager: Optional[BackupManager] = None
version_control: Optional[VersionControl] = None
metrics: Optional[SimpleMetrics] = None


def _emit_event(event_name: str, data: dict):
    msg = json.dumps({"event": event_name, **data})
    with _event_subscriber_lock:
        for q in _event_subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass


def _log_op(op: str, detail: dict):
    log.info("[OP] %s %s", op, json.dumps(detail, default=str, ensure_ascii=False)[:200])


def _auth_middleware(request: Request, call_next):
    public_paths = {
        "/", "/health", "/api/v1/health",
        "/docs", "/openapi.json", "/redoc",
    }
    if request.url.path in public_paths or request.url.path.startswith("/ui"):
        return call_next(request)
    if request.method == "OPTIONS":
        return call_next(request)
    api_key = request.headers.get("X-API-Key", "")
    bearer = request.headers.get("Authorization", "")
    token = bearer.replace("Bearer ", "") if bearer.startswith("Bearer ") else ""
    tenant = None
    if api_key:
        key_hash = auth_manager.hash_key(api_key)
        tenant = auth_manager._api_keys.get(key_hash)
    if tenant is None and token:
        payload = auth_manager.validate_token(token)
        if payload:
            tenant = {"tenant_id": payload.get("sub", "default"), "role": payload.get("role", "agent")}
    if tenant is None:
        tenant = {"tenant_id": "default", "role": "anonymous"}
    request.state.tenant = tenant
    return call_next(request)


def init_state():
    global _field, _state_lock, _persistence, _proactive_engine_stop
    global _loading_complete, _start_time, _system_ops
    global _agent_hb_lock, _agent_heartbeats, _insight_aggregator
    global _emergence_monitor, _phase_detector, _agent_loop
    global _event_subscribers, _event_subscriber_lock
    global STORAGE_DIR, auth_manager, quota_manager, audit_log
    global backup_manager, version_control, metrics

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    STORAGE_DIR = os.environ.get("TETRAMEM_STORAGE", "./tetramem_data_v2")
    os.makedirs(STORAGE_DIR, exist_ok=True)

    _start_time = time.time()
    _state_lock = threading.RLock()
    _loading_complete = False
    _proactive_engine_stop = threading.Event()

    resolution = int(os.environ.get("TETRAMEM_RESOLUTION", "5"))
    spacing = float(os.environ.get("TETRAMEM_SPACING", "1.0"))
    _field = HoneycombNeuralField(resolution=resolution, spacing=spacing)
    _field.initialize()

    persist_dir = os.path.join(STORAGE_DIR, "persistence")
    _persistence = PersistenceEngine(
        storage_dir=persist_dir,
        checkpoint_interval=int(os.environ.get("TETRAMEM_CHECKPOINT_INTERVAL", "300")),
    )
    _persistence.open()

    latest = _persistence.recover()
    if latest:
        try:
            _field.import_full_state(latest)
            n_nodes = len(latest.get("nodes", {}))
            log.info("Restored %d nodes from checkpoint", n_nodes)
            print(f"[TetraMem v7.0] Restored {n_nodes} nodes from checkpoint")
        except Exception as e:
            log.error("Checkpoint restore failed: %s", e)
            print(f"[TetraMem v7.0] Checkpoint restore failed: {e}")

    auth_manager = AuthManager(secret_key=os.environ.get("TETRAMEM_SECRET_KEY"))
    quota_manager = QuotaManager()
    audit_log = AuditLog()
    backup_manager = BackupManager(STORAGE_DIR)
    version_control = VersionControl()

    _system_ops = SystemOperationManager(
        field=_field, persistence=_persistence, storage_dir=STORAGE_DIR
    )

    _agent_hb_lock = threading.Lock()
    _agent_heartbeats = {}
    _insight_aggregator = InsightAggregator()
    _agent_loop = AgentMemoryLoop(_field)
    _phase_detector = HoneycombPhaseTransition()
    _emergence_monitor = None

    _event_subscribers = []
    _event_subscriber_lock = threading.Lock()

    metrics = SimpleMetrics()

    _field.start_pulse_engine()

    threading.Thread(target=_proactive_loop, daemon=True, name="proactive").start()
    threading.Thread(target=_session_cleanup_loop, daemon=True, name="session-cleanup").start()
    threading.Thread(target=_insight_loop, daemon=True, name="insight").start()
    threading.Thread(target=_system_health_loop, daemon=True, name="system-health").start()
    threading.Thread(target=_auto_checkpoint_loop, daemon=True, name="auto-checkpoint").start()

    _loading_complete = True
    print(f"[TetraMem v7.0] Initialized: resolution={resolution}, spacing={spacing}")


def _proactive_loop():
    cycle = 0
    while not _proactive_engine_stop.is_set():
        _proactive_engine_stop.wait(10)
        if _proactive_engine_stop.is_set() or not _loading_complete:
            continue
        cycle += 1
        try:
            if cycle % 5 == 0:
                s = _field.self_check_status() if hasattr(_field, '_self_check') and _field._self_check else {}
                anomalies = s.get("latest_check", {}).get("anomalies_found", 0)
                if anomalies > 0:
                    _emit_event("self_check_alert", {
                        "anomalies": anomalies,
                        "repairs": s.get("latest_check", {}).get("repairs_succeeded", 0),
                    })

            if cycle % 12 == 0:
                with _state_lock:
                    so = _field._self_organize.stats() if _field._self_organize else {}
                if so.get("active_shortcuts", 0) > 0:
                    _emit_event("self_organize_update", {
                        "clusters": so.get("active_clusters", 0),
                        "shortcuts": so.get("active_shortcuts", 0),
                        "entropy": so.get("latest_entropy"),
                    })

            if cycle % 30 == 0:
                now = time.time()
                with _agent_hb_lock:
                    idle_agents = [aid for aid, ts in list(_agent_heartbeats.items()) if now - ts > 60]
                    for aid in idle_agents:
                        if now - _agent_heartbeats.get(aid, 0) > 300:
                            _agent_heartbeats.pop(aid, None)
                            continue
                    stale = [aid for aid, ts in list(_agent_heartbeats.items()) if now - ts > 300]
                    for aid in stale:
                        _agent_heartbeats.pop(aid, None)
                for aid in idle_agents:
                    if aid not in _agent_heartbeats:
                        continue
                    _emit_event("agent_idle", {"agent_id": aid, "idle_seconds": int(now - _agent_heartbeats.get(aid, 0))})
        except Exception as e:
            log.error("Proactive loop error: %s", e)


def _session_cleanup_loop():
    while not _proactive_engine_stop.is_set():
        _proactive_engine_stop.wait(120)
        if _proactive_engine_stop.is_set() or not _loading_complete:
            continue
        try:
            with _state_lock:
                result = _field.session_cleanup(max_age=3600)
            if result.get("expired", 0) > 0:
                _emit_event("sessions_expired", result)
        except Exception as e:
            log.error("Session cleanup error: %s", e)


def _insight_loop():
    while not _proactive_engine_stop.is_set():
        _proactive_engine_stop.wait(30)
        if _proactive_engine_stop.is_set() or not _loading_complete:
            continue
        if _insight_aggregator is None:
            continue
        try:
            insights = _insight_aggregator.collect()
            for ins in insights:
                if ins.get("priority", 0) >= 7:
                    _emit_event("insight_high_priority", {
                        "type": ins.get("type"),
                        "title": ins.get("title"),
                        "priority": ins.get("priority"),
                        "action": ins.get("action"),
                    })
        except Exception as e:
            log.error("Insight loop error: %s", e)


def _system_health_loop():
    while not _proactive_engine_stop.is_set():
        _proactive_engine_stop.wait(60)
        if _proactive_engine_stop.is_set() or not _loading_complete:
            continue
        if _system_ops is None:
            continue
        try:
            report = _system_ops.run_health_check()
            for issue in report.get("issues", []):
                if issue in ("stale_checkpoint", "wal_too_large", "pulse_dead", "orphan_nodes", "memory_high", "disk_low"):
                    recovered = _system_ops.auto_recover(issue)
                    if recovered:
                        _emit_event("auto_recovery", {"issue": issue, "success": True})
                    else:
                        _emit_event("auto_recovery_failed", {"issue": issue, "success": False})
            if _system_ops.degradation_level > 0:
                _emit_event("degradation_alert", {
                    "level": _system_ops.degradation_level,
                    "issues": report.get("issues", []),
                })
            _system_ops.check_scheduled_backups()
        except Exception as e:
            log.error("System health loop error: %s", e)


def _auto_checkpoint_loop():
    while not _proactive_engine_stop.is_set():
        _proactive_engine_stop.wait(60)
        if _proactive_engine_stop.is_set() or not _loading_complete:
            continue
        if _persistence is not None and _persistence.should_checkpoint():
            try:
                with _state_lock:
                    state = _field.export_full_state()
                _persistence.checkpoint(state)
                log.info("Auto-checkpoint: %d memories", len(state.get("nodes", {})))
            except Exception as e:
                log.error("Auto-checkpoint failed: %s", e)


def _emergency_save(signum=None, frame=None):
    print(f"[TetraMem v7.0] Signal {signum} received — emergency save starting")
    global _field, _persistence, _state_lock, _loading_complete
    if not _loading_complete or _field is None or _persistence is None:
        return
    try:
        with _state_lock:
            state = _field.export_full_state()
        _persistence.checkpoint(state)
        n = len(state.get("nodes", {}))
        print(f"[TetraMem v7.0] Emergency save completed: {n} memories")
    except Exception as e:
        print(f"[TetraMem v7.0] Emergency save failed: {e}")
    finally:
        if _persistence is not None:
            _persistence.close()


def _register_signal_handlers():
    import signal as sig_module
    for s in (sig_module.SIGTERM, sig_module.SIGINT):
        try:
            sig_module.signal(s, _emergency_save)
        except (OSError, ValueError):
            pass


@asynccontextmanager
async def lifespan(application):
    init_state()
    _register_signal_handlers()
    yield
    try:
        if _persistence is not None and _persistence.is_dirty():
            with _state_lock:
                state = _field.export_full_state()
            _persistence.checkpoint(state)
            print("[TetraMem v7.0] Final checkpoint completed")
    except Exception as e:
        print(f"[TetraMem v7.0] Final checkpoint failed: {e}")
    finally:
        if _persistence is not None:
            _persistence.close()
    _field.stop_pulse_engine()
    _proactive_engine_stop.set()
    auth_manager.cleanup_expired()
    print("[TetraMem v7.0] Shutdown complete")


app = FastAPI(title="TetraMem-XL v7.0", version="7.0.0", lifespan=lifespan)
app.middleware("http")(_auth_middleware)


def _resolve_node(field, node_id: str) -> str:
    if node_id in field._nodes:
        return node_id
    matches = [nid for nid in field._nodes if nid.startswith(node_id)]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise HTTPException(400, f"Ambiguous ID prefix: {node_id}")
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
def store(req: StoreReq, request: Request):
    tenant = getattr(request.state, "tenant", {"tenant_id": "default", "role": "anonymous"})
    tenant_id = tenant.get("tenant_id", "default")

    content = InputValidator.sanitize_content(req.content)
    valid, err = InputValidator.validate_store(content, req.labels, req.weight, req.metadata)
    if not valid:
        metrics.increment("store_validation_errors")
        raise HTTPException(400, err)

    if not quota_manager.check_allowed(tenant_id):
        raise HTTPException(403, "Quota exceeded")

    try:
        with _state_lock:
            tid = _field.store(content, labels=req.labels, weight=req.weight, metadata=req.metadata)
        version_control.record_version(tid, content, req.weight, req.labels, req.metadata)
        quota_manager.increment(tenant_id)
        audit_log.log(tenant_id, "store", {"labels": req.labels, "weight": req.weight}, node_id=tid)
        _emit_event("memory_stored", {"id": tid, "labels": req.labels, "weight": req.weight})
        _log_op("store", {"id": tid, "content": req.content, "labels": req.labels or [], "weight": req.weight, "centroid": _field._nodes[tid].position.tolist(), "metadata": req.metadata or {}, "creation_time": float(_field._nodes[tid].creation_time)})
        if _persistence is not None:
            _persistence.log_operation("store", {"id": tid, "content": req.content, "labels": req.labels, "weight": req.weight, "metadata": req.metadata, "centroid": _field._nodes[tid].position.tolist(), "creation_time": float(_field._nodes[tid].creation_time)})
        metrics.increment("stores")
        return {"id": tid}
    except Exception as e:
        metrics.increment("store_errors")
        log.error("Store failed: %s", e)
        raise HTTPException(500, str(e))


@app.post("/api/v1/query")
def query(req: QueryReq, request: Request):
    tenant = getattr(request.state, "tenant", {"tenant_id": "default"})
    valid, err = InputValidator.validate_query(req.query, req.k)
    if not valid:
        raise HTTPException(400, err)
    try:
        t = metrics.timer("query_duration")
        with t:
            with _state_lock:
                results = _field.query(req.query, k=req.k, labels=req.labels)
        metrics.increment("queries")
        audit_log.log(tenant.get("tenant_id", "default"), "query",
                       {"k": req.k, "result_count": len(results)})
        return {"results": results}
    except Exception as e:
        metrics.increment("query_errors")
        raise HTTPException(500, str(e))


@app.post("/api/v1/batch-store")
def batch_store(req: list):
    results = []
    for item in req:
        try:
            content = item.get("content", "")
            labels = item.get("labels", [])
            weight = item.get("weight", 1.0)
            metadata = item.get("metadata", {})
            with _state_lock:
                tid = _field.store(content, labels=labels, weight=weight, metadata=metadata)
            results.append({"id": tid, "status": "ok"})
        except Exception as e:
            results.append({"content": item.get("content", "")[:50], "status": "error", "error": str(e)})
    _log_op("batch_store")
    return {"results": results}


@app.post("/api/v1/query-by-label")
def query_by_label(req: dict):
    try:
        labels = req.get("labels", [])
        k = req.get("k", 10)
        with _state_lock:
            results = _field.query("", k=k, labels=labels)
        return {"results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/query-multiparam")
def query_multiparam(req: dict):
    try:
        q = req.get("query", "")
        k = req.get("k", 10)
        labels = req.get("labels", [])
        min_weight = req.get("min_weight", 0.0)
        with _state_lock:
            results = _field.query(q, k=k, labels=labels)
            if min_weight > 0:
                results = [r for r in results if r.get("weight", 0) >= min_weight]
        return {"results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/weight-update")
def weight_update(req: dict):
    try:
        node_id = req.get("id", "")
        new_weight = req.get("weight", 1.0)
        with _state_lock:
            node = _field._nodes.get(node_id)
            if node is None:
                raise HTTPException(404, f"Node {node_id} not found")
            node.weight = float(new_weight)
            _log_op("weight_update", {"id": node_id, "weight": float(new_weight)})
        return {"id": node_id, "weight": float(new_weight), "status": "ok"}
    except HTTPException:
        raise
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
        _log_op("self_organize")
    return {"stats": result}


@app.post("/api/v1/closed-loop")
def closed_loop(request: dict = None):
    req = request or {}
    context = req.get("context", "")
    k = req.get("k", 5)
    force_dream = req.get("force_dream", False)
    result = {"phases": [], "stores": 0, "integrations": 0}
    with _state_lock:
        s_before = _field.stats()
        result["entropy_before"] = {
            "occupied": s_before["occupied_nodes"],
            "avg_activation": s_before["avg_activation"],
            "cascade_count": s_before["cascade_count"],
            "bridge_nodes": s_before["bridge_nodes"],
        }

        result["phases"].append("RECALL")
        if context:
            recall_raw = _field.query(context, k=k)
            memories = [
                {"id": r.get("id", ""), "content": r.get("content", "")[:200], "weight": r.get("weight", 0), "labels": r.get("labels", [])}
                for r in recall_raw
            ]
            recall_method = "query"
        else:
            all_occ = [(n.id, n) for n in _field._nodes.values() if n.is_occupied]
            sample_n = min(k, len(all_occ))
            if sample_n > 0:
                indices = np.random.choice(len(all_occ), size=sample_n, replace=False)
                sampled = [all_occ[i] for i in indices]
            else:
                sampled = []
            memories = [
                {"id": nid, "content": n.content[:200], "weight": n.weight, "labels": n.labels}
                for nid, n in sampled
            ]
            recall_method = "random"
        result["recall"] = {"count": len(memories), "method": recall_method}

        result["phases"].append("THINK")
        if memories:
            weights_arr = [m["weight"] for m in memories]
            avg_w = float(np.mean(weights_arr)) if weights_arr else 0
            all_labels = set()
            for m in memories:
                all_labels.update(l for l in m["labels"] if not l.startswith("__"))
            confidence = min(1.0, avg_w / 5.0)
            think = {"avg_weight": avg_w, "confidence": confidence, "patterns": list(all_labels)[:5], "memory_count": len(memories)}
        else:
            think = {"avg_weight": 0, "confidence": 0.0, "patterns": [], "memory_count": 0}
        result["think"] = think

        result["phases"].append("EXECUTE")
        execute = {"action": f"derived_from_{think['memory_count']}_memories", "confidence": think["confidence"]}
        result["execute"] = execute

        result["phases"].append("REFLECT")
        should_integrate = think["confidence"] >= 0.2
        should_dream = force_dream or think["confidence"] < 0.5
        reflect = {"quality": think["confidence"], "should_integrate": should_integrate, "should_dream": should_dream}
        result["reflect"] = reflect

        result["phases"].append("INTEGRATE")
        integration_count = 0
        if should_integrate:
            so_result = _field.run_self_organize()
            integration_count = so_result.get("merged", 0) + so_result.get("shortcuts_created", 0)
            _field._check_convergence_bridges()
            if think["confidence"] >= 0.4 and think["patterns"]:
                insight = f"Closed-loop insight: patterns={think['patterns'][:3]}, avg_weight={think['avg_weight']:.2f}"
                _field.store(content=insight, labels=["__closed_loop__", "__insight__"] + think["patterns"][:2], weight=0.3 + think["confidence"] * 0.7)
                result["stores"] += 1
        result["integrations"] = integration_count

        result["phases"].append("DREAM")
        if should_dream:
            dream_result = _field.run_dream_cycle()
            result["dream"] = {"triggered": True, "summary": {k: dream_result.get(k) for k in ["creativity", "domains", "synthesized"] if k in dream_result}}
        else:
            result["dream"] = {"triggered": False}

        s_after = _field.stats()
        result["entropy_after"] = {
            "occupied": s_after["occupied_nodes"],
            "avg_activation": s_after["avg_activation"],
            "cascade_count": s_after["cascade_count"],
            "bridge_nodes": s_after["bridge_nodes"],
        }
        result["phases"].append("COMPLETE")

    _log_op("closed_loop")
    _emit_event("closed_loop_completed", {"context": context[:100], "phases": result["phases"], "stores": result["stores"]})
    return result


@app.get("/api/v1/stats")
def stats():
    with _state_lock:
        return _field.stats()


_ui_password = os.environ.get("TETRAMEM_UI_PASSWORD", "CHANGE_ME")


@app.post("/api/v1/setup/set-password")
def set_password(body: Dict[str, str]):
    global _ui_password
    new_pw = body.get("new_password", "")
    if not new_pw or len(new_pw) < 4:
        raise HTTPException(400, "Password must be at least 4 characters")
    if _ui_password != "CHANGE_ME":
        raise HTTPException(403, "Password already set. Use environment variable TETRAMEM_UI_PASSWORD to change.")
    _ui_password = new_pw
    return {"status": "ok", "message": "Password updated for this session. Set TETRAMEM_UI_PASSWORD env var for persistence."}

@app.get("/api/v1/health")
def health():
    if _system_ops is not None:
        report = _system_ops.run_health_check()
        return {
            "status": report["status"],
            "version": "7.0.0",
            "uptime_seconds": time.time() - _start_time,
            "degradation_level": report["degradation_level"],
            "checks": {
                k: v.get("status") for k, v in report.get("checks", {}).items()
            },
            "issues": report.get("issues", []),
        }
    return {"status": "ok", "version": "7.0.0", "uptime_seconds": time.time() - _start_time}


@app.post("/api/v1/login")
def login(body: Dict[str, str] = None):
    req = body or {}
    api_key = req.get("api_key", "")
    if not api_key:
        raise HTTPException(400, "api_key is required")
    token = auth_manager.create_token(api_key)
    if token is None:
        metrics.increment("auth_failures")
        raise HTTPException(401, "Invalid API key")
    metrics.increment("auth_successes")
    return {"token": token}


@app.post("/api/v1/admin/api-key")
def create_api_key(body: Dict[str, Any] = None, request: Request = None):
    tenant = getattr(request.state, "tenant", {})
    if tenant.get("role") not in ("admin", "anonymous"):
        raise HTTPException(403, "Admin role required")
    req = body or {}
    tid = req.get("tenant_id", "default")
    name = req.get("name", "unnamed")
    role = req.get("role", "agent")
    quota = req.get("quota", 100000)
    key = auth_manager.create_api_key(tid, name, role, quota)
    quota_manager.set_quota(tid, quota)
    audit_log.log(tid, "api_key_created", {"name": name, "role": role})
    return {"api_key": key, "tenant_id": tid}


@app.get("/api/v1/metrics")
def get_metrics():
    return metrics.get_stats()


@app.post("/api/v1/backup")
def create_backup(request: Request, body: Dict[str, Any] = None):
    tenant = getattr(request.state, "tenant", {"tenant_id": "default"})
    req = body or {}
    label = req.get("label")
    backup_id = backup_manager.create_backup(label=label, storage_dir=STORAGE_DIR)
    audit_log.log(tenant.get("tenant_id", "default"), "backup_created",
                   {"backup_id": backup_id, "label": label})
    return {"backup_id": backup_id, "status": "ok"}


@app.post("/api/v1/backup/{backup_id}/restore")
def restore_backup(backup_id: str, request: Request):
    tenant = getattr(request.state, "tenant", {"tenant_id": "default"})
    success = backup_manager.restore_backup(backup_id, storage_dir=STORAGE_DIR)
    if not success:
        raise HTTPException(404, "Backup not found")
    audit_log.log(tenant.get("tenant_id", "default"), "backup_restored",
                   {"backup_id": backup_id})
    return {"status": "ok", "backup_id": backup_id}


@app.get("/api/v1/backups")
def list_backups():
    return {"backups": backup_manager.list_backups()}


@app.get("/api/v1/audit")
def query_audit(tenant_id: str = None, action: str = None,
                since: float = None, limit: int = 100):
    return {"entries": audit_log.query(tenant_id=tenant_id, action=action,
                                        since=since, limit=limit)}


@app.get("/api/v1/setup-info")
def setup_info():
    return {
        "version": "7.0.0",
        "system": "TetraMem-XL",
        "description": "BCC Lattice Honeycomb + PCNN Neural Pulse Memory System",
        "default_credentials": "Use setup/set-password to configure",
        "api_prefix": "/api/v1/",
        "ui_path": "/ui/",
        "endpoints": [
            "health", "store", "query", "tetrahedra", "stats",
            "pulse-status", "cascade/trigger", "self-check",
            "self-organize", "dream", "lattice-info", "honeycomb/analysis",
            "crystallized/status", "session", "feedback", "spatial/*",
        ],
    }


@app.get("/api/v1/tetrahedra")
def list_tetrahedra(limit: int = 200, offset: int = 0):
    with _state_lock:
        items = _field.list_occupied()
    total = len(items)
    return {"tetrahedra": items[offset:offset+limit], "count": total, "total": total, "limit": limit, "offset": offset}


@app.get("/api/v1/tetrahedra/{tetra_id}")
def get_tetra(tetra_id: str):
    with _state_lock:
        node = _field.get_node(tetra_id)
    if node is None:
        raise HTTPException(404, "Not found")
    return node


@app.delete("/api/v1/tetrahedra/{tetra_id}")
def delete_tetra(tetra_id: str, request: Request):
    tenant = getattr(request.state, "tenant", {"tenant_id": "default"})
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
    quota_manager.decrement(tenant.get("tenant_id", "default"))
    audit_log.log(tenant.get("tenant_id", "default"), "delete", {}, node_id=tetra_id)
    _log_op("delete", {"id": tetra_id})
    return {"status": "ok", "deleted": tetra_id}

@app.put("/api/v1/tetrahedra/{tetra_id}")
def update_tetra(tetra_id: str, body: dict = None, request: Request = None):
    tenant = getattr(request.state, "tenant", {"tenant_id": "default"})
    req = body or {}
    content = req.get("content")
    weight = req.get("weight")
    labels = req.get("labels")
    with _state_lock:
        node = _field._nodes.get(tetra_id)
        if node is None:
            raise HTTPException(404, "Not found")
        if content is not None:
            node.content = content
        if weight is not None:
            node.weight = max(0.1, min(10.0, float(weight)))
        if labels is not None:
            node.labels = list(labels)
        version_control.record_version(tetra_id, node.content, node.weight, node.labels, node.metadata)
    audit_log.log(tenant.get("tenant_id", "default"), "update",
                   {"weight": weight, "has_content": content is not None}, node_id=tetra_id)
    _log_op("update", {"id": tetra_id, "weight": weight, "labels": labels})
    return {"status": "ok", "updated": tetra_id}




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
                "is_dream": "__dream__" in node.labels or "__pulse_bridge__" in node.labels,
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
                    "is_dream": "__dream__" in node.labels or "__pulse_bridge__" in node.labels,
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
def abstract_reorganize(request: dict = None):
    req = request or {}
    phases = req.get("phases", ["self_organize", "bridge_check", "dream", "decay"])
    results = {}
    with _state_lock:
        if "self_organize" in phases:
            results["self_organize"] = _field.run_self_organize()
        if "bridge_check" in phases:
            _field._check_convergence_bridges()
            results["bridges"] = {"checked": True}
        if "dream" in phases:
            results["dream"] = _field.run_dream_cycle()
        if "decay" in phases:
            _field._global_decay()
            results["decay"] = {"applied": True}
        if "autocorrelation" in phases:
            ac = _field.compute_spatial_autocorrelation()
            results["spatial_autocorrelation"] = {"morans_i": ac}
        results["stats"] = _field.stats()
    _log_op("abstract_reorganize")
    return {"result": results}


@app.post("/api/v1/navigate")
def navigate(request: dict = None):
    req = request or {}
    source = req.get("source_id", "")
    target = req.get("target_id", "")
    if source and target:
        with _state_lock:
            return _field.agent_navigate(source, target)
    seed = req.get("seed_id", "")
    if seed:
        max_steps = req.get("max_steps", 30)
        with _state_lock:
            node = _field._nodes.get(seed)
            if node is None:
                return {"path": [], "nodes": [], "error": "seed_id not found"}
            visited = [seed]
            frontier = [seed]
            steps = 0
            while frontier and steps < max_steps:
                next_frontier = []
                for nid in frontier:
                    n = _field._nodes.get(nid)
                    if n is None:
                        continue
                    for fn in n.face_neighbors:
                        if fn not in visited and fn in _field._nodes:
                            visited.append(fn)
                            next_frontier.append(fn)
                    for en in n.edge_neighbors[:4]:
                        if en not in visited and en in _field._nodes:
                            visited.append(en)
                            next_frontier.append(en)
                frontier = next_frontier
                steps += 1
            path_data = []
            for nid in visited[:max_steps]:
                n = _field._nodes.get(nid)
                if n and n.is_occupied:
                    path_data.append({"id": nid, "content": n.content, "labels": list(n.labels), "weight": float(n.weight)})
            return {"path": visited[:max_steps], "nodes": path_data, "steps": steps}
    return {"path": [], "nodes": [], "error": "provide source_id+target_id or seed_id"}


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
    auto_load = req.get("auto_load_context", True)
    with _state_lock:
        session_id = _field.session_create(agent_id, metadata)
    _emit_event("session_created", {"session_id": session_id, "agent_id": agent_id})

    context = {"session_id": session_id, "auto_loaded": False, "identity": None, "preferences": []}
    if auto_load:
        try:
            with _state_lock:
                identity = _field.query("agent identity opencode", k=3)
                prefs = _field.query("用户偏好 BOSS project status", k=5)
            context["auto_loaded"] = True
            context["identity"] = identity[:2] if identity else []
            context["preferences"] = prefs[:3] if prefs else []
        except Exception:
            pass

    with _agent_hb_lock: _agent_heartbeats[agent_id] = time.time()
    return context


@app.post("/api/v1/agent/heartbeat")
def agent_heartbeat(request: dict = None):
    req = request or {}
    agent_id = req.get("agent_id", "default")
    status = req.get("status", "active")
    with _agent_hb_lock: _agent_heartbeats[agent_id] = time.time()
    if _insight_aggregator is not None:
        _insight_aggregator.register_agent(agent_id)
    idle_triggers = []
    if status == "idle":
        with _state_lock:
            s = _field.stats()
        if s.get("bridge_nodes", 0) > 10:
            idle_triggers.append("bridge_review")
        if s.get("cascade_count", 0) % 1000 < 50:
            idle_triggers.append("cascade_activity")
    return {"acknowledged": True, "agent_id": agent_id, "idle_triggers": idle_triggers}


@app.get("/api/v1/agent/status")
def agent_status():
    now = time.time()
    agents = []
    for aid, ts in _agent_heartbeats.items():
        agents.append({"agent_id": aid, "last_seen": ts, "idle_seconds": int(now - ts), "status": "active" if now - ts < 60 else "idle"})
    return {"agents": agents, "total": len(agents)}


@app.post("/api/v1/agent/{agent_id}/notifications")
def agent_notifications(agent_id: str, request: dict = None):
    req = request or {}
    unread_only = req.get("unread_only", True)
    if _insight_aggregator is None:
        return {"notifications": [], "total": 0}
    _insight_aggregator.register_agent(agent_id)
    items = _insight_aggregator.get_notifications(agent_id, unread_only=unread_only)
    return {"notifications": items, "total": len(items)}


@app.post("/api/v1/agent/{agent_id}/notifications/consume")
def agent_notifications_consume(agent_id: str, request: dict = None):
    req = request or {}
    ids = req.get("notification_ids", [])
    if _insight_aggregator is None:
        return {"consumed": 0}
    _insight_aggregator.mark_consumed(agent_id, ids)
    return {"consumed": len(ids)}


@app.post("/api/v1/agent/evolution-cycle")
def agent_evolution_cycle():
    if _agent_loop is None:
        raise HTTPException(503, "Agent loop not initialized")
    with _state_lock:
        report = _agent_loop.run_evolution_cycle()
    _emit_event("evolution_cycle_completed", {
        "cycle": report.get("cycle"),
        "quality": report.get("phases", {}).get("LEARN", {}).get("quality_score"),
        "duration": report.get("duration_seconds"),
    })
    return report


@app.get("/api/v1/agent/evolution-report")
def agent_evolution_report():
    if _agent_loop is None:
        raise HTTPException(503, "Agent loop not initialized")
    return _agent_loop.get_evolution_report()


@app.get("/api/v1/agent/proactive-suggestions")
def agent_proactive_suggestions(context: str = ""):
    if _agent_loop is None:
        raise HTTPException(503, "Agent loop not initialized")
    suggestions = _agent_loop.get_proactive_suggestions(context)
    return {"suggestions": suggestions, "count": len(suggestions)}


@app.get("/api/v1/agent/{agent_id}/recommendations")
def agent_recommendations(agent_id: str):
    if _insight_aggregator is None:
        return {"recommendations": []}
    _insight_aggregator.register_agent(agent_id)
    notifications = _insight_aggregator.get_notifications(agent_id, unread_only=True)
    recommendations = []
    for n in notifications:
        if n.get("action"):
            recommendations.append({
                "id": n.get("id"),
                "type": n.get("type"),
                "priority": n.get("priority"),
                "title": n.get("title"),
                "action": n.get("action"),
                "timestamp": n.get("timestamp"),
            })
    recommendations.sort(key=lambda x: -x.get("priority", 0))
    return {"recommendations": recommendations[:20]}


@app.post("/api/v1/proactive/trigger")
def proactive_trigger(request: dict = None):
    req = request or {}
    action = req.get("action", "dream")
    results = {}
    with _state_lock:
        if action in ("dream", "all"):
            results["dream"] = _field.run_dream_cycle()
        if action in ("self_organize", "all"):
            results["self_organize"] = _field.run_self_organize()
        if action in ("cascade", "all"):
            results["cascade"] = _field.trigger_cascade()
        if action in ("self_check", "all"):
            results["self_check"] = _field.self_check_status()
    if results:
        _emit_event("proactive_triggered", {"action": action, "results": {k: type(v).__name__ for k, v in results.items()}})
    _log_op("proactive_trigger")
    return {"action": action, "results": results}


@app.get("/api/v1/emergence/status")
def emergence_status():
    if _emergence_monitor is None:
        return {"status": "not_initialized"}
    with _state_lock:
        s = _field.stats()
    pressure_info = _emergence_monitor.compute_pressure(s)
    status = _emergence_monitor.get_status()
    return {"pressure": pressure_info, "monitor": status}


@app.post("/api/v1/emergence/trigger")
def emergence_trigger():
    if _emergence_monitor is None:
        return {"status": "not_initialized"}
    with _state_lock:
        s_before = _field.stats()
        _field.run_self_organize()
        _field.run_dream_cycle()
        s_after = _field.stats()
    effect = abs(s_after.get("occupied_nodes", 0) - s_before.get("occupied_nodes", 0))
    adj = _emergence_monitor.mark_integrated(float(effect))
    _log_op("emergence_trigger")
    return {"triggered": True, "effect": effect, "threshold_adjustment": adj}


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


@app.post("/api/v1/session/{session_id}/close")
def session_close(session_id: str):
    with _state_lock:
        result = _field.session_consolidate(session_id)
    _emit_event("session_closed", {"session_id": session_id})
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
        "version": "7.0.0",
        "total_memories": stats.get("occupied_nodes", 0),
        "pulse_engine_running": stats.get("pulse_engine_running", False),
        "uptime_seconds": time.time() - _start_time,
    }


@app.get("/api/v1/system/status")
def system_status():
    if _system_ops is None:
        raise HTTPException(503, "System ops not initialized")
    with _state_lock:
        stats = _field.stats()
    degradation = _system_ops.get_degradation_status()
    return {
        "system_health": _system_ops._last_health_check or {"status": "no_check_yet"},
        "degradation": degradation,
        "field_summary": {
            "total_memories": stats.get("occupied_nodes", 0),
            "total_nodes": stats.get("total_nodes", 0),
            "pulse_engine_running": stats.get("pulse_engine_running", False),
        },
        "uptime_seconds": time.time() - _start_time,
    }


@app.post("/api/v1/system/backup")
def system_backup(request: dict = None):
    if _system_ops is None:
        raise HTTPException(503, "System ops not initialized")
    req = request or {}
    level = req.get("level", "daily")
    if level not in ("hourly", "daily", "weekly"):
        level = "daily"
    try:
        backup_id = _system_ops.create_scheduled_backup(level)
        _emit_event("system_backup_created", {"backup_id": backup_id, "level": level})
        return {"backup_id": backup_id, "level": level, "status": "ok"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/system/restore")
def system_restore(request: dict = None):
    if _system_ops is None:
        raise HTTPException(503, "System ops not initialized")
    req = request or {}
    backup_id = req.get("backup_id", "")
    if not backup_id:
        raise HTTPException(400, "backup_id is required")
    success = _system_ops.restore_backup(backup_id)
    if not success:
        raise HTTPException(404, "Backup not found or restore failed")
    _emit_event("system_restored", {"backup_id": backup_id})
    return {"status": "ok", "backup_id": backup_id}


@app.get("/api/v1/system/integrity")
def system_integrity():
    if _system_ops is None:
        raise HTTPException(503, "System ops not initialized")
    return _system_ops.verify_data_integrity()


@app.post("/api/v1/sync")
def sync_endpoint(request: dict = None):
    try:
        with _state_lock:
            state = _field.export_full_state()
        _persistence.checkpoint(state)
    except Exception as e:
        print(f"[TetraMem v7.0] Sync checkpoint failed: {e}")
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
    _log_op("self_check")
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
    _log_op("self_organize_run")
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
        sc_data = _field.get_shortcuts(n)
        return {"shortcuts": sc_data, "count": len(sc_data)}


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


ui_dir = Path(__file__).parent / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True))
else:
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(static_dir), html=True))

libs_dir = Path(__file__).parent / "libs"
if libs_dir.exists():
    app.mount("/libs", StaticFiles(directory=str(libs_dir)))


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




@app.get("/api/v1/scene-nodes")
def scene_nodes():
    with _state_lock:
        nodes = []
        for nid, node in _field._nodes.items():
            if not node.is_occupied:
                continue
            pos = node.position
            nodes.append({
                "id": nid,
                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "w": round(float(node.weight), 2),
                "d": "__dream__" in node.labels or "__pulse_bridge__" in node.labels,
            })
        return {"nodes": nodes, "count": len(nodes)}

@app.get("/api/v1/spatial/vacancy-map")
def vacancy_map(top_n: int = 20):
    with _state_lock:
        vacancies = []
        for nid, node in _field._nodes.items():
            if not node.is_occupied:
                pos = node.position
                vacancies.append({
                    "id": nid,
                    "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "neighbors": sum(1 for nb_id in _field._connections.get(nid, []) if _field._nodes.get(nb_id) and _field._nodes[nb_id].is_occupied),
                })
        vacancies.sort(key=lambda v: -v["neighbors"])
        return {"vacancies": vacancies[:top_n], "total": len(vacancies)}


@app.post("/api/v1/query-spatial")
def query_spatial_api(req: dict):
    try:
        center = req.get("center")
        radius = float(req.get("radius", 3.0))
        k = int(req.get("k", 20))
        labels = req.get("labels")
        sort_by = req.get("sort_by", "distance")
        with _state_lock:
            results = _field.query_spatial(
                center=center, radius=radius, k=k,
                labels=labels, sort_by=sort_by,
            )
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/query-direction")
def query_direction_api(req: dict):
    try:
        direction = req.get("direction", [1, 0, 0])
        from_center = bool(req.get("from_center", True))
        max_angle = float(req.get("max_angle", 0.5))
        k = int(req.get("k", 20))
        labels = req.get("labels")
        with _state_lock:
            results = _field.query_direction(
                direction=direction, from_center=from_center,
                max_angle=max_angle, k=k, labels=labels,
            )
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/query-temporal")
def query_temporal_api(req: dict):
    try:
        time_range = req.get("time_range")
        direction = req.get("direction", "newest")
        k = int(req.get("k", 20))
        labels = req.get("labels")
        min_weight = float(req.get("min_weight", 0.0))
        lifecycle_stage = req.get("lifecycle_stage")
        with _state_lock:
            results = _field.query_temporal(
                time_range=time_range, direction=direction,
                k=k, labels=labels, min_weight=min_weight,
                lifecycle_stage=lifecycle_stage,
            )
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/temporal-sequence/{node_id}")
def temporal_sequence_api(node_id: str, direction: str = "forward", max_depth: int = 10):
    try:
        with _state_lock:
            results = _field.query_temporal_sequence(node_id, direction=direction, max_depth=max_depth)
        return {"sequence": results, "length": len(results)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/lifecycle-stats")
def lifecycle_stats_api():
    with _state_lock:
        return _field.get_lifecycle_stats()


@app.post("/api/v1/dark-plane/flow")
def dark_plane_flow_api():
    try:
        with _state_lock:
            result = _field.dark_plane_flow()
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/dark-plane/stats")
def dark_plane_stats_api():
    with _state_lock:
        return _field.get_lifecycle_stats()


@app.post("/api/v1/attention/focus")
def attention_focus_api(req: dict):
    try:
        with _state_lock:
            result = _field.attention_set_focus(
                center=req.get("center"),
                radius=float(req.get("radius", 5.0)),
                strength=float(req.get("strength", 1.0)),
                labels=req.get("labels"),
                query_text=req.get("query_text"),
            )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/attention/clear")
def attention_clear_api():
    with _state_lock:
        _field.attention_clear()
    return {"status": "ok"}


@app.get("/api/v1/attention/status")
def attention_status_api():
    with _state_lock:
        return _field.attention_status()


@app.get("/api/v1/emergence/quality")
def emergence_quality_api():
    with _state_lock:
        return _field.compute_emergence_quality()


@app.get("/api/v1/emergence/history")
def emergence_history_api(n: int = 20):
    with _state_lock:
        history = _field._emergence_history[-n:]
        return {"count": len(history), "history": history}


@app.get("/api/v1/regulation/status")
def regulation_status_api():
    with _state_lock:
        if _field._self_regulation is None:
            return {"active": False}
        return {"active": True, **_field._self_regulation.status()}


@app.post("/api/v1/regulation/trigger")
def regulation_trigger_api():
    with _state_lock:
        if _field._self_regulation is None:
            raise HTTPException(400, "Self-regulation not initialized")
        record = _field._self_regulation.regulate()
        return record


@app.post("/api/v1/regulation/force-mode")
def regulation_force_mode_api(req: dict):
    with _state_lock:
        if _field._self_regulation is None:
            raise HTTPException(400, "Self-regulation not initialized")
        mode = req.get("mode", "")
        _field._self_regulation.force_mode(mode)
        return {"status": "ok", "mode": mode}


@app.get("/api/v1/regulation/history")
def regulation_history_api(n: int = 20):
    with _state_lock:
        if _field._self_regulation is None:
            return {"count": 0, "history": []}
        history = _field._self_regulation.get_history(n)
        return {"count": len(history), "history": history}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("TETRAMEM_PORT", "8000"))
    host = os.environ.get("TETRAMEM_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, log_level="info")