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


@asynccontextmanager
async def lifespan(application):
    init_state()
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
        metrics.increment("stores")
        return {"id": tid}
    except Exception as e:
        metrics.increment("store_errors")
        log.error("Store failed: %s", e)
        raise HTTPException(500, str(e))


@app.post("/api/v1/query")
def query(req: QueryReq, request: Request):
    tenant = getatt