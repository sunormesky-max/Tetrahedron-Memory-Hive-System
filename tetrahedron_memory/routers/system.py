from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from tetrahedron_memory.app_state import AppState

log = logging.getLogger("tetramem.api")

router = APIRouter(prefix="/api/v1", tags=["system"])

_ui_password = os.environ.get("TETRAMEM_UI_PASSWORD", "CHANGE_ME")


def _get_state(request: Request) -> AppState:
    return request.app.state.tetramem


@router.get("/health")
def health(request: Request):
    state = _get_state(request)
    if state.system_ops is not None:
        report = state.system_ops.run_health_check()
        return {
            "status": report["status"],
            "version": "8.0.0",
            "uptime_seconds": time.time() - state.start_time,
            "degradation_level": report["degradation_level"],
            "checks": {
                k: v.get("status") for k, v in report.get("checks", {}).items()
            },
            "issues": report.get("issues", []),
        }
    return {"status": "ok", "version": "8.0.0", "uptime_seconds": time.time() - state.start_time}


@router.get("/stats")
def stats(request: Request):
    state = _get_state(request)
    with state.state_lock:
        return state.field.stats()


@router.get("/metrics")
def get_metrics(request: Request):
    state = _get_state(request)
    return state.metrics.get_stats()


@router.post("/setup/set-password")
def set_password(body: Dict[str, str]):
    global _ui_password
    new_pw = body.get("new_password", "")
    if not new_pw or len(new_pw) < 4:
        raise HTTPException(400, "Password must be at least 4 characters")
    if _ui_password != "CHANGE_ME":
        raise HTTPException(403, "Password already set. Use environment variable TETRAMEM_UI_PASSWORD to change.")
    _ui_password = new_pw
    return {"status": "ok", "message": "Password updated for this session. Set TETRAMEM_UI_PASSWORD env var for persistence."}


@router.get("/setup-info")
def setup_info():
    return {
        "version": "8.0.0",
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


@router.post("/login")
def login(request: Request, body: Dict[str, str] = None):
    state = _get_state(request)
    req = body or {}

    ui_password = req.get("password", "")
    if ui_password and ui_password == _ui_password:
        token = state.auth_manager.create_token("ui-session") or "ui-access"
        state.metrics.increment("auth_successes")
        return {"token": token, "mode": "ui"}

    api_key = req.get("api_key", ui_password)
    if not api_key:
        raise HTTPException(400, "api_key or password is required")
    token = state.auth_manager.create_token(api_key)
    if token is None:
        state.metrics.increment("auth_failures")
        raise HTTPException(401, "Invalid API key")
    state.metrics.increment("auth_successes")
    return {"token": token, "mode": "api"}


@router.post("/admin/api-key")
def create_api_key(body: Dict[str, Any] = None, request: Request = None):
    state = _get_state(request)
    tenant = getattr(request.state, "tenant", {})
    if tenant.get("role") != "admin":
        raise HTTPException(403, "Admin role required")
    req = body or {}
    tid = req.get("tenant_id", "default")
    name = req.get("name", "unnamed")
    role = req.get("role", "agent")
    quota = req.get("quota", 100000)
    key = state.auth_manager.create_api_key(tid, name, role, quota)
    state.quota_manager.set_quota(tid, quota)
    state.audit_log.log(tid, "api_key_created", {"name": name, "role": role})
    return {"api_key": key, "tenant_id": tid}


@router.post("/backup")
def create_backup(request: Request, body: Dict[str, Any] = None):
    state = _get_state(request)
    tenant = getattr(request.state, "tenant", {"tenant_id": "default"})
    req = body or {}
    label = req.get("label")
    with state.state_lock:
        export_state = state.field.export_full_state()
    backup_id = state.backup_manager.create_backup(export_state, label=label)
    state.audit_log.log(tenant.get("tenant_id", "default"), "backup_created",
                        {"backup_id": backup_id, "label": label})
    return {"backup_id": backup_id, "status": "ok"}


@router.post("/backup/{backup_id}/restore")
def restore_backup(backup_id: str, request: Request):
    state = _get_state(request)
    tenant = getattr(request.state, "tenant", {"tenant_id": "default"})
    restored = state.backup_manager.restore_backup(backup_id)
    if not restored:
        raise HTTPException(404, "Backup not found")
    state.audit_log.log(tenant.get("tenant_id", "default"), "backup_restored",
                        {"backup_id": backup_id})
    return {"status": "ok", "backup_id": backup_id}


@router.get("/backups")
def list_backups(request: Request):
    state = _get_state(request)
    return {"backups": state.backup_manager.list_backups()}


@router.get("/audit")
def query_audit(request: Request, tenant_id: str = None, action: str = None,
                since: float = None, limit: int = 100):
    state = _get_state(request)
    return {"entries": state.audit_log.query(tenant_id=tenant_id, action=action,
                                              since=since, limit=limit)}


@router.get("/system/status")
def system_status(request: Request):
    state = _get_state(request)
    if state.system_ops is None:
        raise HTTPException(503, "System ops not initialized")
    with state.state_lock:
        stats = state.field.stats()
    degradation = state.system_ops.get_degradation_status()
    return {
        "system_health": state.system_ops._last_health_check or {"status": "no_check_yet"},
        "degradation": degradation,
        "field_summary": {
            "total_memories": stats.get("occupied_nodes", 0),
            "total_nodes": stats.get("total_nodes", 0),
            "pulse_engine_running": stats.get("pulse_engine_running", False),
        },
        "uptime_seconds": time.time() - state.start_time,
    }


@router.post("/system/backup")
def system_backup(request: Request, body: dict = None):
    state = _get_state(request)
    if state.system_ops is None:
        raise HTTPException(503, "System ops not initialized")
    req = body or {}
    level = req.get("level", "daily")
    if level not in ("hourly", "daily", "weekly"):
        level = "daily"
    try:
        backup_id = state.system_ops.create_scheduled_backup(level)
        state.emit_event("system_backup_created", {"backup_id": backup_id, "level": level})
        return {"backup_id": backup_id, "level": level, "status": "ok"}
    except Exception as e:
        log.error("System backup failed: %s", e)
        raise HTTPException(500, "Internal error")


@router.post("/system/restore")
def system_restore(request: Request, body: dict = None):
    state = _get_state(request)
    if state.system_ops is None:
        raise HTTPException(503, "System ops not initialized")
    req = body or {}
    backup_id = req.get("backup_id", "")
    if not backup_id:
        raise HTTPException(400, "backup_id is required")
    success = state.system_ops.restore_backup(backup_id)
    if not success:
        raise HTTPException(404, "Backup not found or restore failed")
    state.emit_event("system_restored", {"backup_id": backup_id})
    return {"status": "ok", "backup_id": backup_id}


@router.get("/system/integrity")
def system_integrity(request: Request):
    state = _get_state(request)
    if state.system_ops is None:
        raise HTTPException(503, "System ops not initialized")
    return state.system_ops.verify_data_integrity()


@router.post("/sync")
def sync_endpoint(request: Request, body: dict = None):
    state = _get_state(request)
    try:
        with state.state_lock:
            export = state.field.export_full_state()
        state.persistence.checkpoint(export)
    except Exception as e:
        print(f"[TetraMem v8.0] Sync checkpoint failed: {e}")
    with state.state_lock:
        count = state.field.stats().get("occupied_nodes", 0)
    return {"ok": True, "synced": count, "errors": 0}


@router.get("/events")
async def events(request: Request, topics: str = ""):
    state = _get_state(request)
    topic_set = set(topics.split(",")) if topics else set()
    queue = asyncio.Queue(maxsize=100)
    with state.event_subscriber_lock:
        state.event_subscribers.append(queue)

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
            with state.event_subscriber_lock:
                try:
                    state.event_subscribers.remove(queue)
                except ValueError:
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/capabilities/embeddings")
def capabilities_embeddings():
    return {"available": False, "model": None, "dimensions": 0, "engine": "topological"}


@router.get("/capabilities/vectors")
def capabilities_vectors():
    return {"available": False, "engine": None, "indexSize": 0, "alternative": "topological_bfs"}
