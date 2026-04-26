import asyncio
import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Any, Dict, List, Optional

from tetrahedron_memory.app_state import AppState

log = logging.getLogger("tetramem.api")

router = APIRouter(prefix="/api/v1", tags=["observer"])

_trajectory_subscribers: list = []


def _get_state(request: Request) -> AppState:
    return request.app.state.tetramem


def _get_observer(request: Request):
    state = _get_state(request)
    obs = getattr(state.field, "_runtime_observer", None)
    if obs is None:
        raise HTTPException(400, "RuntimeObserver not initialized")
    return obs


def _store_hook(narration: str, category: str, weight: float, metadata: Dict[str, Any]) -> None:
    msg = json.dumps({
        "event": "trajectory_stored",
        "category": category,
        "weight": round(weight, 3),
        "narration": narration[:200],
        "timestamp": time.time(),
    }, ensure_ascii=False)
    for q in list(_trajectory_subscribers):
        try:
            q.put_nowait(msg)
        except Exception:
            pass


@router.get("/observer/stats")
def observer_stats(request: Request):
    obs = _get_observer(request)
    return obs.get_stats()


@router.post("/observer/flush")
def observer_flush(request: Request):
    obs = _get_observer(request)
    stored = obs.flush_all()
    return {"flushed": stored}


@router.post("/observer/enable")
def observer_enable(request: Request):
    obs = _get_observer(request)
    obs.set_enabled(True)
    return {"enabled": True}


@router.post("/observer/disable")
def observer_disable(request: Request):
    obs = _get_observer(request)
    obs.set_enabled(False)
    return {"enabled": False}


@router.post("/observer/observe")
def observer_manual_observe(req: dict, request: Request):
    level = req.get("level", "INFO")
    module = req.get("module", "manual")
    message = req.get("message", "")
    source = req.get("source")
    metadata = req.get("metadata")
    if not message:
        raise HTTPException(400, "message is required")
    obs = _get_observer(request)
    accepted = obs.observe(
        level=level, module=module, message=message, source=source,
        metadata=metadata,
    )
    return {"accepted": accepted}


@router.post("/observer/batch")
def observer_batch(req: dict, request: Request):
    events = req.get("events", [])
    if not events:
        raise HTTPException(400, "events array is required")
    if len(events) > 100:
        raise HTTPException(400, "batch size limited to 100 events")

    obs = _get_observer(request)
    results = {"accepted": 0, "rejected": 0, "errors": []}
    for evt in events:
        message = evt.get("message", "")
        if not message:
            results["rejected"] += 1
            continue
        accepted = obs.observe(
            level=evt.get("level", "INFO"),
            module=evt.get("module", "external-agent"),
            message=message,
            source=evt.get("source", "agent-batch"),
            metadata=evt.get("metadata"),
        )
        if accepted:
            results["accepted"] += 1
        else:
            results["rejected"] += 1

    return results


@router.post("/observer/ingest-json")
def observer_ingest_json(req: dict, request: Request):
    lines = req.get("lines", [])
    if not lines:
        raise HTTPException(400, "lines array is required")
    if len(lines) > 200:
        raise HTTPException(400, "batch size limited to 200 lines")

    from tetrahedron_memory.runtime_observer import _parse_line_auto

    obs = _get_observer(request)
    results = {"parsed": 0, "accepted": 0, "rejected": 0}
    for line in lines:
        if not isinstance(line, str) or not line.strip():
            continue
        level, module, message, extra = _parse_line_auto(line)
        results["parsed"] += 1
        accepted = obs.observe(
            level=level,
            module=module,
            message=message,
            source="agent-json-ingest",
            metadata=extra if extra else None,
        )
        if accepted:
            results["accepted"] += 1
        else:
            results["rejected"] += 1

    return results


@router.get("/observer/trajectories")
async def observer_trajectory_stream(request: Request):
    _get_observer(request)
    queue = asyncio.Queue(maxsize=100)
    _trajectory_subscribers.append(queue)

    async def generator():
        try:
            while True:
                try:
                    msg = queue.get_nowait()
                    yield f"data: {msg}\n\n"
                except asyncio.QueueEmpty:
                    pass
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _trajectory_subscribers.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/observer/register-classifier")
def observer_register_classifier(req: dict, request: Request):
    obs = _get_observer(request)
    pattern_str = req.get("pattern")
    category = req.get("category", "custom")
    weight = float(req.get("weight", 0.5))
    immediate = bool(req.get("immediate", False))

    if not pattern_str:
        raise HTTPException(400, "pattern is required")

    import re
    try:
        compiled = re.compile(pattern_str, re.IGNORECASE)
    except re.error as e:
        raise HTTPException(400, f"invalid regex: {e}")

    def classifier(event):
        if compiled.search(event.message):
            return category, weight, immediate
        json_fields = event.metadata.get("json_fields", {})
        for v in json_fields.values():
            if isinstance(v, str) and compiled.search(v):
                return category, weight, immediate
        return None

    obs.register_classifier(classifier)
    return {"registered": True, "category": category, "weight": weight}
