import logging

from fastapi import APIRouter, HTTPException, Request

from tetrahedron_memory.app_state import AppState

log = logging.getLogger("tetramem.api")

router = APIRouter(prefix="/api/v1", tags=["observer"])


def _get_state(request: Request) -> AppState:
    return request.app.state.tetramem


def _get_observer(request: Request):
    state = _get_state(request)
    obs = getattr(state.field, "_runtime_observer", None)
    if obs is None:
        raise HTTPException(400, "RuntimeObserver not initialized")
    return obs


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
    if not message:
        raise HTTPException(400, "message is required")
    obs = _get_observer(request)
    accepted = obs.observe(
        level=level, module=module, message=message, source=source,
    )
    return {"accepted": accepted}
