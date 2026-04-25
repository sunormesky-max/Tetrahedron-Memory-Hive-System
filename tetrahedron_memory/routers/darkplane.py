import logging

from fastapi import APIRouter, HTTPException, Request

from tetrahedron_memory.app_state import AppState, _resolve_node
from tetrahedron_memory.spatial_reflection import SpatialReflectionField

log = logging.getLogger("tetramem.api")

router = APIRouter(prefix="/api/v1", tags=["darkplane"])


def _get_state(request: Request) -> AppState:
    return request.app.state.tetramem


@router.post("/dark-plane/flow")
def dark_plane_flow_api(request: Request):
    state = _get_state(request)
    try:
        with state.state_lock:
            result = state.field.dark_plane_flow()
        return result
    except Exception as e:
        log.error("Dark plane flow failed: %s", e)
        raise HTTPException(500, "Internal error")


@router.get("/dark-plane/stats")
def dark_plane_stats_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        engine = getattr(state.field, "_dark_plane_engine", None)
        if engine is None:
            return {"error": "not initialized"}
        return engine.get_system_energy_stats()


@router.get("/dark-plane/node/{node_id}")
def dark_plane_node_report(node_id: str, request: Request):
    state = _get_state(request)
    with state.state_lock:
        engine = getattr(state.field, "_dark_plane_engine", None)
        if engine is None:
            return {"error": "not initialized"}
        return engine.get_node_energy_report(node_id)


@router.get("/dark-plane/history")
def dark_plane_history(last_n: int = 20, request: Request = None):
    state = _get_state(request)
    with state.state_lock:
        engine = getattr(state.field, "_dark_plane_engine", None)
        if engine is None:
            return {"error": "not initialized"}
        return {"history": engine.get_energy_history(last_n)}


@router.post("/attention/focus")
def attention_focus_api(req: dict, request: Request):
    state = _get_state(request)
    try:
        with state.state_lock:
            result = state.field.attention_set_focus(
                center=req.get("center"),
                radius=float(req.get("radius", 5.0)),
                strength=float(req.get("strength", 1.0)),
                labels=req.get("labels"),
                query_text=req.get("query_text"),
            )
        return result
    except Exception as e:
        log.error("Attention focus failed: %s", e)
        raise HTTPException(500, "Internal error")


@router.post("/attention/clear")
def attention_clear_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        state.field.attention_clear()
    return {"status": "ok"}


@router.get("/attention/status")
def attention_status_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        return state.field.attention_status()


@router.get("/reflection-field/status")
def reflection_field_status(request: Request):
    state = _get_state(request)
    with state.state_lock:
        if state.field._reflection_field is None:
            return {"status": "not_initialized"}
        return state.field._reflection_field.stats()


@router.post("/reflection-field/run")
def reflection_field_run(request: Request):
    state = _get_state(request)
    with state.state_lock:
        if state.field._reflection_field is None:
            state.field._reflection_field = SpatialReflectionField()
        return state.field._reflection_field.run_reflection_cycle(state.field)


@router.get("/reflection-field/energy/{node_id}")
def reflection_field_energy(node_id: str, request: Request):
    state = _get_state(request)
    with state.state_lock:
        try:
            nid = _resolve_node(state.field, node_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if state.field._reflection_field is None:
            return {"node_id": nid, "energy": 0.5}
        energy = state.field._reflection_field.get_node_energy(nid)
        quality = state.field._reflection_field.get_spatial_quality(state.field, nid)
        return {"node_id": nid, "energy": round(energy, 4), "spatial_quality": round(quality, 4)}


@router.get("/regulation/status")
def regulation_status_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        if state.field._self_regulation is None:
            return {"active": False}
        return {"active": True, **state.field._self_regulation.status()}


@router.post("/regulation/trigger")
def regulation_trigger_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        if state.field._self_regulation is None:
            raise HTTPException(400, "Self-regulation not initialized")
        record = state.field._self_regulation.regulate()
        return record


@router.post("/regulation/force-mode")
def regulation_force_mode_api(req: dict, request: Request):
    state = _get_state(request)
    with state.state_lock:
        if state.field._self_regulation is None:
            raise HTTPException(400, "Self-regulation not initialized")
        mode = req.get("mode", "")
        state.field._self_regulation.force_mode(mode)
        return {"status": "ok", "mode": mode}


@router.get("/regulation/history")
def regulation_history_api(n: int = 20, request: Request = None):
    state = _get_state(request)
    with state.state_lock:
        if state.field._self_regulation is None:
            return {"count": 0, "history": []}
        history = state.field._self_regulation.get_history(n)
        return {"count": len(history), "history": history}
