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


@router.get("/dark-plane/substrate/stats")
def substrate_stats_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        substrate = getattr(state.field, "_dark_plane_substrate", None)
        if substrate is None:
            return {"error": "substrate not initialized"}
        return substrate.get_stats()


@router.get("/dark-plane/substrate/features")
def substrate_features_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        substrate = getattr(state.field, "_dark_plane_substrate", None)
        if substrate is None:
            return {"error": "substrate not initialized"}
        st = substrate._state
        return {
            "h0": [
                {"birth": f.birth, "death": f.death, "persistence": f.persistence, "topo_charge": f.topo_charge}
                for f in st.features_h0[:50]
            ],
            "h1": [
                {"birth": f.birth, "death": f.death, "persistence": f.persistence, "topo_charge": f.topo_charge}
                for f in st.features_h1[:50]
            ],
            "h2": [
                {"birth": f.birth, "death": f.death, "persistence": f.persistence, "topo_charge": f.topo_charge}
                for f in st.features_h2[:50]
            ],
        }


@router.get("/dark-plane/homology/h3-h6")
def homology_h3_h6_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        substrate = getattr(state.field, "_dark_plane_substrate", None)
        if substrate is None:
            return {"error": "substrate not initialized"}
        st = substrate._state
        return {
            "h3": {"count": st.h3.count, "energy": st.h3.energy, "growth_rate": st.h3.growth_rate},
            "h4": {"count": st.h4.count, "energy": st.h4.energy, "growth_rate": st.h4.growth_rate},
            "h5": {"count": st.h5.count, "energy": st.h5.energy, "growth_rate": st.h5.growth_rate},
            "h6": {"count": st.h6.count, "energy": st.h6.energy, "growth_rate": st.h6.growth_rate},
            "psi_field": substrate._psi_field,
            "h5_regulation": substrate._h5_regulation,
        }


@router.get("/dark-plane/coherence")
def coherence_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        substrate = getattr(state.field, "_dark_plane_substrate", None)
        if substrate is None:
            return {"error": "substrate not initialized"}
        return {
            "coherence": substrate._state.coherence,
            "psi_field": substrate._psi_field,
            "total_dim_energy": substrate._state.total_dim_energy,
            "cascade_potential": substrate._state.cascade_potential,
        }


@router.get("/dark-plane/phase-transition/history")
def phase_transition_history_api(n: int = 50, request: Request = None):
    state = _get_state(request)
    with state.state_lock:
        substrate = getattr(state.field, "_dark_plane_substrate", None)
        if substrate is None:
            return {"error": "substrate not initialized"}
        pts = substrate._state.phase_transitions[-n:]
        return {
            "total": substrate._state.total_phase_transitions,
            "last": substrate._state.last_phase_transition,
            "events": [
                {
                    "timestamp": pt.timestamp,
                    "level": pt.level,
                    "energy": pt.energy,
                    "coherence": pt.coherence,
                    "trigger_condition": pt.trigger_condition,
                }
                for pt in pts
            ],
        }


@router.get("/void-channels")
def void_channels_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        vc = getattr(state.field, "_void_channel", None)
        if vc is None:
            return {"error": "void channels not initialized"}
        return {"stats": vc.get_stats(), "channels": vc.get_state()}


@router.get("/void-channels/{node_id}")
def void_channels_node_api(node_id: str, request: Request):
    state = _get_state(request)
    with state.state_lock:
        vc = getattr(state.field, "_void_channel", None)
        if vc is None:
            return {"error": "void channels not initialized"}
        channels = vc.get_channels_for_node(node_id)
        return {
            "node_id": node_id,
            "channels": [
                {
                    "node_a": ch.node_a,
                    "node_b": ch.node_b,
                    "strength": ch.strength,
                    "dimension": ch.dimension,
                    "energy_coupling": ch.energy_coupling,
                }
                for ch in channels
            ],
        }


@router.get("/void-channels/stats")
def void_channels_stats_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        vc = getattr(state.field, "_void_channel", None)
        if vc is None:
            return {"error": "void channels not initialized"}
        return vc.get_stats()
