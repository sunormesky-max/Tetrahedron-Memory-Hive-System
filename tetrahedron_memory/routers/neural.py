from __future__ import annotations

import logging
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from tetrahedron_memory.app_state import AppState
from tetrahedron_memory.phase_transition_honeycomb import HoneycombPhaseTransition

log = logging.getLogger("tetramem.api")

router = APIRouter(prefix="/api/v1", tags=["neural"])


def _get_state(request: Request) -> AppState:
    return request.app.state.tetramem


@router.get("/pcnn/states")
def pcnn_states(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return {"states": s.field.get_pcnn_node_states()}


@router.get("/pcnn/tension-map")
def pcnn_tension(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return {"tension_map": s.field.get_tension_map()}


@router.get("/pcnn/hebbian")
def hebbian_paths(request: Request):
    s = _get_state(request)
    with s.state_lock:
        ps = s.field.pulse_status()
        return {
            "hebbian": ps.get("hebbian", {}),
            "top_paths": ps.get("hebbian_top_paths", []),
        }


@router.get("/pcnn/config")
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


@router.get("/pulse-status")
def pulse_status(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return s.field.pulse_status()


@router.get("/pulse-snapshot")
def pulse_snapshot(request: Request):
    s = _get_state(request)
    with s.state_lock:
        pulses = []
        for nid, node in s.field._nodes.items():
            if node.pulse_accumulator > 0.05 or (node.is_occupied and node.activation > 0.5):
                pulses.append({
                    "id": nid,
                    "centroid": node.position.tolist(),
                    "strength": node.pulse_accumulator if not node.is_occupied else node.activation,
                    "occupied": node.is_occupied,
                })
        return {"pulses": pulses, "count": len(pulses)}


@router.post("/cascade/trigger")
def cascade_trigger(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    strength = req.get("strength", 0.5)
    source_id = req.get("source_id")
    with s.state_lock:
        return s.field.trigger_cascade(source_id=source_id, strength=strength)


@router.post("/structure-pulse/trigger")
def structure_pulse_trigger(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    source_id = req.get("source_id")
    with s.state_lock:
        return s.field.trigger_structure_pulse(source_id=source_id)


@router.post("/dream")
def dream(request: Request):
    s = _get_state(request)
    with s.state_lock:
        result = s.field.run_dream_cycle()
    return {"result": result}


@router.get("/dream/status")
def dream_status(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return s.field.dream_status()


@router.get("/dream/history")
def dream_history(request: Request, n: int = 10):
    s = _get_state(request)
    with s.state_lock:
        return {"history": s.field.dream_history(n)}


@router.post("/self-organize")
def self_organize(request: Request):
    s = _get_state(request)
    with s.state_lock:
        result = s.field.run_self_organize()
        s.log_op("self_organize")
    return {"stats": result}


@router.get("/self-organize/status")
def self_organize_status(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return s.field.self_organize_status()


@router.get("/self-organize/history")
def self_organize_history(request: Request, n: int = 10):
    s = _get_state(request)
    with s.state_lock:
        return {"history": s.field.self_organize_history(n)}


@router.get("/emergence/status")
def emergence_status(request: Request):
    s = _get_state(request)
    with s.state_lock:
        st = s.field.stats()
    emergence = st.get("emergence_summary", {})
    return {
        "emergence_score": emergence.get("overall_score", 0),
        "emergence_level": emergence.get("emergence_level", "unknown"),
        "clustering": emergence.get("clustering", {}).get("score", 0),
        "bridges": emergence.get("bridges", {}).get("score", 0),
        "crystal": emergence.get("crystal", {}).get("score", 0),
        "phase": emergence.get("phase", {}).get("score", 0),
    }


@router.post("/emergence/trigger")
def emergence_trigger(request: Request):
    s = _get_state(request)
    with s.state_lock:
        s_before = s.field.stats()
        s.field.run_self_organize()
        s.field.run_dream_cycle()
        s_after = s.field.stats()
    effect = abs(s_after.get("occupied_nodes", 0) - s_before.get("occupied_nodes", 0))
    s.log_op("emergence_trigger")
    return {"triggered": True, "effect": effect}


@router.get("/emergence/quality")
def emergence_quality(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return s.field.compute_emergence_quality()


@router.get("/emergence/history")
def emergence_history(request: Request, n: int = 20):
    s = _get_state(request)
    with s.state_lock:
        history = s.field._emergence_history[-n:]
        return {"count": len(history), "history": history}


@router.get("/self-check/status")
def self_check_status(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return s.field.self_check_status()


@router.post("/self-check/run")
def self_check_run(request: Request):
    s = _get_state(request)
    with s.state_lock:
        result = s.field.run_self_check()
    s.log_op("self_check")
    return result


@router.get("/self-check/history")
def self_check_history(request: Request, n: int = 10):
    s = _get_state(request)
    with s.state_lock:
        return {"history": s.field.self_check_history(n)}


@router.get("/crystallized/status")
def crystallized_status(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return s.field.crystallized_status()


@router.post("/crystallized/force")
def force_crystallize(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return s.field.force_crystallize()


@router.get("/duplicates")
def detect_duplicates(request: Request):
    s = _get_state(request)
    with s.state_lock:
        dupes = s.field.detect_duplicates()
    return {"duplicates": dupes, "count": len(dupes)}


@router.get("/isolated")
def detect_isolated(request: Request):
    s = _get_state(request)
    with s.state_lock:
        isolated = s.field.detect_isolated()
    return {"isolated": isolated, "count": len(isolated)}


@router.get("/clusters")
def get_clusters(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return {"clusters": s.field.get_clusters()}


@router.get("/shortcuts")
def get_shortcuts(request: Request, n: int = 20):
    s = _get_state(request)
    with s.state_lock:
        sc_data = s.field.get_shortcuts(n)
        return {"shortcuts": sc_data, "count": len(sc_data)}


@router.get("/phase-transition/status")
def phase_status(request: Request):
    s = _get_state(request)
    try:
        with s.state_lock:
            detector = s.phase_detector
        if detector is None:
            return {"status": "not_initialized"}
        gt, tensions = detector.compute_global_tension(s.field)
        return {
            "status": "ok",
            "global_tension": round(gt, 3),
            "nodes_with_tension": len(tensions),
            "trend": detector.get_tension_trend(),
            "total_transitions": detector._transition_count,
        }
    except Exception as e:
        log.error("Phase status failed: %s", e)
        raise HTTPException(500, "Internal error")
@router.post("/phase-transition/trigger")
def phase_trigger(request: Request):
    s = _get_state(request)
    try:
        with s.state_lock:
            detector = HoneycombPhaseTransition(tension_threshold=0.0, cooldown_seconds=0)
            global_tension, tensions = detector.compute_global_tension(s.field)
            clusters = detector.identify_tension_clusters(tensions, s.field)
        if not clusters:
            return {"status": "no_tension", "global_tension": global_tension}
        result = detector.execute_transition(s.field, tensions, clusters)
        return {"status": "transition_complete", "result": result}
    except Exception as e:
        log.error("Phase trigger failed: %s", e)
        raise HTTPException(500, "Internal error")
@router.get("/tension-map")
def tension_map(request: Request):
    s = _get_state(request)
    try:
        with s.state_lock:
            field = s.field
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

        with s.state_lock:
            detailed = field.get_tension_map(top_n=10)
        return {
            "global_tension": round(global_tension, 3),
            "top_tension_nodes": [{"id": tid[:12], "tension": round(t, 3)} for tid, t in top],
            "detailed_tension": detailed,
            "total_scored": len(tensions),
        }
    except Exception as e:
        log.error("Tension map failed: %s", e)
        raise HTTPException(500, "Internal error")
@router.post("/reorganize")
def abstract_reorganize(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    phases = req.get("phases", ["self_organize", "bridge_check", "dream", "decay"])
    results = {}
    with s.state_lock:
        if "self_organize" in phases:
            results["self_organize"] = s.field.run_self_organize()
        if "bridge_check" in phases:
            s.field._check_convergence_bridges()
            results["bridges"] = {"checked": True}
        if "dream" in phases:
            results["dream"] = s.field.run_dream_cycle()
        if "decay" in phases:
            s.field._global_decay()
            results["decay"] = {"applied": True}
        if "autocorrelation" in phases:
            ac = s.field.compute_spatial_autocorrelation()
            results["spatial_autocorrelation"] = {"morans_i": ac}
        results["stats"] = s.field.stats()
    s.log_op("abstract_reorganize")
    return {"result": results}
