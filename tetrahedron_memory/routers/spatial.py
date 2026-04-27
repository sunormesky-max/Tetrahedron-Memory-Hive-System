from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from tetrahedron_memory.app_state import AppState, _resolve_node

log = logging.getLogger("tetramem.api")

router = APIRouter(prefix="/api/v1", tags=["spatial"])


def _get_state(request: Request) -> AppState:
    return request.app.state.tetramem


def _resolve(state: AppState, node_id: str) -> str:
    try:
        return _resolve_node(state.field, node_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/topology-graph")
def topology_graph(request: Request):
    state = _get_state(request)
    with state.state_lock:
        _field = state.field
        occupied_ids = {nid for nid, _ in _field.occupied_items()}
        nearby_ids = set(occupied_ids)
        for oid in list(occupied_ids)[:50]:
            node = _field.node_get(oid)
            if node:
                nearby_ids.update(node.face_neighbors)
                nearby_ids.update(node.edge_neighbors[:6])

        nodes = []
        for nid in nearby_ids:
            node = _field.node_get(nid)
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
        for n1, n2, etype in _field.edges_items():
            if n1 in nid_set and n2 in nid_set:
                edges.append({"source": n1, "target": n2, "type": etype})

        return {"nodes": nodes, "edges": edges}


@router.get("/lattice-info")
def lattice_info(request: Request):
    state = _get_state(request)
    with state.state_lock:
        _field = state.field
        memories = []
        for nid, node in _field.all_nodes_items():
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
            "resolution": _field.get_resolution(),
            "spacing": _field.get_spacing(),
            "total_nodes": len(_field.all_nodes_items()),
            "memories": memories,
        }


@router.get("/topology-health")
def topology_health(request: Request):
    state = _get_state(request)
    return {"result": state.field.stats()}


@router.get("/lattice-integrity/check")
def lattice_integrity_check(request: Request):
    state = _get_state(request)
    with state.state_lock:
        return state.field.run_lattice_check()


@router.get("/lattice-integrity/status")
def lattice_integrity_status(request: Request):
    state = _get_state(request)
    with state.state_lock:
        return state.field.lattice_check_status()


@router.get("/lattice-integrity/history")
def lattice_integrity_history(request: Request, n: int = 10):
    state = _get_state(request)
    with state.state_lock:
        return {"history": state.field.lattice_check_history(n)}


@router.get("/scene-nodes")
def scene_nodes(request: Request):
    state = _get_state(request)
    with state.state_lock:
        _field = state.field
        nodes = []
        for nid, node in _field.all_nodes_items():
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


@router.get("/spatial/vacancy-map")
def vacancy_map(request: Request, top_n: int = 20):
    state = _get_state(request)
    with state.state_lock:
        _field = state.field
        vacancies = []
        for nid, node in _field.all_nodes_items():
            if not node.is_occupied:
                pos = node.position
                vacancies.append({
                    "id": nid,
                    "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "neighbors": sum(1 for nb_id in node.face_neighbors if _field.is_node_occupied(nb_id)),
                })
        vacancies.sort(key=lambda v: -v["neighbors"])
        return {"vacancies": vacancies[:top_n], "total": len(vacancies)}


@router.post("/query-spatial")
def query_spatial_api(request: Request, req: dict):
    state = _get_state(request)
    try:
        center = req.get("center")
        radius = float(req.get("radius", 3.0))
        k = int(req.get("k", 20))
        labels = req.get("labels")
        sort_by = req.get("sort_by", "distance")
        with state.state_lock:
            results = state.field.query_spatial(
                center=center, radius=radius, k=k,
                labels=labels, sort_by=sort_by,
            )
        return {"results": results, "count": len(results)}
    except Exception as e:
        log.error("Query spatial failed: %s", e)
        raise HTTPException(500, "Internal error")
@router.post("/query-direction")
def query_direction_api(request: Request, req: dict):
    state = _get_state(request)
    try:
        direction = req.get("direction", [1, 0, 0])
        from_center = bool(req.get("from_center", True))
        max_angle = float(req.get("max_angle", 0.5))
        k = int(req.get("k", 20))
        labels = req.get("labels")
        with state.state_lock:
            results = state.field.query_direction(
                direction=direction, from_center=from_center,
                max_angle=max_angle, k=k, labels=labels,
            )
        return {"results": results, "count": len(results)}
    except Exception as e:
        log.error("Query direction failed: %s", e)
        raise HTTPException(500, "Internal error")
@router.post("/query-temporal")
def query_temporal_api(request: Request, req: dict):
    state = _get_state(request)
    try:
        time_range = req.get("time_range")
        direction = req.get("direction", "newest")
        k = int(req.get("k", 20))
        labels = req.get("labels")
        min_weight = float(req.get("min_weight", 0.0))
        lifecycle_stage = req.get("lifecycle_stage")
        with state.state_lock:
            results = state.field.query_temporal(
                time_range=time_range, direction=direction,
                k=k, labels=labels, min_weight=min_weight,
                lifecycle_stage=lifecycle_stage,
            )
        return {"results": results, "count": len(results)}
    except Exception as e:
        log.error("Query temporal failed: %s", e)
        raise HTTPException(500, "Internal error")
@router.get("/temporal-sequence/{node_id}")
def temporal_sequence_api(request: Request, node_id: str, direction: str = "forward", max_depth: int = 10):
    state = _get_state(request)
    try:
        with state.state_lock:
            results = state.field.query_temporal_sequence(node_id, direction=direction, max_depth=max_depth)
        return {"sequence": results, "length": len(results)}
    except Exception as e:
        log.error("Temporal sequence failed: %s", e)
        raise HTTPException(500, "Internal error")
@router.get("/lifecycle-stats")
def lifecycle_stats_api(request: Request):
    state = _get_state(request)
    with state.state_lock:
        return state.field.get_lifecycle_stats()


@router.get("/spatial/quality/{node_id}")
def spatial_quality(request: Request, node_id: str):
    state = _get_state(request)
    with state.state_lock:
        _field = state.field
        nid = _resolve(state, node_id)
        gq = _field.compute_node_geometric_quality(nid)
        div = _field.compute_geometric_topo_divergence(nid)
        rf_energy = _field.reflection_field_node_energy(nid)
        return {"node_id": nid, "geometric_quality": gq, "geo_topo_divergence": round(div, 4), "field_energy": rf_energy}


@router.get("/spatial/crystallographic-direction")
def crystallographic_direction_test(request: Request):
    state = _get_state(request)
    with state.state_lock:
        _field = state.field
        occupied = _field.occupied_items()
        if len(occupied) < 2:
            return {"error": "need at least 2 nodes"}
        a, b = occupied[0][1], occupied[1][1]
        factor = _field.bcc_direction_factor(a.position, b.position)
        dist = float(np.linalg.norm(a.position - b.position))
        return {"factor": round(factor, 4), "distance": round(dist, 4), "sample_nodes": [occupied[0][0][:8], occupied[1][0][:8]]}


@router.get("/spatial/autocorrelation")
def spatial_autocorrelation(request: Request):
    state = _get_state(request)
    with state.state_lock:
        _field = state.field
        morans_i = _field.compute_spatial_autocorrelation()
        return {
            "morans_i": round(morans_i, 4),
            "interpretation": "clustered" if morans_i > 0.1 else ("dispersed" if morans_i < -0.1 else "random"),
            "history_len": _field.autocorrelation_history_len(),
        }


@router.get("/spatial/bcc-cell-coherence/{node_id}")
def bcc_cell_coherence(request: Request, node_id: str):
    state = _get_state(request)
    with state.state_lock:
        _field = state.field
        nid = _resolve(state, node_id)
        coherence = _field.bcc_cell_coherence(nid)
        cellmates = _field.get_bcc_cellmates(nid)
        occ_cellmates = sum(1 for cmid in cellmates if _field.is_node_occupied(cmid))
        return {
            "node_id": nid,
            "bcc_cell_coherence": round(coherence, 4),
            "total_cellmates": len(cellmates),
            "occupied_cellmates": occ_cellmates,
        }


@router.get("/honeycomb/analysis")
def honeycomb_analysis(request: Request):
    state = _get_state(request)
    with state.state_lock:
        return state.field.honeycomb_analysis()


@router.get("/honeycomb/cells")
def honeycomb_cells(request: Request, n: int = 20, sort_by: str = "quality"):
    state = _get_state(request)
    with state.state_lock:
        return {"cells": state.field.get_tetrahedral_cells(n, sort_by), "count": n}


@router.get("/honeycomb/cells/{node_id}")
def honeycomb_cells_for_node(request: Request, node_id: str):
    state = _get_state(request)
    with state.state_lock:
        cells = state.field.get_cell_for_node(node_id)
        return {"node_id": node_id, "cells": cells, "count": len(cells)}
