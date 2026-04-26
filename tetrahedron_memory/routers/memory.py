from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from tetrahedron_memory.app_state import AppState, _resolve_node
from tetrahedron_memory.input_validation import InputValidator

log = logging.getLogger("tetramem.api")

router = APIRouter(prefix="/api/v1", tags=["memory"])


def _get_state(request: Request) -> AppState:
    return request.app.state.tetramem


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


@router.post("/store")
def store(req: StoreReq, request: Request):
    s = _get_state(request)
    tenant = getattr(request.state, "tenant", {"tenant_id": "default", "role": "anonymous"})
    tenant_id = tenant.get("tenant_id", "default")

    content = InputValidator.sanitize_content(req.content)
    valid, err = InputValidator.validate_store(content, req.labels, req.weight, req.metadata)
    if not valid:
        s.metrics.increment("store_validation_errors")
        raise HTTPException(400, err)

    if not s.quota_manager.check_allowed(tenant_id):
        raise HTTPException(403, "Quota exceeded")

    try:
        with s.state_lock:
            tid = s.field.store(content, labels=req.labels, weight=req.weight, metadata=req.metadata)
        s.version_control.record_version(tid, content, req.weight, req.labels, req.metadata)
        s.quota_manager.increment(tenant_id)
        s.audit_log.log(tenant_id, "store", {"labels": req.labels, "weight": req.weight}, node_id=tid)
        s.emit_event("memory_stored", {"id": tid, "labels": req.labels, "weight": req.weight})
        s.log_op("store", {"id": tid, "content": req.content, "labels": req.labels or [], "weight": req.weight, "centroid": s.field._nodes[tid].position.tolist(), "metadata": req.metadata or {}, "creation_time": float(s.field._nodes[tid].creation_time)})
        if s.persistence is not None:
            s.persistence.log_operation("store", {"id": tid, "content": req.content, "labels": req.labels, "weight": req.weight, "metadata": req.metadata, "centroid": s.field._nodes[tid].position.tolist(), "creation_time": float(s.field._nodes[tid].creation_time)})
        s.metrics.increment("stores")
        return {"id": tid}
    except Exception as e:
        s.metrics.increment("store_errors")
        log.error("Store failed: %s", e)
        raise HTTPException(500, "Internal error")


@router.post("/query")
def query(req: QueryReq, request: Request):
    s = _get_state(request)
    tenant = getattr(request.state, "tenant", {"tenant_id": "default"})
    valid, err = InputValidator.validate_query(req.query, req.k)
    if not valid:
        raise HTTPException(400, err)
    try:
        t = s.metrics.timer("query_duration")
        with t:
            with s.state_lock:
                results = s.field.query(req.query, k=req.k, labels=req.labels)
        s.metrics.increment("queries")
        s.audit_log.log(tenant.get("tenant_id", "default"), "query",
                         {"k": req.k, "result_count": len(results)})
        return {"results": results}
    except Exception as e:
        s.metrics.increment("query_errors")
        log.error("Query failed: %s", e)
        raise HTTPException(500, "Internal error")


@router.post("/batch-store")
def batch_store(request: Request, req: List[Dict[str, Any]]):
    s = _get_state(request)
    results = []
    for item in req:
        try:
            content = item.get("content", "")
            labels = item.get("labels", [])
            weight = item.get("weight", 1.0)
            metadata = item.get("metadata", {})
            with s.state_lock:
                tid = s.field.store(content, labels=labels, weight=weight, metadata=metadata)
            results.append({"id": tid, "status": "ok"})
        except Exception as e:
            results.append({"content": item.get("content", "")[:50], "status": "error", "error": str(e)})
    s.log_op("batch_store")
    return {"results": results}


@router.post("/query-by-label")
def query_by_label(request: Request, req: dict):
    s = _get_state(request)
    try:
        labels = req.get("labels", [])
        k = req.get("k", 10)
        with s.state_lock:
            results = s.field.query("", k=k, labels=labels)
        return {"results": results}
    except Exception as e:
        log.error("Query by label failed: %s", e)
        raise HTTPException(500, "Internal error")
@router.post("/query-multiparam")
def query_multiparam(request: Request, req: dict):
    s = _get_state(request)
    try:
        q = req.get("query", "")
        k = req.get("k", 10)
        labels = req.get("labels", [])
        min_weight = req.get("min_weight", 0.0)
        with s.state_lock:
            results = s.field.query(q, k=k, labels=labels)
            if min_weight > 0:
                results = [r for r in results if r.get("weight", 0) >= min_weight]
        return {"results": results}
    except Exception as e:
        log.error("Query multiparam failed: %s", e)
        raise HTTPException(500, "Internal error")
@router.post("/weight-update")
def weight_update(request: Request, req: dict):
    s = _get_state(request)
    try:
        node_id = req.get("id", "")
        new_weight = req.get("weight", 1.0)
        with s.state_lock:
            node = s.field._nodes.get(node_id)
            if node is None:
                raise HTTPException(404, f"Node {node_id} not found")
            node.weight = float(new_weight)
            s.log_op("weight_update", {"id": node_id, "weight": float(new_weight)})
        return {"id": node_id, "weight": float(new_weight), "status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        log.error("Weight update failed: %s", e)
        raise HTTPException(500, "Internal error")
@router.post("/associate")
def associate(request: Request, req: AssociateReq):
    s = _get_state(request)
    try:
        with s.state_lock:
            results = s.field.associate(req.tetra_id, max_depth=req.max_depth)
        return {"associations": results}
    except Exception as e:
        log.error("Associate failed: %s", e)
        raise HTTPException(500, "Internal error")
@router.get("/browse")
def list_tetrahedra(request: Request, limit: int = 200, offset: int = 0):
    s = _get_state(request)
    with s.state_lock:
        items = s.field.list_occupied()
    total = len(items)
    return {"tetrahedra": items[offset:offset+limit], "count": total, "total": total, "limit": limit, "offset": offset}


@router.get("/tetrahedra/{tetra_id}")
def get_tetra(tetra_id: str, request: Request):
    s = _get_state(request)
    with s.state_lock:
        node = s.field.get_node(tetra_id)
    if node is None:
        raise HTTPException(404, "Not found")
    return node


@router.delete("/tetrahedra/{tetra_id}")
def delete_tetra(tetra_id: str, request: Request):
    s = _get_state(request)
    tenant = getattr(request.state, "tenant", {"tenant_id": "default"})
    with s.state_lock:
        node = s.field._nodes.get(tetra_id)
        if node is None:
            raise HTTPException(404, "Not found")
        if not node.is_occupied:
            raise HTTPException(404, "Not found")
        s.field._clear_node(tetra_id, node)
        if s.version_control:
            s.version_control.remove_versions(tetra_id)
    s.quota_manager.decrement(tenant.get("tenant_id", "default"))
    s.audit_log.log(tenant.get("tenant_id", "default"), "delete", {}, node_id=tetra_id)
    s.log_op("delete", {"id": tetra_id})
    return {"status": "ok", "deleted": tetra_id}


@router.put("/tetrahedra/{tetra_id}")
def update_tetra(tetra_id: str, request: Request, body: dict = None):
    s = _get_state(request)
    tenant = getattr(request.state, "tenant", {"tenant_id": "default"})
    req = body or {}
    content = req.get("content")
    weight = req.get("weight")
    labels = req.get("labels")
    with s.state_lock:
        node = s.field._nodes.get(tetra_id)
        if node is None:
            raise HTTPException(404, "Not found")
        if content is not None:
            node.content = content
        if weight is not None:
            node.weight = max(0.1, min(10.0, float(weight)))
        if labels is not None:
            node.labels = list(labels)
        s.version_control.record_version(tetra_id, node.content, node.weight, node.labels, node.metadata)
    s.audit_log.log(tenant.get("tenant_id", "default"), "update",
                     {"weight": weight, "has_content": content is not None}, node_id=tetra_id)
    s.log_op("update", {"id": tetra_id, "weight": weight, "labels": labels})
    return {"status": "ok", "updated": tetra_id}


@router.post("/timeline")
def timeline(request: Request, req: TimelineReq):
    s = _get_state(request)
    with s.state_lock:
        items, total = s.field.browse_timeline(
            direction=req.direction,
            limit=req.limit,
            offset=req.offset,
            label_filter=req.labels,
            min_weight=req.min_weight,
        )
    return {"items": items, "count": len(items), "total": total}


@router.post("/export")
@router.get("/export")
def export(request: Request):
    s = _get_state(request)
    with s.state_lock:
        items = s.field.list_occupied()
    lines = ["# TetraMem-XL v8.0 Memory Export\n"]
    lines.append(f"Total memories: {len(items)}\n")
    for item in items:
        labels_str = ", ".join(item.get("labels", [])) or "-"
        lines.append(f"## [{item['id'][:8]}] (w={item['weight']:.2f} a={item.get('activation',0):.2f}) [{labels_str}]")
        lines.append(item.get("content", ""))
        lines.append("")
    text = "\n".join(lines)
    out = Path(s.storage_dir) / "tetramem_export.md"
    out.write_text(text, encoding="utf-8")
    return {"status": "ok", "path": str(out), "size": len(text)}


@router.post("/import")
def import_memories(request: Request, body: dict = None):
    s = _get_state(request)
    tenant = getattr(request.state, "tenant", {"tenant_id": "default"})
    req = body or {}
    items = req.get("memories", [])
    if not items:
        raise HTTPException(400, "No memories provided")
    results = []
    imported = 0
    errors = 0
    with s.state_lock:
        for item in items:
            try:
                content = item.get("content", "")
                labels = item.get("labels", [])
                weight = item.get("weight", 1.0)
                metadata = item.get("metadata", {})
                tid = s.field.store(content, labels=labels, weight=weight, metadata=metadata)
                results.append({"id": tid, "status": "ok"})
                imported += 1
            except Exception as e:
                results.append({"content": item.get("content", "")[:50], "status": "error", "error": str(e)})
                errors += 1
    s.audit_log.log(tenant.get("tenant_id", "default"), "import",
                     {"imported": imported, "errors": errors})
    s.log_op("import", {"imported": imported, "errors": errors})
    return {"results": results, "imported": imported, "errors": errors}


@router.post("/navigate")
def navigate(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    source = req.get("source_id", "")
    target = req.get("target_id", "")
    if source and target:
        with s.state_lock:
            return s.field.agent_navigate(source, target)
    seed = req.get("seed_id", "")
    if seed:
        max_steps = req.get("max_steps", 30)
        with s.state_lock:
            node = s.field._nodes.get(seed)
            if node is None:
                return {"path": [], "nodes": [], "error": "seed_id not found"}
            visited = [seed]
            frontier = [seed]
            steps = 0
            while frontier and steps < max_steps:
                next_frontier = []
                for nid in frontier:
                    n = s.field._nodes.get(nid)
                    if n is None:
                        continue
                    for fn in n.face_neighbors:
                        if fn not in visited and fn in s.field._nodes:
                            visited.append(fn)
                            next_frontier.append(fn)
                    for en in n.edge_neighbors[:4]:
                        if en not in visited and en in s.field._nodes:
                            visited.append(en)
                            next_frontier.append(en)
                frontier = next_frontier
                steps += 1
            path_data = []
            for nid in visited[:max_steps]:
                n = s.field._nodes.get(nid)
                if n and n.is_occupied:
                    path_data.append({"id": nid, "content": n.content, "labels": list(n.labels), "weight": float(n.weight)})
            return {"path": visited[:max_steps], "nodes": path_data, "steps": steps}
    return {"path": [], "nodes": [], "error": "provide source_id+target_id or seed_id"}


@router.post("/seed-by-label")
def seed_by_label(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    labels = req.get("labels", [])
    if not labels:
        return {"id": None}
    with s.state_lock:
        results = s.field.query(" ".join(labels), k=1, labels=labels)
        if results:
            return {"id": results[0]["id"]}
        return {"id": None}


@router.post("/search")
def search_endpoint(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    query_text = req.get("query", "")
    limit = req.get("limit", 10)
    if not query_text:
        return {"results": []}
    with s.state_lock:
        results = s.field.query(query_text, k=limit)
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


@router.post("/read")
def read_endpoint(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    path = req.get("path", "")
    if not path:
        return {"content": "", "text": ""}
    with s.state_lock:
        node = s.field.get_node(path)
    if node is None:
        with s.state_lock:
            results = s.field.query(path, k=1)
            if results:
                r = results[0]
                return {"content": r.get("content", ""), "text": r.get("content", ""), "metadata": {"id": r.get("id", "")}}
        return {"content": "", "text": ""}
    return {"content": node.get("content", ""), "text": node.get("content", ""), "metadata": node.get("metadata", {})}
