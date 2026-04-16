"""
TetraMem-XL API v2.5 — TetraMesh + TetraDreamCycle Core
Hybrid query + SQLite persistence + semantic geometry embedding
"""
import os, time, hashlib, threading, numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from tetrahedron_memory.tetra_mesh import TetraMesh, TetraMeshStore, text_to_geometry
from tetrahedron_memory.tetra_dream import TetraDreamCycle
from tetrahedron_memory.tetra_self_org import TetraSelfOrganizer

STORAGE_DIR = os.environ.get("TETRAMEM_STORAGE", "./tetramem_data_v2")

_mesh: TetraMesh = None
_dream: TetraDreamCycle = None
_self_org: TetraSelfOrganizer = None
_store: TetraMeshStore = None
_state_lock = threading.RLock()
_save_timer: threading.Timer = None
_save_lock = threading.Lock()
_dirty = False
_dirty_ids: set = set()
_start_time = time.time()


def _init_llm_executor():
    provider = os.environ.get("TETRAMEM_LLM_PROVIDER", "")
    if not provider:
        return None
    try:
        from tetrahedron_memory.llm_integration import create_executor
        executor = create_executor(provider)
        if executor:
            print(f"[TetraMem v2] LLM dream executor: {provider}")
        return executor
    except Exception as e:
        print(f"[TetraMem v2] LLM init failed: {e}, using default")
        return None


def init_state():
    """Initialize global state. Called by lifespan or directly in tests."""
    global _mesh, _dream, _self_org, _store
    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)

    db_path = os.path.join(STORAGE_DIR, "tetramem.db")
    _store = TetraMeshStore(db_path)
    _store.init_db()

    loaded = _store.load_full_mesh()
    if loaded and len(loaded.tetrahedra) > 0:
        _mesh = loaded
        print(f"[TetraMem v2.5] Loaded {len(_mesh.tetrahedra)} tetrahedra from SQLite")
    else:
        _mesh = TetraMesh()
        json_file = Path(STORAGE_DIR) / "mesh_index.json"
        if json_file.exists():
            import json
            data = json.loads(json_file.read_text())
            for item in data.get("tetrahedra", []):
                pt = np.array(item["centroid"])
                _mesh.store(
                    content=item["content"],
                    seed_point=pt,
                    labels=item.get("labels", []),
                    weight=item.get("weight", 1.0),
                    metadata=item.get("metadata"),
                )
            _store.save_full_mesh(_mesh)
            print(f"[TetraMem v2.5] Migrated {len(_mesh.tetrahedra)} tetrahedra from JSON to SQLite")
        else:
            print("[TetraMem v2.5] Fresh start")

    llm = _init_llm_executor()
    _dream = TetraDreamCycle(_mesh, llm_executor=llm)
    _self_org = TetraSelfOrganizer(_mesh)


@asynccontextmanager
async def lifespan(application):
    init_state()
    yield
    _flush_save()
    if _store:
        _store.close()
    print("[TetraMem v2.5] Shutdown complete, data flushed to SQLite")


app = FastAPI(title="TetraMem-XL v2", version="2.5.0", lifespan=lifespan)


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


class DreamReq(BaseModel):
    force: bool = False


class SelfOrgReq(BaseModel):
    max_iterations: int = Field(default=5, ge=1, le=20)


class AbstractReorgReq(BaseModel):
    min_density: int = Field(default=2, ge=1)
    max_operations: int = Field(default=20, ge=1, le=100)


class NavigateReq(BaseModel):
    seed_id: str
    max_steps: int = Field(default=30, ge=1, le=100)
    strategy: str = "bfs"


class SeedByLabelReq(BaseModel):
    labels: List[str]


def _do_save():
    global _dirty, _dirty_ids
    with _state_lock:
        mesh = _mesh
        store = _store
        ids_to_save = _dirty_ids.copy()
    with _save_lock:
        if store:
            store.incremental_save(mesh, dirty_ids=ids_to_save if ids_to_save else None)
        _dirty = False
        _dirty_ids = set()


def _schedule_save(dirty_id: str = None):
    with _save_lock:
        global _save_timer, _dirty, _dirty_ids
        _dirty = True
        if dirty_id:
            _dirty_ids.add(dirty_id)
        if _save_timer is not None:
            _save_timer.cancel()
        _save_timer = threading.Timer(2.0, _do_save)
        _save_timer.daemon = True
        _save_timer.start()


def _flush_save():
    global _save_timer
    if _save_timer is not None:
        _save_timer.cancel()
        _save_timer = None
    if _dirty:
        _do_save()


def _text_to_point(text: str, labels=None) -> np.ndarray:
    return text_to_geometry(text, labels=labels)


@app.post("/api/v1/store")
def store(req: StoreReq):
    try:
        with _state_lock:
            mesh = _mesh
        pt = _text_to_point(req.content, labels=req.labels)
        tid = mesh.store(
            content=req.content,
            seed_point=pt,
            labels=req.labels,
            weight=req.weight,
            metadata=req.metadata,
            dedup=True,
        )
        _schedule_save(dirty_id=tid)
        return {"id": tid}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/query")
def query(req: QueryReq):
    try:
        with _state_lock:
            mesh = _mesh
        pt = _text_to_point(req.query)
        results = mesh.query_topological(pt, k=req.k * 3, labels=req.labels, query_text=req.query)
        exclude = set()
        if req.exclude_dreams:
            exclude.add("__dream__")
        if req.exclude_low_confidence:
            exclude.add("low_confidence")
        items = []
        for tid, dist in results:
            t = mesh.get_tetrahedron(tid)
            if t and not (exclude & set(t.labels)):
                items.append({
                    "id": tid,
                    "content": t.content,
                    "distance": float(dist),
                    "weight": float(t.weight),
                    "labels": list(t.labels) if t.labels else [],
                })
                if len(items) >= req.k:
                    break
        return {"results": items}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/associate")
def associate(req: AssociateReq):
    try:
        with _state_lock:
            mesh = _mesh
        results = mesh.associate_topological(req.tetra_id, max_depth=req.max_depth)
        items = []
        for tid, score, atype in results:
            t = mesh.get_tetrahedron(tid)
            if t:
                items.append({
                    "id": tid, "content": t.content,
                    "score": float(score), "type": atype,
                    "weight": float(t.weight), "labels": list(t.labels) if t.labels else [],
                })
        return {"associations": items}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/dream")
def dream(req: DreamReq):
    try:
        with _state_lock:
            dream_cycle = _dream
        result = dream_cycle.trigger_now()
        _schedule_save()
        return {"result": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/self-organize")
def self_organize(req: SelfOrgReq):
    try:
        with _state_lock:
            global _self_org
            _self_org = TetraSelfOrganizer(_mesh, max_iterations=req.max_iterations)
            self_org = _self_org
            mesh = _mesh
        stats = self_org.run()
        _schedule_save()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/abstract-reorganize")
def abstract_reorganize(req: AbstractReorgReq):
    try:
        with _state_lock:
            mesh = _mesh
        result = mesh.abstract_reorganize(
            min_density=req.min_density,
            max_operations=req.max_operations,
        )
        _schedule_save()
        return {"result": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/navigate")
def navigate(req: NavigateReq):
    try:
        with _state_lock:
            mesh = _mesh
        path = mesh.navigate_topology(
            seed_id=req.seed_id,
            max_steps=req.max_steps,
            strategy=req.strategy,
        )
        items = []
        for tid, conn_type, hops in path:
            t = mesh.get_tetrahedron(tid)
            if t:
                items.append({
                    "id": tid, "content": t.content,
                    "connection": conn_type, "hops": hops,
                    "weight": float(t.weight), "labels": list(t.labels) if t.labels else [],
                })
        return {"path": items}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/seed-by-label")
def seed_by_label(req: SeedByLabelReq):
    try:
        with _state_lock:
            mesh = _mesh
        tid = mesh.seed_by_label(req.labels)
        if tid is None:
            return {"id": None}
        t = mesh.get_tetrahedron(tid)
        return {"id": tid, "content": t.content if t else None}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/stats")
def stats():
    try:
        with _state_lock:
            mesh = _mesh
        return mesh.get_statistics()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": "2.5.0", "uptime_seconds": time.time() - _start_time}


@app.post("/api/v1/export")
@app.get("/api/v1/export")
def export():
    try:
        with _state_lock:
            mesh = _mesh
        lines = ["# TetraMem-XL Memory Export\n"]
        lines.append(f"Total tetrahedra: {len(mesh.tetrahedra)}\n")
        for tid, t in mesh.tetrahedra.items():
            labels_str = ", ".join(t.labels) if t.labels else "-"
            lines.append(f"## [{tid[:8]}] (w={t.weight:.2f}) [{labels_str}]")
            lines.append(t.content)
            lines.append("")
        text = "\n".join(lines)
        out_local = Path(STORAGE_DIR) / "tetramem_export.md"
        out_local.write_text(text, encoding="utf-8")
        return {"status": "ok", "path": str(out_local), "size": len(text)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/closed-loop")
def closed_loop():
    with _state_lock:
        dream_cycle = _dream
        self_org = _self_org
        mesh = _mesh
    results = {}
    try:
        results["dream"] = dream_cycle.trigger_now()
    except Exception as e:
        results["dream_error"] = str(e)
    try:
        results["self_org"] = self_org.run()
    except Exception as e:
        results["self_org_error"] = str(e)
    try:
        results["abstract_reorg"] = mesh.abstract_reorganize()
    except Exception as e:
        results["reorg_error"] = str(e)
    _schedule_save()
    return {"result": results}


@app.get("/api/v1/topology-health")
def topology_health():
    try:
        with _state_lock:
            mesh = _mesh
        return {"result": mesh.get_statistics()}
    except Exception as e:
        raise HTTPException(500, str(e))


class TimelineReq(BaseModel):
    direction: str = "newest"
    limit: int = Field(default=20, ge=1, le=100)
    labels: Optional[List[str]] = None
    min_weight: float = Field(default=0.0, ge=0.0)
    exclude_labels: Optional[List[str]] = None


@app.post("/api/v1/timeline")
def timeline(req: TimelineReq):
    try:
        with _state_lock:
            mesh = _mesh
        items = mesh.browse_timeline(
            direction=req.direction,
            limit=req.limit,
            label_filter=req.labels,
            min_weight=req.min_weight,
            exclude_labels=req.exclude_labels,
        )
        return {"items": items, "count": len(items)}
    except Exception as e:
        raise HTTPException(500, str(e))


class UpdateTetraReq(BaseModel):
    content: Optional[str] = None
    labels: Optional[List[str]] = None
    weight: Optional[float] = Field(default=None, ge=0.1, le=10.0)


class ImportReq(BaseModel):
    memories: List[Dict[str, Any]]


@app.get("/api/v1/tetrahedra")
def list_tetrahedra():
    try:
        with _state_lock:
            mesh = _mesh
        items = []
        for tid, t in mesh.tetrahedra.items():
            items.append({
                "id": tid,
                "content": t.content,
                "centroid": t.centroid.tolist() if hasattr(t.centroid, 'tolist') else list(t.centroid),
                "labels": list(t.labels) if t.labels else [],
                "weight": float(t.weight),
                "creation_time": t.creation_time,
                "last_access_time": t.last_access_time,
                "access_count": t.access_count,
                "integration_count": t.integration_count,
                "filtration": float(t.filtration(mesh._time_lambda)),
                "vertex_indices": list(t.vertex_indices),
                "metadata": {k: v for k, v in t.metadata.items() if k in ("type", "source", "fusion_quality", "confidence", "source_clusters")},
            })
        return {"tetrahedra": items, "count": len(items)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/tetrahedra/{tetra_id}")
def get_tetra(tetra_id: str):
    try:
        with _state_lock:
            mesh = _mesh
        t = mesh.get_tetrahedron(tetra_id)
        if t is None:
            raise HTTPException(404, "Tetrahedron not found")
        return {
            "id": t.id,
            "content": t.content,
            "centroid": t.centroid.tolist() if hasattr(t.centroid, 'tolist') else list(t.centroid),
            "labels": list(t.labels) if t.labels else [],
            "weight": float(t.weight),
            "creation_time": t.creation_time,
            "last_access_time": t.last_access_time,
            "access_count": t.access_count,
            "integration_count": t.integration_count,
            "filtration": float(t.filtration(mesh._time_lambda)),
            "vertex_indices": list(t.vertex_indices),
            "metadata": dict(t.metadata) if t.metadata else {},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.put("/api/v1/tetrahedra/{tetra_id}")
def update_tetra(tetra_id: str, req: UpdateTetraReq):
    try:
        with _state_lock:
            mesh = _mesh
        t = mesh.get_tetrahedron(tetra_id)
        if t is None:
            raise HTTPException(404, "Tetrahedron not found")
        if req.content is not None:
            t.content = req.content
        if req.labels is not None:
            old_labels = set(t.labels)
            new_labels = set(req.labels)
            for lbl in old_labels - new_labels:
                mesh._label_index[lbl].discard(tetra_id)
            for lbl in new_labels - old_labels:
                mesh._label_index[lbl].add(tetra_id)
            t.labels = list(req.labels)
        if req.weight is not None:
            t.weight = req.weight
        _schedule_save(dirty_id=tetra_id)
        return {"status": "ok", "id": tetra_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/v1/tetrahedra/{tetra_id}")
def delete_tetra(tetra_id: str):
    try:
        with _state_lock:
            mesh = _mesh
        if mesh.get_tetrahedron(tetra_id) is None:
            raise HTTPException(404, "Tetrahedron not found")
        mesh._remove_tetrahedron(tetra_id)
        _schedule_save()
        return {"status": "ok", "deleted": tetra_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/topology-graph")
def topology_graph():
    try:
        with _state_lock:
            mesh = _mesh
        nodes = []
        for tid, t in mesh.tetrahedra.items():
            nodes.append({
                "id": tid,
                "centroid": t.centroid.tolist() if hasattr(t.centroid, 'tolist') else list(t.centroid),
                "weight": float(t.weight),
                "labels": list(t.labels) if t.labels else [],
                "is_dream": "__dream__" in t.labels,
            })
        edges = []
        seen_pairs = set()
        for tid, t in mesh.tetrahedra.items():
            for fk in mesh._faces_of_tetra(t.vertex_indices):
                face = mesh._faces.get(fk)
                if face:
                    for other_id in face.tetrahedra:
                        if other_id != tid:
                            pair = tuple(sorted([tid, other_id]))
                            if pair not in seen_pairs:
                                seen_pairs.add(pair)
                                shared_verts = set(t.vertex_indices) & set(mesh._tetrahedra[other_id].vertex_indices)
                                conn_type = "face" if len(shared_verts) == 3 else ("edge" if len(shared_verts) == 2 else "vertex")
                                edges.append({"source": pair[0], "target": pair[1], "type": conn_type})
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/import")
def import_memories(req: ImportReq):
    try:
        with _state_lock:
            mesh = _mesh
        imported = []
        for mem in req.memories:
            content = mem.get("content", "")
            if not content:
                continue
            labels = mem.get("labels", [])
            weight = mem.get("weight", 1.0)
            pt = _text_to_point(content, labels=labels)
            tid = mesh.store(
                content=content,
                seed_point=pt,
                labels=labels,
                weight=weight,
                metadata=mem.get("metadata"),
                dedup=True,
            )
            imported.append(tid)
        _schedule_save()
        return {"imported": len(imported), "ids": imported}
    except Exception as e:
        raise HTTPException(500, str(e))


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True))


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("TETRAMEM_HOST", "127.0.0.1")
    port = int(os.environ.get("TETRAMEM_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
