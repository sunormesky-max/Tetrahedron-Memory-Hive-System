"""
TetraMem-XL API v2.4 — TetraMesh + TetraDreamCycle Core
P0 fixes + P1 perf + P2 thread-safety + LLM dream synthesis
"""
import os, time, hashlib, threading, numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from tetrahedron_memory.tetra_mesh import TetraMesh, text_to_geometry
from tetrahedron_memory.tetra_dream import TetraDreamCycle
from tetrahedron_memory.tetra_self_org import TetraSelfOrganizer

STORAGE_DIR = os.environ.get("TETRAMEM_STORAGE", "./tetramem_data_v2")

_mesh: TetraMesh = None
_dream: TetraDreamCycle = None
_self_org: TetraSelfOrganizer = None
_state_lock = threading.RLock()
_save_timer: threading.Timer = None
_save_lock = threading.Lock()
_dirty = False
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
    global _mesh, _dream, _self_org
    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    _mesh = TetraMesh()

    llm = _init_llm_executor()
    _dream = TetraDreamCycle(_mesh, llm_executor=llm)
    _self_org = TetraSelfOrganizer(_mesh)

    index_file = Path(STORAGE_DIR) / "mesh_index.json"
    if index_file.exists():
        import json
        data = json.loads(index_file.read_text())
        for item in data.get("tetrahedra", []):
            pt = np.array(item["centroid"])
            _mesh.store(
                content=item["content"],
                seed_point=pt,
                labels=item.get("labels", []),
                weight=item.get("weight", 1.0),
                metadata=item.get("metadata"),
            )
        print(f"[TetraMem v2] Loaded {len(_mesh.tetrahedra)} tetrahedra")
    else:
        print("[TetraMem v2] Fresh start")


@asynccontextmanager
async def lifespan(application):
    init_state()
    yield
    _flush_save()
    print("[TetraMem v2] Shutdown complete, data flushed")


app = FastAPI(title="TetraMem-XL v2", version="2.4.0", lifespan=lifespan)


class StoreReq(BaseModel):
    content: str
    labels: Optional[List[str]] = None
    weight: float = Field(default=1.0, ge=0.1, le=10.0)
    metadata: Optional[Dict[str, Any]] = None


class QueryReq(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=100)
    labels: Optional[List[str]] = None


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
    import json
    with _state_lock:
        mesh = _mesh
    with _save_lock:
        global _dirty
        tetras = []
        for tid, t in mesh.tetrahedra.items():
            tetras.append({
                "id": tid,
                "content": t.content,
                "centroid": t.centroid.tolist() if hasattr(t.centroid, 'tolist') else list(t.centroid),
                "labels": list(t.labels) if t.labels else [],
                "weight": float(t.weight),
                "metadata": dict(t.metadata) if t.metadata else {},
            })
        Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
        tmp = Path(STORAGE_DIR) / "mesh_index.json.tmp"
        tmp.write_text(json.dumps({"tetrahedra": tetras}, ensure_ascii=False))
        tmp.replace(Path(STORAGE_DIR) / "mesh_index.json")
        _dirty = False


def _schedule_save():
    with _save_lock:
        global _save_timer, _dirty
        _dirty = True
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


def _text_to_point(text: str) -> np.ndarray:
    return text_to_geometry(text)


@app.post("/api/v1/store")
def store(req: StoreReq):
    try:
        with _state_lock:
            mesh = _mesh
        pt = _text_to_point(req.content)
        tid = mesh.store(
            content=req.content,
            seed_point=pt,
            labels=req.labels,
            weight=req.weight,
            metadata=req.metadata,
            dedup=True,
        )
        _schedule_save()
        return {"id": tid}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/query")
def query(req: QueryReq):
    try:
        with _state_lock:
            mesh = _mesh
        pt = _text_to_point(req.query)
        results = mesh.query_topological(pt, k=req.k, labels=req.labels, query_text=req.query)
        items = []
        for tid, dist in results:
            t = mesh.get_tetrahedron(tid)
            if t:
                items.append({
                    "id": tid,
                    "content": t.content,
                    "distance": float(dist),
                    "weight": float(t.weight),
                    "labels": list(t.labels) if t.labels else [],
                })
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
    return {"status": "ok", "version": "2.4.0", "uptime_seconds": time.time() - _start_time}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
