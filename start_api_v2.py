from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tetrahedron_memory.app_state import AppState
from tetrahedron_memory.routers import memory, agent, system, neural, spatial, darkplane, observer

log = logging.getLogger("tetramem.api")


async def _auth_middleware(request: Request, call_next):
    public_paths = {
        "/", "/health", "/api/v1/health",
        "/docs", "/openapi.json", "/redoc",
    }
    if request.url.path in public_paths or request.url.path.startswith("/ui"):
        return await call_next(request)
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path in ("/api/v1/login", "/api/v1/setup/set-password"):
        return await call_next(request)
    s: AppState = request.app.state.tetramem
    api_key = request.headers.get("X-API-Key", "")
    bearer = request.headers.get("Authorization", "")
    token = bearer.replace("Bearer ", "") if bearer.startswith("Bearer ") else ""
    tenant = None
    if api_key:
        key_hash = s.auth_manager.hash_key(api_key)
        tenant = s.auth_manager.resolve_key(key_hash)
    if tenant is None and token:
        payload = s.auth_manager.validate_token(token)
        if payload:
            tenant = {"tenant_id": payload.get("tenant_id", payload.get("sub", "default")), "role": payload.get("role", "agent")}
    if tenant is None:
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Authentication required for write operations"})
        tenant = {"tenant_id": "default", "role": "anonymous"}
    request.state.tenant = tenant
    return await call_next(request)


@asynccontextmanager
async def lifespan(application):
    state = AppState()
    application.state.tetramem = state
    state.initialize()
    state.register_signal_handlers()
    obs = state.field.observer_ref()
    if obs is not None:
        from tetrahedron_memory.routers.observer import _store_hook
        obs.register_on_store(_store_hook)
    yield
    try:
        if state.persistence is not None and state.persistence.is_dirty():
            with state.state_lock:
                snapshot = state.field.export_full_state()
            state.persistence.checkpoint(snapshot)
            print("[TetraMem v8.0] Final checkpoint completed")
    except Exception as e:
        print(f"[TetraMem v8.0] Final checkpoint failed: {e}")
    finally:
        if state.persistence is not None:
            state.persistence.close()
    state.proactive_engine_stop.set()
    state.field.stop_pulse_engine()
    state.auth_manager.cleanup_expired()
    print("[TetraMem v8.0] Shutdown complete")


app = FastAPI(title="TetraMem-XL v8.0", version="8.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("TETRAMEM_CORS_ORIGINS", "http://localhost:3000,http://localhost:8082").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(_auth_middleware)

app.include_router(memory.router)
app.include_router(agent.router)
app.include_router(system.router)
app.include_router(neural.router)
app.include_router(spatial.router)
app.include_router(darkplane.router)
app.include_router(observer.router)

ui_dir = Path(__file__).parent / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True))
else:
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(static_dir), html=True))

libs_dir = Path(__file__).parent / "libs"
if libs_dir.exists():
    app.mount("/libs", StaticFiles(directory=str(libs_dir)))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("TETRAMEM_PORT", "8000"))
    host = os.environ.get("TETRAMEM_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, log_level="info")
