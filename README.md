# TetraMem-XL

**An Eternal, Self-Aware Memory System for AI Agents**

Memory that never forgets. That watches itself. That dreams.

[中文文档](./README.cn.md) | [Integration Guide](./INTEGRATION_GUIDE.md) | [Agent Guide](./AGENTS.md) | [Changelog](./CHANGELOG.md)

---

## What is this?

TetraMem-XL is a memory system designed for AI agents that need more than a vector database.

- **Eternal** — Memories are never deleted. Only consolidated and recombined.
- **Self-Organizing** — Memories cluster, migrate, and form pathways autonomously.
- **Self-Aware** — The system observes its own runtime behavior and stores self-reflection memories.
- **Alive** — Dream cycles create cross-domain associations. Pulse waves propagate activation. The memory landscape evolves even without input.

Under the hood: a novel fusion of crystal geometry, neural pulse propagation, and topological analysis — all in pure Python.

---

## Quick Start

### One-Click Install (Linux)

```bash
curl -sSL https://raw.githubusercontent.com/sunormesky-max/Tetrahedron-Memory-Hive-System/main/install.sh | bash
```

### Docker

```bash
git clone https://github.com/sunormesky-max/Tetrahedron-Memory-Hive-System.git
cd Tetrahedron-Memory-Hive-System
TETRAMEM_UI_PASSWORD=mypass docker compose up -d
```

### Python

```bash
pip install -e ".[dev]"
```

```python
import requests

BASE = "http://localhost:8000/api/v1"
H = {"X-API-Key": "your-key", "Content-Type": "application/json"}

# Store a memory
r = requests.post(f"{BASE}/store", json={
    "content": "AI memory should be eternal",
    "labels": ["principle", "ai"],
    "weight": 1.5,
}, headers=H)
print(r.json())  # {"id": "abc123..."}

# Query memories
r = requests.post(f"{BASE}/query", json={"query": "eternal memory", "k": 5}, headers=H)
for m in r.json()["results"]:
    print(f"  {m['content'][:50]}... score={m['distance']:.3f}")

# Let it dream
r = requests.post(f"{BASE}/dream", headers=H)

# Trigger self-organization
r = requests.post(f"{BASE}/self-organize", headers=H)
```

---

## Core Features

| Feature | Description |
|---------|-------------|
| **Eternal Memory** | No deletion, no forgetting. Only consolidation and abstraction. |
| **Self-Organizing** | Autonomous clustering, pathway formation, and memory migration. |
| **Dream Engine** | Cross-domain creative recombination — the system generates new associations while "sleeping". |
| **Self-Reflection** | RuntimeObserver watches system behavior, classifies events, and stores trajectory memories. |
| **Adaptive Regulation** | Six-layer physiological control system maintains memory health autonomously. |
| **Topological Analysis** | High-dimensional topological invariants drive the memory energy landscape. |
| **Attention Mechanism** | Focus, diffuse, decay — controllable attention over the memory space. |
| **144 REST Endpoints** | Full API for store, query, dream, organize, observe, regulate, and more. |
| **MCP Server** | 17 tools for Claude Desktop / Cursor integration out of the box. |
| **3D Visualization** | Real-time Three.js honeycomb crystal browser. |

---

## Performance

| Scale | Store | Query | Memory |
|-------|-------|-------|--------|
| 500 nodes | 408 ops/s | 3.8ms | ~36MB |
| 1,000 nodes | 286 ops/s | 45ms | ~72MB |
| 5,000 nodes | ~200 ops/s | 173ms | ~1.2GB |

Zero external engine dependencies. Pure Python + NumPy.

---

## API at a Glance

### Core Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/store` | POST | Store a memory |
| `/query` | POST | Semantic query |
| `/browse` | GET | Browse memories |
| `/tetrahedra/{id}` | GET/DELETE | Get/delete single memory |
| `/stats` | GET | System statistics |
| `/health` | GET | Health check |

### Autonomous Systems

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dream` | POST | Trigger dream cycle |
| `/self-organize` | POST | Trigger self-organization |
| `/dark-plane/flow` | POST | Run energy landscape evolution |
| `/dark-plane/stats` | GET | Energy landscape statistics |
| `/regulation/trigger` | POST | Trigger self-regulation |
| `/cascade/trigger` | POST | Trigger pulse cascade |
| `/observer/stats` | GET | Self-reflection statistics |

### Full API

See [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) for complete endpoint documentation.

---

## Integration Options

### For External AI Agents

Point TetraMem at your log file — it auto-classifies events and distills them into trajectory memories:

```python
from tetrahedron_memory.observer_config import auto_attach
observer = auto_attach(field, config_path="./observer_config.json")
```

### For Embedded Agents

Inject events directly from your agent code:

```python
from tetrahedron_memory.runtime_observer import attach_callback_observer
observer = attach_callback_observer(field)
observer.observe("ERROR", "my_agent", "Connection timeout")
```

### For LLM Tools (MCP)

Built-in Model Context Protocol server with 17 tools — works with Claude Desktop, Cursor, and any MCP-compatible client.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TETRAMEM_STORAGE` | `./tetramem_data_v2` | Data directory |
| `TETRAMEM_UI_PASSWORD` | `CHANGE_ME` | UI login password |
| `TETRAMEM_PORT` | `8000` | API port |
| `TETRAMEM_CORS_ORIGINS` | `http://localhost:3000,http://localhost:8082` | CORS origins |

---

## UI

- **Embedded**: `http://<host>:8000/ui/`
- **Standalone Dashboard**: Open `ui/dashboard.html` in any browser

---

## License

AGPL-3.0-or-later

**Author**: sunorme (刘启航)
**Email**: sunormesky@gmail.com
