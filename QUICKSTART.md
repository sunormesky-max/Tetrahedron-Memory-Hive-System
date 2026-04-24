# TetraMem-XL Quick Start

Get from zero to a living, dreaming, topology-aware memory system in **5 minutes**.

---

## Install

```bash
pip install tetrahedron-memory
```

Core only depends on `numpy`, `scipy`, and `gudhi`. Everything else is optional.

---

## 30-Second Demo

```python
from tetrahedron_memory import GeoMemoryBody

# Create a 3D memory organism
memory = GeoMemoryBody(dimension=3, precision="fast")

# Store memories as living tetrahedral cells
memory.store("I learned about persistent homology today", labels=["math", "topology"], weight=1.0)
memory.store("H1 detects loops and tunnels in data", labels=["math", "homology"], weight=0.8)
memory.store("Alpha complexes are weighted Delaunay triangulations", labels=["geometry"], weight=0.9)

# Query by semantic similarity (topological navigation, not cosine!)
results = memory.query("topology and loops", k=3)
for r in results:
    print(f"  [{r['score']:.3f}] {r['content']}")

# Watch it think — self-organize based on topological pressure
memory.self_organize()

# Prove no memory was ever deleted
audit = memory.verify_eternity()
print(f"Eternity verified: {audit['verified']}, memories alive: {audit['total_alive']}")
```

---

## Core Concepts in 60 Seconds

### Memory = Geometry

Traditional AI memory: flat vector → cosine similarity → done.

TetraMem: text → centroid in 3D → tetrahedron grows → mesh self-organizes → dreams fuse memories → emergent abstractions form.

### The Dream Cycle

```python
# Trigger an autonomous dream cycle
dream_report = memory.dream()

print(f"Memories fused: {dream_report['fused_count']}")
print(f"New abstractions: {dream_report['emerged_count']}")
print(f"Mapping cone stability: {dream_report['cone_stability']:.3f}")
```

Dreams use **Persistent Homology weights** to decide which memories to fuse. The mapping cone `C(f): X_pre → X_post` tracks exactly what changed.

### Eternity Audit

Every operation (store, merge, transform, dream) is logged with a SHA-256 content hash. The preservation chain is transitive and verifiable:

```python
audit = memory.verify_eternity()
# {'verified': True, 'total_stored': 100, 'total_alive': 100, 'violations': []}
```

---

## REST API

```bash
# Install with API support
pip install tetrahedron-memory[api]

# Start the server
python -m tetrahedron_memory.router
```

```python
import httpx

# Store
r = httpx.post("http://localhost:8000/memory", json={
    "content": "Hello from the API",
    "labels": ["test"],
    "weight": 1.0
})
memory_id = r.json()["memory_id"]

# Query
r = httpx.post("http://localhost:8000/memory/query", json={
    "query": "hello",
    "k": 5
})

# Get by ID
r = httpx.get(f"http://localhost:8000/memory/{memory_id}")

# Update
r = httpx.put(f"http://localhost:8000/memory/{memory_id}", json={
    "content": "Updated content",
    "weight": 1.5
})

# Health check
r = httpx.get("http://localhost:8000/health")
```

**20 endpoints** available: CRUD, query (single/batch/by-label), dream, self-organize, eternity audit, metrics, and more.

---

## LLM Tool Integration

TetraMem ships with **17 ready-to-use LLM tools** in OpenAI function-calling format:

```python
from tetrahedron_memory.llm_tool import list_tools

tools = list_tools()
# Use directly with OpenAI, Anthropic, or any function-calling LLM
```

---

## MCP Server (for Agent Frameworks)

```bash
# Install
pip install tetrahedron-memory

# Run as MCP stdio server
python -m tetrahedron_memory.mcp_server
```

Provides 6 tools: `tetramem_store`, `tetramem_query`, `tetramem_get`, `tetramem_update`, `tetramem_dream`, `tetramem_health`.

---

## Persistence

```python
from tetrahedron_memory import GeoMemoryBody
from tetrahedron_memory.persistence import ParquetPersistence

# Save to disk
persistence = ParquetPersistence(storage_path="./my_memory_data")
memory = GeoMemoryBody(dimension=3, persistence=persistence)

# Memories are automatically persisted after operations
memory.store("This will survive restart", labels=["persistent"])

# Later, load back
memory2 = GeoMemoryBody(dimension=3, persistence=ParquetPersistence(storage_path="./my_memory_data"))
```

---

## What to Explore Next

| Topic | Where |
|-------|-------|
| Full API reference | `README.md` |
| Architecture deep-dive | `REPORT.md` |
| Contributing | `CONTRIBUTING.md` |
| Mathematical foundation | `REPORT.md` → Theoretical Framework |
| All test examples | `tests/` directory |
| Dream cycle internals | `tetrahedron_memory/tetra_dream.py` |
| Mapping cone tracking | `tetrahedron_memory/zigzag_persistence.py` |
| Eternity audit system | `tetrahedron_memory/eternity_audit.py` |

---

## Theoretical Foundation

TetraMem-XL is built on peer-reviewed mathematics:

- **Persistent Homology** — tracking topological features (H₀/H₁/H₂) across scales
- **Alpha Complex** — weighted Delaunay triangulation for memory geometry
- **Zigzag Persistence** — dynamic feature tracking through time
- **Mapping Cones** — algebraic topology for dream cycle analysis
- **Persistent Entropy** — information-theoretic emergence detection

**DOI**: [10.5281/zenodo.19429105](https://doi.org/10.5281/zenodo.19429105)

---

## License

CC BY-NC 4.0 — free for personal learning and non-commercial use.

---

> *"The universe is built on geometry. Why shouldn't memory be?"*
