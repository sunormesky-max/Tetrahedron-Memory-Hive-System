# TetraMem-XL

**Tetrahedral Spatial Memory Body** — Pure geometric-driven, topologically self-organizing next-gen AI memory system

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19429105.svg)](https://doi.org/10.5281/zenodo.19429105)
[![CI](https://github.com/sunormesky-max/sunorm-space-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/sunormesky-max/sunorm-space-memory/actions)

## Vision

TetraMem-XL uses 3-simplices (tetrahedra) as fundamental memory units, replacing flat vector embeddings and cosine similarity with pure geometric primitives + Topological Data Analysis (TDA) for memory storage, retrieval, and association.

Memories live as a dynamic topological structure growing in 3D geometric space, with multi-scale association, dynamic self-organization, and distributed scaling.

## Core Principles

- **Eternity** — No decay/forgetting. All memories are permanent. Noise is transformed through integration, never deleted.
- **Integration** — Dream cycles perform self-association + periodic autonomous integration for abstraction, fusion, and reorganization.
- **Self-Emergence** — The system spontaneously generates new memories, associations, and insights without external input.
- **Closed Loop** — Memory → Think → Execute → Reflect → Integrate → Dream, repeatable from any stage.
- **Spatial Structure** — Large-scale honeycomb memory system with tetrahedra as the core unit.

## Installation

```bash
pip install tetrahedron-memory
```

Optional dependency groups:

```bash
pip install tetrahedron-memory[visualization]   # plotly, matplotlib
pip install tetrahedron-memory[distributed]      # ray
pip install tetrahedron-memory[api]              # fastapi, uvicorn, prometheus-client
pip install tetrahedron-memory[cloud]            # boto3 (S3 storage)
pip install tetrahedron-memory[all]              # everything
```

## Quick Start

```python
from tetrahedron_memory import GeoMemoryBody

memory = GeoMemoryBody(dimension=3, precision="fast")

# Store
memory.store(content="AI memory architecture", labels=["ai", "memory"], weight=1.0)

# Query
results = memory.query("AI memory", k=5)

# Association
associations = memory.associate(memory_id="some_id", max_depth=2)

# Self-organization
stats = memory.self_organize()

# Global integration catalyst
result = memory.global_catalyze_integration(strength=1.0)

# Multi-parameter filtered query
results = memory.query_multiparam("AI memory", k=10, labels_required=["ai"])

# Resolution pyramid query (fast coarse-to-fine)
memory.build_pyramid()
results = memory.query_pyramid("AI memory", k=5)

# Zigzag persistence tracking
snapshot = memory.record_zigzag_snapshot()
prediction = memory.predict_topology()

# Mapping cone: track dream cycle transformations
cones = memory.get_mapping_cone_history()
guidance = memory.get_dream_guidance()

# Eternity audit: prove no memory was ever deleted
report = memory.verify_eternity()
chain = memory.get_eternity_trail("some_id")
```

## Production Deployment

```bash
# Start REST API with persistence
python start_api_persisted.py

# Or use the CLI
tetramem store "my memory" -l topic1 topic2
tetramem query "search text" -k 5
tetramem dream -n 3
tetramem mquery "filtered search" --labels topic1
tetramem pyquery "fast search" -k 10
tetramem zigzag
tetramem predict
tetramem stats
```

## Theoretical Innovations (v2.2)

### Iterative Mapping Cone Modeling
Each dream cycle constructs a **mapping cone** C(f): X_pre → X_post in the Zigzag persistence framework:
- **Forward map**: which features survived, were born, or killed during the dream
- **Backward map**: origin tracing — which pre-dream features gave rise to each post-dream feature
- **Stability certification**: features classified as stable / born / killed / merged per cone
- **Iterative accumulation**: cones chain across dream cycles, forming cumulative stability analysis
- **Dream guidance**: historical cones guide future dreams toward unstable or under-explored regions

### Dynamic Adaptive Resolution Pyramid
The resolution pyramid now features a **closed-loop feedback** mechanism:
- Dream entropy change → feedback → pyramid auto-adjusts max levels and coarsening ratio
- Query hit rate per level → feedback → pyramid adapts granularity
- Positive feedback increases resolution; negative feedback increases coarsening
- The pyramid evolves with the system rather than using static parameters

### Eternity Principle Strict Audit
A formal verification system that **proves no memory was ever deleted**:
- SHA-256 content hash for every memory at every operation
- Transitive preservation chain tracking (s1 → m1 → t1)
- `verify()` scans entire history and proves no violations
- Full audit trail: every store/merge/transform/dream/reintegration is logged with content proof

## REST API

```bash
pip install tetrahedron-memory[api]
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/store` | Store a memory |
| POST | `/api/v1/query` | Text query |
| POST | `/api/v1/query-multiparam` | Multi-parameter filtered query |
| POST | `/api/v1/query-pyramid` | Resolution pyramid query |
| POST | `/api/v1/build-pyramid` | Build resolution pyramid |
| GET | `/api/v1/associate/{id}` | Association retrieval |
| POST | `/api/v1/self-organize` | Trigger self-organization |
| POST | `/api/v1/dream` | Trigger dream cycle |
| POST | `/api/v1/closed-loop` | Run cognitive closed-loop cycle |
| POST | `/api/v1/batch-store` | Batch insert memories |
| POST | `/api/v1/weight-update` | Update memory weight |
| POST | `/api/v1/persist` | Flush to persistent storage |
| GET | `/api/v1/stats` | Memory statistics |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/consistency` | Consistency status |
| POST | `/api/v1/zigzag-snapshot` | Record zigzag persistence snapshot |
| GET | `/api/v1/zigzag-status` | Zigzag tracker status |
| GET | `/api/v1/predict-topology` | Topology prediction |
| GET | `/api/v1/dynamic-barcode` | Dynamic barcode timeline |
| GET | `/metrics` | Prometheus metrics |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TetraMem-XL Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ REST API   │  │ LLM Tools    │  │ Monitoring   │  │ Persistence  │  │
│  │ (20 ep)    │  │ (14 tools)   │  │ (Prometheus) │  │ (Parquet/S3) │  │
│  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│        └────────────────┴─────────────────┴─────────────────┘           │
│                                   │                                     │
│                        ┌──────────▼──────────┐                         │
│                        │   GeoMemoryBody      │                         │
│                        │   (core.py)          │                         │
│                        │  ┌────────────────┐  │                         │
│                        │  │ TetraMesh      │  │                         │
│                        │  │ + Dream Cycle  │  │                         │
│                        │  │ + Self-Org     │  │                         │
│                        │  │ + Emergence    │  │                         │
│                        │  └────────────────┘  │                         │
│                        │  ┌────────────────┐  │                         │
│                        │  │ Zigzag Tracker │  │                         │
│                        │  │ Resolution Pyr │  │                         │
│                        │  │ MultiParam Qry │  │                         │
│                        │  └────────────────┘  │                         │
│                        └──────────┬──────────┘                         │
│             ┌─────────────────────┼──────────────────┐                 │
│  ┌──────────▼──────┐  ┌──────────▼──────────┐  ┌────▼──────────────┐  │
│  │ SpatialBucket    │  │ GlobalCoarseMesh    │  │ Consistency      │  │
│  │ Router           │  │ (feedback loop)     │  │ (VectorClock,    │  │
│  │ (partitioning)   │  │                     │  │  Compensation)   │  │
│  └──────┬───────────┘  └─────────────────────┘  └──────────────────┘  │
│         │                                                               │
│  ┌──────▼──────────────────────────────────────────────────────────┐   │
│  │                 BucketActor Pool (Ray/Local)                     │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐               │   │
│  │  │Bucket 0│  │Bucket 1│  │Bucket 2│  │Bucket N│               │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │          Multimodal Input (PixHomology)                         │   │
│  │    Image | Audio (STFT→MFCC) | Video (Keyframes)               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

| Module | Function |
|--------|----------|
| `core.py` | `GeoMemoryBody` — main engine with mesh, dream, self-org, emergence, zigzag, pyramid integration |
| `tetra_mesh.py` | `TetraMesh` — dynamic tetrahedral mesh with topological navigation |
| `tetra_dream.py` | `TetraDreamCycle` — PH-weighted random walk + deep fusion synthesis |
| `tetra_self_org.py` | `TetraSelfOrganizer` — H0/H1/H2-driven geometric surgery |
| `persistent_entropy.py` | Persistent entropy computation + `EntropyTracker` |
| `closed_loop.py` | `ClosedLoopEngine` — RECALL→THINK→EXECUTE→REFLECT→INTEGRATE→DREAM |
| `emergence.py` | `EmergencePressure` + `AdaptiveThreshold` — self-evolving emergence triggering |
| `zigzag_persistence.py` | `ZigzagTracker` — dynamic topological feature tracking and phase transition detection |
| `resolution_pyramid.py` | `ResolutionPyramid` — multi-scale hierarchical memory representation |
| `multiparameter_filter.py` | `MultiParameterQuery` — 6-dimensional composable filtering engine |
| `geometry.py` | Text→geometry mapping, `TextToGeometryMapper`, geometric primitives |
| `partitioning.py` | `Octree`, `M3NOPartitioner`, `BucketActor`, `SpatialBucketRouter`, `GhostCell` |
| `persistence.py` | `ParquetPersistence` (two-phase commit), `S3Storage`, `RayController` |
| `multimodal.py` | `PixHomology` — image, audio (STFT→MFCC→PH), video |
| `multimodal_bridge.py` | `MultimodalBridge` — connects PixHomology to TetraMesh |
| `router.py` | FastAPI REST API (20 endpoints) |
| `monitoring.py` | Prometheus counters/gauges/histograms + Grafana 15-panel Dashboard |
| `consistency.py` | `VectorClock`, `ConsistencyManager`, `CompensationLog` |
| `llm_tool.py` | OpenAI function calling tools (14 tools) |
| `cli.py` | Stateful CLI with persistence (14 commands) |
| `hooks.py` | Session startup hook for auto-loading |

## Key Algorithms

### 4-Layer Association Rules
1. **Direct Adjacency**: Shared simplices in the topological mesh
2. **Path Connectivity**: Dijkstra on dual-graph with PH-weighted edge costs
3. **Metric Proximity**: `0.5·Euclidean + 0.3·Jaccard + 0.2·VolumeRatio` composite score
4. **PH Patterns**: Persistent interval IoU pattern matching

### PH-Driven Self-Organization

| Dimension | Operation | Trigger |
|-----------|-----------|---------|
| H₀ | Edge Contraction / Merge | Short-lived simplices merge |
| H₁ | Node Repulsion | Persistent loop structures |
| H₂ | Cave Growth + H2 Geometric Repulsion | Void detection + vertex displacement |
| Global | Integration Catalyst | Weight boost (no decay) |

### Emergence Pressure Composite Signal
- Persistent entropy delta (35%)
- H₂ void growth (25%)
- H₁ loop change (15%)
- Local density anomaly (15%)
- Integration staleness (10%)

### Multi-Parameter Filter Dimensions
- **Spatial**: geometric proximity to query point
- **Temporal**: creation/access recency with half-life decay
- **Density**: local neighbor count (cKDTree)
- **Weight**: memory importance score
- **Label**: semantic category matching (required/preferred/penalized)
- **Topology**: integration count + connectivity depth

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
black tetrahedron_memory/
mypy tetrahedron_memory/
```

## Citation

```bibtex
@misc{liu2026pyramidmemory,
  author       = {Liu, Qihang},
  title        = {Pyramid Memory: A Tetrahedron-Based Memory System with O(1) Retrieval Complexity for AI Agents},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19429105},
  url          = {https://doi.org/10.5281/zenodo.19429105}
}
```

## License

CC BY-NC 4.0 - Personal Learning & Non-Commercial Use Only

## Author

Liu Qihang (sunorme) - sunormesky@gmail.com
