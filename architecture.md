# TetraMem-XL Architecture

## System Overview

TetraMem-XL is a production-grade geometric memory system that uses 3-simplices (tetrahedra) as fundamental memory units. It replaces flat vector embeddings with a dynamic topological structure in 3D geometric space.

## Core Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      External Interfaces                             │
│  REST API (21 ep) │ LLM Tools (15) │ CLI (14 cmd) │ Session Hook    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                        GeoMemoryBody (core.py)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  TetraMesh   │  │ DreamCycle   │  │ SelfOrg      │               │
│  │  (mesh.py)   │  │ (dream.py)   │  │ (self_org)   │               │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘               │
│  ┌──────▼──────┐  ┌──────▼───────┐  ┌──────▼───────┐               │
│  │ Persistent  │  │ Emergence    │  │ ClosedLoop   │               │
│  │ Entropy     │  │ Pressure     │  │ Engine       │               │
│  └─────────────┘  │ + Adaptive   │  └──────────────┘               │
│                    │   Threshold  │                                   │
│  ┌─────────────┐  └──────────────┘  ┌──────────────┐               │
│  │ Zigzag +    │                    │ Resolution   │               │
│  │ MappingCone │  ┌──────────────┐  │ Pyramid      │               │
│  └─────────────┘  │ MultiParam   │  └──────────────┘               │
│                    │ Filter       │                                   │
│                    └──────────────┘  ┌──────────────┐               │
│                                      │ Eternity     │               │
│                                      │ Audit        │               │
│                                      └──────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                     Infrastructure Layer                              │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ Consistency│  │ Persistence  │  │ Monitoring   │                │
│  │ (VC+CM+CL) │  │ (Parquet/S3) │  │ (Prometheus) │                │
│  └────────────┘  └──────────────┘  └──────────────┘                │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ Structured │  │ Partitioning │  │ Multimodal   │                │
│  │ Logging    │  │ (Octree/Ray) │  │ (PixHomology)│                │
│  └────────────┘  └──────────────┘  └──────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

## Five Eternal Principles → Implementation Mapping

| Principle | Implementation | Module |
|-----------|---------------|--------|
| **Eternity** | No decay code. `catalyze_integration()` only boosts weights, never reduces below minimum. | `core.py`, `tetra_mesh.py` |
| **Integration** | Dream cycle: PH-weighted walk → semantic fusion → bridge insertion → reintegration. Noise transformed, never deleted. | `tetra_dream.py` |
| **Self-Emergence** | Background daemon: EmergencePressure ≥ AdaptiveThreshold → self-organize → catalyze → dream → record. | `emergence.py`, `core.py` |
| **Closed Loop** | RECALL → THINK → EXECUTE → REFLECT → INTEGRATE → DREAM. Triggerable from any phase. | `closed_loop.py` |
| **Spatial Structure** | TetraMesh: each memory = 1 tetrahedron, growing outward via boundary face attachment. | `tetra_mesh.py` |

## Production Features Mapping

### Phase 1: Core Engine
- [x] TetraMesh with O(1) boundary fast path
- [x] Persistent entropy computation + EntropyTracker
- [x] ClosedLoopEngine with 6-phase cycle
- [x] Dream cycle with LLM-ready callback architecture

### Phase 2: Distributed
- [x] Octree/M3NO spatial partitioning
- [x] BucketActor per partition (Ray/Local)
- [x] Ghost Cell cross-bucket association
- [x] TetraMeshRouter with consistency + persistence

### Phase 3: Persistence & Monitoring
- [x] Parquet two-phase commit (atomic rename)
- [x] S3 backend
- [x] Prometheus 12 metrics + Grafana 15 panels + 4 alert rules
- [x] ConsistencyManager with VectorClock + auto-conflict resolution
- [x] Structured logging with distributed tracing

### Phase 4: Production Validation
- [x] 174+ tests (24 production-grade + 24 GAP-specific + 50 extensions + 30 integration + 27 topology + 19 CLI)
- [x] REST API 21 endpoints
- [x] LLM Tools 15 tools
- [x] CLI 14 commands
- [x] Deploy script with health checks

### P0-P4: Topological Intelligence
- [x] P0: Meaningful dream fusion (semantic label extraction)
- [x] P1: EmergencePressure composite signal (5 components)
- [x] P2: AdaptiveThreshold self-evolution
- [x] P3: Entropy-guided integration priority
- [x] P4: H2 geometric repulsion (vertex displacement)

### Phase 3-4 Extensions
- [x] Zigzag Persistence: sliding window tracking, phase transition detection, topology prediction
- [x] Resolution Pyramid: multi-scale k-means clustering, auto-route coarse→fine
- [x] Multi-Parameter Filter: 6-dimensional composable filtering

### GAP-1: Zigzag Iterative Mapping Cone Modeling
- [x] **MappingConeRecord**: each dream cycle constructs a mapping cone C(f): X_pre → X_post
- [x] Forward map (dream impact): which features survived the dream, with weights
- [x] Backward map (origin tracing): which pre-dream features gave rise to each post-dream feature
- [x] Stability certification: features classified as stable / born / killed / merged per cone
- [x] Iterative accumulation: each dream's cone feeds into the next, forming cumulative stability analysis
- [x] Dream guidance: `get_dream_guidance()` uses historical cone data to recommend focus areas for future dreams
- [x] Integrated into `TetraDreamCycle._execute()` — cones built automatically, no manual triggering

### GAP-2: Dynamic Adaptive Pyramid Closed-Loop Feedback
- [x] `record_dream_feedback()`: dream entropy change → pyramid parameter adjustment
- [x] `record_query_feedback()`: query hit rate tracking per pyramid level
- [x] `_adapt_parameters()`: automatic adjustment of max_levels and coarsening ratio based on feedback
- [x] Positive feedback → increase resolution (more levels, finer granularity)
- [x] Negative feedback → increase coarsening (fewer levels, aggressive clustering)
- [x] Closed loop: dream → feedback → pyramid adapts → next dream uses new pyramid → repeat

### GAP-3: Eternity Principle Strict Audit
- [x] `EternityAudit`: records every store/merge/transform/dream/reintegration operation
- [x] Content hash tracking: SHA-256 fingerprint for every memory content
- [x] Preservation map: tracks transitive content lineage (s1 → m1 → t1 chain)
- [x] `verify(mesh)`: full scan proving no memory was ever deleted — checks mesh liveness + preservation chains
- [x] `get_preservation_chain()`: trace any memory's complete content ancestry
- [x] `get_audit_trail()`: full operation history for any memory ID
- [x] Integrated into `GeoMemoryBody.store()` — every store auto-records audit entry
- [x] Public API: `verify_eternity()`, `get_eternity_status()`, `get_eternity_trail()`

### Production Hardening
- [x] MemoryTetrahedron: `__slots__` + float32 centroids
- [x] Zigzag snapshot compression (old snapshots → entropy-only)
- [x] Version conflict auto-resolution (version priority → timestamp fallback)
- [x] Multi-bucket read repair
- [x] Zigzag + threshold state persistence as meta-dream memories
- [x] Topology health endpoint

## Data Flow

### Store Flow
```
content → TextToGeometry → seed_point → TetraMesh.store() → boundary attach
    → ConsistencyManager.record_version() → auto_persist (batched)
```

### Query Flow
```
query_text → TextToGeometry → query_point
    → Mesh: TetraMesh.query_topological()
    → Legacy: Alpha Complex + spatial candidates + 4-layer association
```

### Dream Flow
```
EmergencePressure ≥ AdaptiveThreshold
    → self_organize() [H0/H1/H2 surgery]
    → catalyze_integration() [weight boost]
    → DreamCycle.trigger_now() [walk → cluster → synthesize → insert]
    → MappingCone C(f): X_pre → X_post [forward + backward + stability]
    → ZigzagTracker.record_snapshot() [topology tracking]
    → ResolutionPyramid.record_dream_feedback() [adaptive parameters]
    → EternityAudit.record_dream() + record_reintegration() [eternity proof]
    → persist_zigzag_snapshot() [meta-dream memory]
    → persist_emergence_state() [threshold history]
```

## Consistency Model

```
Write: validate_before_write() → record_version() → conflict auto-resolution
Read:  query() → read_repair_multi() if stale detected
Compensation: compensate_operation() → retry_all() on recovery
```

Conflict resolution priority:
1. Higher version wins
2. Newer timestamp wins (if versions equal)
3. Auto-logged in conflict_history
