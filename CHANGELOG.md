# Changelog

All notable changes to TetraMem-XL are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [5.1.0] - 2026-04-18

### Added — Self-Organization Engine + Cluster Detection + Entropy Balance + Consolidation + Shortcuts

#### SelfOrganizeEngine
- Unified self-organization engine with four autonomous modules per cycle
- Runs every 180 pulse cycles (configurable via `SELF_ORGANIZE_INTERVAL`)
- `OrganizeResult` — structured report per cycle: clusters, entropy, consolidations, shortcuts

#### Cluster Detection
- Label-cooccurrence grouping: memories sharing labels are clustered together
- Spatial proximity verification: clusters validated by geometric centroid analysis
- `SemanticCluster` data class with cluster_id, labels, node_ids, centroid, avg_weight
- Automatic intra-cluster Hebbian reinforcement for weakly-connected cluster members
- Minimum cluster size = 3 nodes, maximum 8 cluster labels analyzed per cycle

#### Entropy Balance
- Shannon entropy computed over weight distribution (normalized)
- When weight concentration ratio > 5.0 (one node >> average):
  - High-weight nodes suppressed by 10% (weight reflows to system)
  - Low-weight nodes boosted by 20% of average weight
- Preserves total system weight while preventing dominance

#### Memory Consolidation
- Low-weight duplicates (weight < 1.0, similarity >= 65%) automatically merged
- Higher-weight memory absorbs lower-weight memory's weight (60% transfer), labels, activation
- Donor memory marked `__consolidated__` with `consolidated_into` metadata
- Maximum 5 consolidations per cycle to avoid burst disruption
- Hebbian path recorded between consolidator and consolidated

#### Topological Shortcuts
- Virtual connections between semantically close but topologically distant memory pairs
- Criteria: ≥2 shared labels, topological distance 3-6 hops, not already direct neighbors
- Shortcut strength = base * shared_labels / total_unique_labels
- Hebbian path recorded at 2x strength to create lasting bias
- Maximum 3 shortcuts per cycle, 500 total with LRU eviction

#### New API Endpoints (5)
- `POST /api/v1/self-organize/run` — trigger self-organization cycle
- `GET /api/v1/self-organize/status` — engine status, clusters, shortcuts, entropy
- `GET /api/v1/self-organize/history` — previous organize cycle results
- `GET /api/v1/clusters` — detected semantic clusters
- `GET /api/v1/shortcuts` — topological shortcuts

#### MCP Tool Server v5.1.0
- 5 new tools: self_organize_run, self_organize_status, clusters, shortcuts
- Total: 40 tools

#### Visualization UI
- New "自组织" (Self-Organization) panel with cluster display, shortcut table, entropy gauge
- Organize history table with per-cycle metrics

### Changed
- `stats()` now includes `self_organize` section
- Pulse engine startup log updated to v5.1

## [5.0.0] - 2026-04-18

### Added — Structural Cascade + Lattice Integrity + Crystallized Pathways

#### Cascade Pulse Engine
- `PulseType.CASCADE` — multi-directional branching propagation
- At each hop, cascade spawns K child pulses (K = branching factor = 3)
- Energy conservation: total child energy ≤ parent * conservation_factor * branching_decay
- Max cascade depth = 4 levels before falling back to single-path propagation
- `trigger_cascade()` API for manual cascade triggering
- `_propagate_cascade()` recursive multi-path propagation
- Crystal channel boost during cascade (1.8x transmission through crystallized paths)
- `_bias_cascade()` — biased toward high-weight nodes and crystallized paths

#### Lattice Integrity Verification
- `LatticeIntegrityChecker` — BCC crystal structure verification engine
- Bidirectionality check: all edges verified as bidirectional
- Orphan node detection: nodes with zero neighbors
- Coordination number verification: body-center = 8 face neighbors, corner = 8 face + 6 edge
- Connectivity analysis: BFS-based connected component counting
- Occupied node health check: weight/activation bounds, crystal channel validity
- `LatticeIntegrityReport` — detailed report with integrity score (0-1)
- Integrity score formula: 1.0 - (critical_errors * 0.5 + coord_ratio * 0.5)
- Automatic periodic check every 600 pulse cycles
- `PulseType.STRUCTURE` — structure pulse biased toward low-connectivity regions

#### Crystallized Pathways
- `CrystallizedPathway` — permanent structural fast-path management
- When Hebbian edge weight exceeds threshold (3.0), edge crystallizes
- Crystal channels: zero-decay, permanently reinforced pulse conduits
- Max 200 crystals with LRU eviction when capacity exceeded
- Crystal boost factor: 1.8x pulse transmission through crystal channels
- `crystal_channels` per-node index for fast crystal lookup
- Automatic crystallization maintenance every 90 pulse cycles
- `force_crystallize()` for manual crystallization scan

#### New Pulse Types
- `PulseType.CASCADE` (15% probability) — multi-directional branching wave
- `PulseType.STRUCTURE` (7% probability) — lattice structure reinforcement

#### New API Endpoints (7)
- `GET /api/v1/lattice-integrity/check` — run full BCC lattice integrity verification
- `GET /api/v1/lattice-integrity/status` — checker status and history
- `GET /api/v1/lattice-integrity/history` — previous integrity reports
- `GET /api/v1/crystallized/status` — crystallized pathway stats and top crystals
- `POST /api/v1/cascade/trigger` — trigger cascade pulse wave
- `POST /api/v1/structure-pulse/trigger` — trigger structure reinforcement pulse
- `POST /api/v1/crystallized/force` — force crystallization maintenance scan

#### MCP Tool Server v5.0.0
- 7 new tools: lattice_check, lattice_status, lattice_history, cascade_trigger, structure_pulse, crystallized_status, force_crystallize
- Total: 35 tools covering all v5.0 endpoints

#### Visualization UI
- New "级联脉冲" (Cascade Pulse) panel with trigger and crystal display
- New "晶格完整性" (Lattice Integrity) panel with score and error details
- New "结晶通路" (Crystallized Pathways) panel with top crystals table
- Pulse type display updated: cascade (blue), structure (green) badges

### Changed
- `HoneycombNode` gains `crystal_channels` slot for per-node crystal index
- `NeuralPulse` gains `cascade_depth` and `cascade_parent_id` for cascade tracking
- `pulse_status()` now includes cascade_count and crystallized stats
- `stats()` now includes cascade_count, crystal_nodes, lattice_integrity, crystallized stats
- PCNN pulse engine startup log updated to v5.0

## [4.1.0] - 2026-04-18

### Added — Self-Check Pulse Engine + Duplicate & Isolated Detection

#### Self-Check Pulse Engine
- `SelfCheckEngine` — periodic autonomous inspection engine (60s cycle)
- Detects orphan nodes, duplicate memories, low-vitality nodes
- Auto-repair: trigger reinforcement pulses for isolated memories
- `PulseType.SELF_CHECK` — new pulse type with dedicated bias function and source selector
- 6 new API endpoints: `/self-check/status`, `/self-check/run`, `/self-check/history`, `/duplicates`, `/isolated`, `/self-check/trigger`

#### Duplicate Detection
- `detect_duplicates()` — Jaccard similarity >= 70% memory pair discovery
- Lower-weight duplicate auto-annotated with `__duplicate_of__` metadata
- Memory integrity preserved (no deletion, only annotation)

#### Isolated Memory Detection
- `detect_isolated()` — real-time detection of memories with no face/edge neighbors
- Empty association detection during SELF_CHECK pulse propagation

#### MCP Tool Server v4.1.0
- Expanded from 12 to 28 tools covering all v4.0/v4.1 API endpoints
- Tool descriptions updated to BCC lattice / PCNN terminology

#### Visualization UI
- New "Self-Check Pulse" panel with status display and manual trigger
- Self-check pulse type badge in pulse monitor

## [4.0.1] - 2026-04-17

### Fixed
- Pulse accumulator four-channel clamping fix — prevents overflow in high-frequency pulse trains

## [4.0.0] - 2026-04-16

### Added — PCNN Pulse Engine + BCC Honeycomb Architecture

#### PCNN Pulse-Coupled Neural Network Engine
- `PulseType` enum: EXPLORATORY, REINFORCEMENT, TENSION, SELF_CHECK
- Pulse emission, propagation, and Hebbian path recording
- Multi-type pulse bias functions and source selectors
- Face decay=0.72, edge decay=0.36, beta=0.4 parameters
- 4 PCNN API endpoints: `/pulse/emit`, `/pulse/status`, `/pulse/history`, `/hebbian/stats`

#### Hebbian Path Memory
- Path recording with success/failure tracking
- Synaptic strength modulation based on path history

#### Phase Transition Detection
- `phase_transition_honeycomb.py` — Honeycomb-compatible phase transition analysis
- BCC lattice-specific topological feature extraction

### Changed
- Core architecture migrated from TetraMesh to BCC Lattice Honeycomb (`HoneycombNeuralField`)
- API backend upgraded to v4.0 with PCNN-integrated endpoints

## [3.0.0] - 2026-04-14

### Added — BCC Lattice Honeycomb Neural Field Foundation

#### BCC Crystal Lattice
- Body-Centered Cubic lattice generation with configurable resolution and spacing
- Face-sharing and edge-sharing neighbor topology
- `HoneycombNeuralField` — new core class replacing TetraMesh for memory operations

#### Neural Field Core
- `store()`, `query()`, `browse_timeline()`, `stats()` — full memory lifecycle
- Label-based filtering and weight-based ranking
- JSON persistence (`mesh_index.json`)

#### FastAPI Backend v2
- `start_api_v2.py` — new FastAPI application with RESTful endpoints
- Memory CRUD, timeline browsing, statistics, export/import
- Nginx reverse proxy integration (port 8082)

#### Visualization UI
- Web-based 3D topology visualization
- Login authentication (tetramem / Hive@2026)
- Chinese-localized interface (only "TetraMem-XL" in English)

## [2.2.0] - 2026-04-12

### Added — Theoretical Innovation: Mapping Cone + Adaptive Feedback + Eternity Audit

#### GAP-1: Zigzag Iterative Mapping Cone Modeling
- `MappingConeRecord` — data class capturing forward map, backward map, and stability certification per dream cycle
- `ZigzagTracker.construct_mapping_cone()` — builds C(f): X_pre → X_post with feature-level birth/death/stability analysis
- `ZigzagTracker.get_mapping_cone_history()` — full iterative cone chain across all dream cycles
- `ZigzagTracker.get_dream_guidance()` — uses historical cone data to recommend focus areas for future dreams
- `TetraDreamCycle._execute()` — automatically constructs mapping cone each cycle, feeds guidance to next dream
- `_synthesize_and_insert_tracked()` / `_reintegrate_dreams_tracked()` — return ID lists for cone tracking

#### GAP-2: Dynamic Adaptive Pyramid Closed-Loop Feedback
- `ResolutionPyramid.record_dream_feedback()` — dream entropy change → pyramid parameter adjustment
- `ResolutionPyramid.record_query_feedback()` — query hit rate tracking per level
- `ResolutionPyramid._adapt_parameters()` — automatic max_levels and coarsening ratio based on feedback history
- Closed loop: dream effect → feedback → pyramid adapts → next dream uses updated pyramid → repeat

#### GAP-3: Eternity Principle Strict Audit
- `eternity_audit.py` — new module: `EternityAudit` class for strict no-deletion verification
- Content hash tracking (SHA-256) for every memory, transitive preservation chain propagation
- `verify(mesh)` — full scan proving no memory was ever lost; checks liveness + preservation chains
- `get_preservation_chain()` — trace any memory's complete content ancestry (s1 → m1 → t1)
- `get_audit_trail()` — full operation history for any memory ID
- Integrated into `GeoMemoryBody.store()` — automatic audit recording

#### GAP-Specific Tests
- `tests/test_gap_critical.py` — 24 tests covering all three GAPs

### Fixed — P0/P1 Audit
- P0-1: emergence loop split into snapshot→operate→persist phases (lock hold time reduced)
- P0-2: vertex leak fixed with reference counting `_vertex_ref_count` + auto-compaction
- P0-3: `_nodes` property now acquires `mesh._lock` + returns centroid.copy()
- P0-4: multimodal variable shadowing fixed (`persistence` → `length`)
- P0-5: dream reintegration threshold fixed (0.85 decay + dynamic threshold)
- P1-6: all 5 lockless modules now have RLock protection
- P1-7: 15+ bare except replaced with `except Exception as e:` + logging
- P1-8: topology_shortcuts per-node cap at 50
- P1-9: LLM Tool returns error on empty instance instead of creating one

## [2.1.0] - 2026-04-12

### Added — Production Hardening (4-Stage Production Roadmap)

#### Stage 1: Stability & Consistency
- `ConsistencyManager.validate_before_write()` — write-before-validate with automatic conflict resolution (version priority → timestamp fallback)
- `ConflictRecord` — full conflict history tracking with auto-resolution status
- `ConsistencyManager.read_repair_multi()` — batch multi-bucket staleness repair
- `ConsistencyManager.compensate_operation()` — structured failure compensation with logging
- `ConsistencyManager.get_health()` — comprehensive consistency health reporting
- Zigzag snapshot persistence — `_persist_zigzag_snapshot()` records topology state as meta-dream memory after each emergence cycle
- Emergence state persistence — `_persist_emergence_state()` saves entropy/threshold history as mesh memories
- `GET /api/v1/health/topology` — topology health endpoint returning entropy, H₂ voids, zigzag stability, threshold status, consistency health

#### Stage 2: Performance Optimization
- `MemoryTetrahedron.__slots__` — reduced memory footprint per tetrahedron
- `MemoryTetrahedron.centroid` — float32 (was float64), halving centroid memory
- `catalyze_integration_batch()` — removed outer lock for parallel execution
- Zigzag snapshot compression — old snapshots auto-compressed to entropy-only (barcodes discarded)
- Pyramid `auto_route()` — bbox-based spatial exclusion for fast coarse→fine queries

#### Stage 3: Observability
- `structured_log.py` — new module: `StructuredLogger` with JSON output + `trace_context()` for distributed tracing
- `get_alert_rules()` — 4 Prometheus alert rules: entropy spike, error rate, store latency, query latency
- Nested trace context support — `trace_context()` supports nesting with stack-based tracking

#### Stage 4: Final Validation & Deployment
- 24 production-grade tests covering: eternity principle (4), integration quality (2), closed loop (2), performance (2), consistency (5), zigzag stability (2), pyramid stability (1), multi-param (2), structured logging (3)
- `deploy.sh` — 5-step automated deployment with testing, Ray, API, health check, monitoring
- Circuit breaker + rate limiter for emergence protection
- Hot tetrahedron query cache (LRU)
- 72-hour stress test framework

#### Documentation
- `README.md` — comprehensive English version (21 API endpoints, 15 LLM tools, 14 CLI commands, 24 modules)
- `README.cn.md` — full Chinese version
- `architecture.md` — architecture document with principle→implementation mapping, data flow diagrams, consistency model
- `deployment_guide.md` — 5-step deployment guide with monitoring configuration and fault recovery procedures
- `paper.md` — academic paper with methodology, experimental results, and references

### Changed
- `consistency.py` — `VersionedNode` gains `operation` field; `record_version()` accepts `operation` parameter; conflict auto-resolution on every store/dream/integrate
- `core.py` — `_record_version()` passes `operation` parameter; emergence loop calls `_persist_zigzag_snapshot()` and `_persist_emergence_state()`
- `tetra_mesh.py` — `MemoryTetrahedron` uses `__slots__` + manual `__init__` + float32 centroids; ID generation includes timestamp+counter to prevent collisions
- `monitoring.py` — added `ALERT_RULES`, `GRAFANA_ALERT_GROUPS`, `get_alert_rules()`

## [2.0.0] - 2026-04-12

### Added

#### Phase 1: Core Engine Productionization
- Removed all decay/forgetting code, replaced with integration catalyst (Eternity principle)
- New modules: `persistent_entropy.py`, `closed_loop.py`
- Time law changed to integration catalyst
- Dream cycle upgraded with LLM-ready `DreamSynthesisInput` callback architecture
- Self-organizing entropy convergence with `EntropyTracker`

#### Phase 2: Distributed & Partitioning
- `tetra_router.py` — `TetraMeshRouter` with 4 runtime bug fixes
- `partitioning.py` — Ray fallback fix, consistency + persistence integration
- `multimodal_bridge.py` — `query_by_modality()` for filtered multimodal queries

#### Phase 3: Persistence, Consistency, Monitoring
- `persistence.py` — Parquet two-phase commit with atomic rename, upsert fix
- `monitoring.py` — 5 new metrics (DREAM_COUNTER, ENTROPY_GAUGE, INTEGRATION_COUNTER, STORE_LATENCY, QUERY_LATENCY), Grafana 15-panel Dashboard
- `consistency.py` — `ConsistencyManager` with `List[VersionedNode]`, VectorClock, CompensationLog
- `global_coarse_mesh.py` — `GlobalCoarseMesh` with `_apply_corrections` feedback loop

#### Phase 4: Production Validation & Optimization
- TetraMesh performance optimization: O(1) boundary fast path, pure-Python hot path
- 21 production validation tests
- ConsistencyManager integrated into GeoMemoryBody + TetraMeshRouter + BucketActor
- Persistence integrated into GeoMemoryBody (auto-save/load + compact)
- Self-emergence daemon (background thread)
- GlobalCoarseMesh feedback loop (auto-correction)
- REST API expanded to 20 endpoints
- LLM tools expanded to 14 tools
- `store_batch` consistency/persistence hooks

#### P0-P4: Topological Intelligence Features
- **P0: Meaningful Dream Fusion** — Semantic fusion with label intersection extraction, weight-weighted ranking, topology bridge description, depth tags
- **P1: Emergence Pressure Composite** — `EmergencePressure` class: persistent entropy rate + H₂ void growth + H₁ loop change + local density anomaly + integration staleness
- **P2: Adaptive Threshold Evolution** — `AdaptiveThreshold`: good effect → lower threshold → encourage emergence; poor effect → raise threshold → avoid waste
- **P3: Entropy-Guided Integration Priority** — Dream walk prefers high label diversity, low weight regions
- **P4: H2 Geometric Repulsion** — Actual vertex displacement with linear force decay, stability theorem guaranteed bounded perturbation

#### Phase 3-4 Extensions
- **Zigzag Persistence Dynamic Modeling** — `ZigzagTracker`: sliding window topological feature tracking, phase transition detection, feature lifetime analysis, topology prediction
- **Resolution Pyramid** — `ResolutionPyramid`: multi-scale hierarchical clustering, auto-route coarse-to-fine queries, k-means spatial coarsening
- **Multi-Parameter Filter** — `MultiParameterQuery`: 6-dimensional composable filtering (spatial, temporal, density, weight, label, topology), hard/soft filter modes

#### Infrastructure
- Stateful CLI with persistence (14 commands: store, query, label, stats, clear, persist, dream, self-org, catalyze, status, mquery, build-pyramid, pyquery, zigzag, predict)
- Session hook (`hooks.py`) with mesh-mode-aware loading
- `tetramem_sync.py` export utility

### Fixed
- `hooks.py` mesh mode: loaded nodes now go through `_mesh.store()` instead of vanishing via `_nodes` setter
- `multimodal_bridge.py` sort bug: `query_by_modality()` without query_point returns highest-weight memories first
- `start_api_persisted.py` mesh mode: same fix as hooks.py
- `tetra_mesh.py` ID collision: same content no longer silently overwrites (ID includes timestamp + counter)
- `tetra_mesh.py` duplicate `_boundary_dirty` assignment removed
- `__init__.py` `TextToGeometryMapper` no longer shadowed by multimodal stub
- `__init__.py` `generate_prometheus_metrics` removed from `__all__` (conditionally imported)
- Removed unused `networkx` dependency from `pyproject.toml`

### Removed
- Dead code: `alpha.py`, `shim.py`, `mapping.py`, `rollback.py`, `rollback_runbook.py`, `legacy_v2/` directory
- Dead test files: 9 test files testing removed dead modules
- Artifacts: `.bak` file, `.db` files, `pyramid_memory/` cache directory

### Changed
- All `load_from_persistence` / `hooks.py` / `start_api_persisted.py` now correctly use `_mesh.store()` in mesh mode
- Hardcoded absolute paths replaced with `os.path.expanduser()` defaults + environment variable overrides
- `tetramem_sync.py` export path now configurable via `TETRAMEM_EXPORT` env var

## [1.0.0] - 2025-12-01

### Added

- Geometric memory storage using 3D point clouds on unit sphere
- GUDHI Alpha Complex-based topological structure
- Persistent Homology for topological feature extraction (H0, H1)
- 4-layer association rules (direct adjacency, path connectivity, metric proximity, self-organizing)
- Text-to-geometry mapping via deterministic hashing
- Dynamic weight updates with EMA smoothing
- Memory conflict detection
- JSON-based persistence
- Thread-safe operations
- Octree spatial indexing
- `MemoryNode` dataclass with id, content, geometry, timestamp, weight, labels, metadata
- `GeoMemoryBody` main interface (store, query, query_by_label, associate, update_weight, detect_conflicts)
