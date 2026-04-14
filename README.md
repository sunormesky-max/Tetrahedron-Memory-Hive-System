# Tetrahedron-Memory-Hive-System

**四面体记忆蜂巢系统** | Tetrahedral Memory Hive System

纯几何驱动 · 永恒记忆 · 自涌现闭环 — 以四面体（3-simplex）为基本单元的下一代 AI 记忆体

[中文文档](./README.cn.md) | [架构文档](./ARCHITECTURE.md) | [演示脚本](./demo_tetramem.py)

## Core Principles

| 原则 | 含义 |
|------|------|
| **Eternity** | 所有记忆永久保留，不删除、不衰减、不覆盖 |
| **Integration** | 记忆持续抽象重组，形成更高阶概念 |
| **Self-Emergence** | 无外部输入时也能自发产生新记忆和新关联 |
| **Closed Loop** | Memory → Think → Execute → Reflect → Integrate → Dream → 循环 |

## Quick Start

```bash
pip install -e ".[all]"
```

```python
import numpy as np
from tetrahedron_memory.tetra_distributed import TetraDistributedController

ctrl = TetraDistributedController(num_buckets=4, use_ray=False)
ctrl.initialize()

# Store memories on tetrahedra
bid, tid = ctrl.store(
    "AI memory should be eternal",
    seed_point=np.array([0.5, 0.0, 0.0]),
    labels=["principle", "ai"],
    weight=1.5,
)

# Pure topological query (no vector embeddings)
results = ctrl.query(np.array([0.5, 0.0, 0.0]), k=5)

# Run dream cycle (self-emergent synthesis)
dream_stats = ctrl.run_dream_cycle(walk_steps=12)

# Run self-organization
org_stats = ctrl.run_self_organization(max_iterations=5)

# View statistics
stats = ctrl.get_statistics()
```

## Architecture

```
TetraDistributedController (unified API)
  └── TetraMeshRouter (spatial partitioning + cross-bucket topology)
        ├── TetraBucket (TetraMesh + Ghost Cells)
        ├── TetraBucket (TetraMesh + Ghost Cells)
        └── ... (auto-split when full)
              └── TetraMesh (dynamic tetrahedral mesh)
                    ├── MemoryTetrahedron (memory on 3-simplex)
                    ├── DreamProtocol (THINK → EXECUTE → REFLECT)
                    └── Self-Organizer (PH-driven geometric surgery)
```

## Key Features

- **Pure geometric retrieval** — BFS along shared faces/edges/vertices, zero vector embeddings
- **Eternal memory** — SHA-256 audit, no deletion, only integration
- **7-dimension fusion quality scoring** — topology connectivity, source depth, centroid dispersion, label diversity, weight balance, content richness, source diversity
- **DreamProtocol** — structured THINK → EXECUTE → REFLECT with quality gate
- **DreamStore** — full provenance tracking, source indexing, quality statistics
- **Distributed** — spatial bucketing + Ghost Cell v2 (versioning + stale detection + batch verification)
- **Multimodal** — image/audio/video → PixHomology → tetrahedral anchors
- **125+ tests, 0 regressions**

## Fusion Quality Score v2 (7 Dimensions)

| Dimension | Weight | Meaning |
|-----------|--------|---------|
| Topological connectivity | 0.20 | Shared labels as topology proxy |
| Source diversity | 0.15 | Number of distinct sources |
| Source depth | 0.15 | integration_count weighting |
| Centroid dispersion | 0.15 | Spatial bridging value |
| Content richness | 0.15 | Synthesis output quality |
| Label diversity | 0.10 | Unique label spread |
| Weight balance | 0.10 | How balanced source weights are |

## DreamProtocol (THINK → EXECUTE → REFLECT)

```python
from tetrahedron_memory.tetra_dream import DreamProtocol

protocol = DreamProtocol(
    think_fn=my_analyzer,      # optional: custom analysis
    execute_fn=my_llm,         # optional: LLM synthesis
    reflect_fn=my_evaluator,   # optional: quality evaluation
    quality_threshold=0.3,     # accept/reject gate
)

result = protocol.run(source_inputs)
if result["accepted"]:
    mesh.store(result["content"], ...)
```

## Demo

```bash
python demo_tetramem.py
```

See [demo_tetramem.py](./demo_tetramem.py) for a full 8-section walkthrough.

## Resources

- [ARCHITECTURE.md](./ARCHITECTURE.md) — detailed architecture, scoring system, eternal audit
- [paper.md](./paper.md) — academic paper draft
- [examples/](./examples/) — usage examples

## License

CC BY-NC 4.0 (Personal & Non-Commercial Use)

**Author**: sunorme (Liu Qihang)
