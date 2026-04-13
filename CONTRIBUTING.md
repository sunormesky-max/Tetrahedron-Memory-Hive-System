# Contributing to TetraMem-XL

Welcome, topologist! We're building the world's first topology-aware AI memory system, and we'd love your help.

> **"The universe is built on geometry. Why shouldn't memory be?"**

This project is research-driven, so we value **correctness**, **mathematical rigor**, and **reproducibility** over feature velocity.

---

## Quick Start for Contributors

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/Tetrahedron-Memory-Hive-System.git
cd Tetrahedron-Memory-Hive-System

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install with dev dependencies
pip install -e ".[dev,test]"

# 4. Verify everything works
pytest tests/ -v
```

**Python 3.9+ required** (GUDHI dependency constraint).

---

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/sunormesky-max/Tetrahedron-Memory-Hive-System/labels/good%20first%20issue) or [`help wanted`](https://github.com/sunormesky-max/Tetrahedron-Memory-Hive-System/labels/help%20wanted).

### Suggested beginner contributions:

| Area | Ideas |
|------|-------|
| **Tests** | Add edge-case tests for `TetraMesh`, `ZigzagTracker`, or `DreamCycle` |
| **Docs** | Improve docstrings, add usage examples to README |
| **Visualization** | Build Jupyter notebooks showing tetrahedral memory in 3D (plotly) |
| **Performance** | Profile and optimize hot paths in `core.py` or `geometry.py` |
| **Integrations** | Add LangChain/LlamaIndex memory adapter, or new LLM tools |
| **Math** | Implement new topological operations (e.g., Morse theory, spectral sequences) |
| **i18n** | Translate documentation to other languages |

---

## Code Style

- **PEP 8** — run `ruff check .` before committing
- **Optional deps** — Ray, FastAPI, Prometheus, boto3 must be guarded with `try/except` at import. Core library must work without any optional dep
- **Thread-safety** — use `threading.RLock()` for all shared mutable state (never `Lock`)
- **No bare excepts** — always `except Exception as e:` with logging
- **No type suppression** — no `# type: ignore`, `@ts-ignore`, or `as any`
- **No comments in code** — let function/variable names speak

---

## Testing

- All new features must include tests in `tests/`
- Run the full suite: `pytest tests/ -v`
- **Current baseline: 430 passed, 0 failed**
- Optional dependency tests: use `pytest.importorskip("ray")` or `@pytest.mark.skipif`

### Test naming convention

```
test_<module>_<scenario>_<expected_result>.py
```

Example: `test_dream_cycle_mapping_cone_stability.py`

---

## Pull Request Process

1. **Fork** the repository
2. **Branch**: `git checkout -b feature/your-feature` or `fix/your-fix`
3. **Code**: Make your changes with tests
4. **Verify**: `pytest tests/ -v` and `ruff check .` must pass
5. **PR**: Open against `main` with a clear description

### PR checklist:
- [ ] Tests added/updated
- [ ] `pytest tests/ -v` passes
- [ ] `ruff check .` clean
- [ ] New optional deps guarded with `try/except`
- [ ] No secrets/credentials in code

Keep PRs **small and focused**. Large refactors should be discussed in an issue first.

---

## Reporting Issues

### Bug reports:
- Python version, OS, package versions (`pip list | grep -i gudhi`)
- Minimal reproducible example
- Expected vs actual behavior

### Mathematical/topological questions:
- Describe the expected topological behavior with specific input
- Include the relevant homology dimension (H₀, H₁, H₂)
- If possible, provide a small-scale example showing the discrepancy

---

## Architecture Overview

```
tetrahedron_memory/
├── core.py              # GeoMemoryBody — main entry point
├── geometry.py          # Spatial primitives + SemanticEmbedder
├── tetra_mesh.py        # Tetrahedral mesh operations
├── tetra_dream.py       # Dream cycle (PH-weighted synthesis)
├── zigzag_persistence.py # Zigzag PH tracking + mapping cones
├── resolution_pyramid.py # Multi-scale resolution + adaptive feedback
├── emergence.py         # Self-emergence pressure + threshold
├── eternity_audit.py    # SHA-256 no-deletion verification
├── persistence.py       # Parquet/S3/JSON persistence
├── consistency.py       # Consistency management + staleness
├── multimodal.py        # Multi-parametric filtration
├── router.py            # REST API (FastAPI)
├── llm_tool.py          # LLM function-calling tools
├── distributed.py       # Ray actor distribution
└── monitoring.py        # Prometheus metrics
```

---

## Community

- **Discussions**: [GitHub Discussions](https://github.com/sunormesky-max/Tetrahedron-Memory-Hive-System/discussions) — ask questions, share ideas, show off your projects
- **Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions

We are committed to providing a welcoming and inclusive experience. Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

By contributing, you agree that your contributions will be licensed under **CC BY-NC 4.0**, the same license as the project.
