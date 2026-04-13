# Contributing to TetraMem

Thanks for your interest in contributing. This project is a research-driven geometric memory system, so we value correctness and mathematical rigor over feature velocity.

## Development Setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/sunormesky-max/sunorm-space-memory.git
cd sunorm-space-memory
pip install -e ".[dev,test]"

# Verify tests pass
pytest tests/ -v
```

Python 3.9+ is required (GUDHI does not support 3.8).

## Code Style

- Follow PEP 8. Use `ruff check .` before committing.
- All optional dependencies (Ray, FastAPI, Prometheus, boto3) must be guarded with `try/except` at import time. The core library must work without any optional dep installed.
- Thread-safety: use `threading.RLock()` for all shared mutable state.
- No `as any`, `@ts-ignore`, or type error suppression.
- No empty catch blocks.

## Testing

- All new features must include tests in `tests/`.
- Run the full suite: `pytest tests/ -v`
- Current baseline: 121 passed, 7 skipped (starlette/httpx compat), 0 failed.
- If adding optional dependency tests, use `pytest.importorskip("ray")` or `@pytest.mark.skipif` so tests pass without the dep.

## Pull Request Process

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Make your changes with tests.
4. Ensure `pytest` and `ruff check .` pass.
5. Open a PR against `master` with a clear description of what changed and why.

Keep PRs small and focused. Large refactors should be discussed in an issue first.

## Reporting Issues

- Include Python version, OS, and package versions (`pip list | grep -i gudhi`).
- Provide a minimal reproducible example.
- For mathematical/topological questions, describe the expected vs actual behavior with specific input.

## License

By contributing, you agree that your contributions will be licensed under CC BY-NC 4.0, the same license as the project.
