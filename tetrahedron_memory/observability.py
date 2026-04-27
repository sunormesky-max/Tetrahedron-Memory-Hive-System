import logging
import time
import threading
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

_log = logging.getLogger("tetramem.observability")


class SimpleMetrics:
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._timers: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, value: int = 1):
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    @contextmanager
    def timer(self, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - t0
            with self._lock:
                if name not in self._timers:
                    self._timers[name] = []
                self._timers[name].append(elapsed)
                if len(self._timers[name]) > 1000:
                    self._timers[name] = self._timers[name][-500:]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = {"counters": dict(self._counters), "timers": {}}
            for name, values in self._timers.items():
                if values:
                    sorted_v = sorted(values)
                    stats["timers"][name] = {
                        "count": len(values),
                        "avg_ms": sum(values) / len(values) * 1000,
                        "p50_ms": sorted_v[len(sorted_v) // 2] * 1000,
                        "p95_ms": sorted_v[int(len(sorted_v) * 0.95)] * 1000,
                        "p99_ms": sorted_v[int(len(sorted_v) * 0.99)] * 1000,
                    }
            return stats


def get_logger(name: str = "tetramem") -> "logging.Logger":
    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class HealthChecker:
    def __init__(self):
        self._checks: Dict[str, Any] = {}
        self._last_check: Optional[Dict] = None
        self._check_interval: float = 60.0
        self._last_check_time: float = 0.0

    def register_check(self, name: str, check_fn):
        self._checks[name] = check_fn

    def run_checks(self) -> Dict[str, Any]:
        results = {}
        issues = []
        for name, fn in self._checks.items():
            try:
                result = fn()
                results[name] = result
                if isinstance(result, dict) and result.get("status") == "unhealthy":
                    issues.append(name)
            except Exception as e:
                _log.warning("Health check '%s' raised exception: %s", name, e, exc_info=True)
                results[name] = {"status": "error", "error": str(e)}
                issues.append(name)
        self._last_check = {"results": results, "issues": issues, "ts": time.time()}
        self._last_check_time = time.time()
        return self._last_check
