import threading
import logging
from typing import Any, Dict, Optional

_log = logging.getLogger("tetramem.monitoring")
_lock = threading.Lock()
_counters: Dict[str, int] = {}
_gauges: Dict[str, float] = {}
_errors: Dict[str, int] = {}

def increment_counter(name: str, value: int = 1):
    with _lock:
        _counters[name] = _counters.get(name, 0) + value

def set_gauge(name: str, value: float):
    with _lock:
        _gauges[name] = value

def observe_histogram(name: str, value: float):
    pass

def record_error(name: str, error: str = ""):
    with _lock:
        _errors[name] = _errors.get(name, 0) + 1
    _log.debug("record_error: %s — %s", name, error)

def health_check() -> Dict[str, Any]:
    with _lock:
        return {"status": "ok", "counters": dict(_counters), "gauges": dict(_gauges), "errors": dict(_errors)}

def get_metrics_registry():
    with _lock:
        return {"counters": dict(_counters), "gauges": dict(_gauges), "errors": dict(_errors)}

STORE_COUNTER = "stores"
QUERY_COUNTER = "queries"
ASSOCIATE_COUNTER = "associates"
SELF_ORGANIZE_COUNTER = "self_organize"
NODE_COUNT_GAUGE = "node_count"
WEIGHT_HISTOGRAM = "weight"
ERROR_COUNTER = "errors"
STORE_LATENCY = "store_latency"
QUERY_LATENCY = "query_latency"
ENTROPY_GAUGE = "entropy"
INTEGRATION_COUNTER = "integrations"
DREAM_COUNTER = "dreams"

def get_grafana_dashboard_json():
    return {"dashboard": {}}

def get_ray_cluster_status():
    return {"status": "not_available"}
