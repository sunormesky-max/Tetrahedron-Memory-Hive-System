import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()
_counters: Dict[str, int] = {}

def increment_counter(name: str, value: int = 1):
    with _lock:
        _counters[name] = _counters.get(name, 0) + value

def set_gauge(name: str, value: float):
    pass

def observe_histogram(name: str, value: float):
    pass

def record_error(name: str, error: str = ""):
    pass

def health_check() -> Dict[str, Any]:
    return {"status": "ok", "counters": dict(_counters)}

def get_metrics_registry():
    return _counters

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
