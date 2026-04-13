from typing import Any, Dict, Optional

_prometheus_available = False
try:
    from prometheus_client import (
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _prometheus_available = True
except ImportError:
    Counter = None
    Gauge = None
    Histogram = None
    CollectorRegistry = None
    generate_latest = None
    REGISTRY = None

_custom_registry = None
if _prometheus_available:
    _custom_registry = CollectorRegistry()

STORE_COUNTER: Optional[Any] = None
QUERY_COUNTER: Optional[Any] = None
ASSOCIATE_COUNTER: Optional[Any] = None
SELF_ORGANIZE_COUNTER: Optional[Any] = None
NODE_COUNT_GAUGE: Optional[Any] = None
WEIGHT_HISTOGRAM: Optional[Any] = None
ERROR_COUNTER: Optional[Any] = None
STORE_LATENCY: Optional[Any] = None
QUERY_LATENCY: Optional[Any] = None
ENTROPY_GAUGE: Optional[Any] = None
INTEGRATION_COUNTER: Optional[Any] = None
DREAM_COUNTER: Optional[Any] = None

if _prometheus_available and _custom_registry is not None:
    STORE_COUNTER = Counter(
        "tetramem_store_total", "Total store operations", registry=_custom_registry
    )
    QUERY_COUNTER = Counter(
        "tetramem_query_total", "Total query operations", registry=_custom_registry
    )
    ASSOCIATE_COUNTER = Counter(
        "tetramem_associate_total", "Total associate operations", registry=_custom_registry
    )
    SELF_ORGANIZE_COUNTER = Counter(
        "tetramem_self_organize_total", "Total self_organize calls", registry=_custom_registry
    )
    NODE_COUNT_GAUGE = Gauge(
        "tetramem_node_count", "Current number of memory nodes", registry=_custom_registry
    )
    WEIGHT_HISTOGRAM = Histogram(
        "tetramem_weight_distribution", "Weight distribution of nodes", registry=_custom_registry
    )
    ERROR_COUNTER = Counter(
        "tetramem_errors_total", "Total errors", ["operation"], registry=_custom_registry
    )
    STORE_LATENCY = Histogram(
        "tetramem_store_latency_seconds",
        "Store operation latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        registry=_custom_registry,
    )
    QUERY_LATENCY = Histogram(
        "tetramem_query_latency_seconds",
        "Query operation latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        registry=_custom_registry,
    )
    ENTROPY_GAUGE = Gauge(
        "tetramem_persistent_entropy",
        "Current persistent entropy value",
        registry=_custom_registry,
    )
    INTEGRATION_COUNTER = Counter(
        "tetramem_integration_total",
        "Total integration catalyst operations",
        registry=_custom_registry,
    )
    DREAM_COUNTER = Counter(
        "tetramem_dream_cycle_total",
        "Total dream cycles executed",
        registry=_custom_registry,
    )


def increment_counter(metric: Optional[Any], value: float = 1) -> None:
    if metric is not None:
        try:
            metric.inc(value)
        except Exception:
            pass


def set_gauge(metric: Optional[Any], value: float) -> None:
    if metric is not None:
        try:
            metric.set(value)
        except Exception:
            pass


def observe_histogram(metric: Optional[Any], value: float) -> None:
    if metric is not None:
        try:
            metric.observe(value)
        except Exception:
            pass


def record_error(operation: str) -> None:
    if ERROR_COUNTER is not None:
        try:
            ERROR_COUNTER.labels(operation=operation).inc()
        except Exception:
            pass


def generate_metrics() -> str:
    if not _prometheus_available or generate_latest is None or _custom_registry is None:
        return ""
    try:
        return generate_latest(_custom_registry).decode("utf-8")
    except Exception:
        return ""


def get_metrics_registry() -> Optional[Any]:
    return _custom_registry


def get_ray_cluster_status() -> Dict[str, Any]:
    try:
        import ray

        if not ray.is_initialized():
            return {"status": "not_initialized", "ray_available": True}

        return {
            "status": "running",
            "ray_available": True,
            "nodes": len(ray.nodes()),
            "resources": ray.cluster_resources(),
            "available_resources": ray.available_resources(),
        }
    except ImportError:
        return {"status": "ray_not_installed", "ray_available": False}
    except Exception as e:
        return {"status": "error", "error": str(e), "ray_available": True}


GRAFANA_DASHBOARD_TEMPLATE = {
    "annotations": {
        "list": [
            {
                "builtIn": 1,
                "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                "enable": True,
                "hide": True,
                "name": "Annotations & Alerts",
                "type": "dashboard",
            }
        ]
    },
    "description": "Production monitoring dashboard for TetraMem-XL geometric memory system",
    "editable": True,
    "fiscalYearStartMonth": 0,
    "graphTooltip": 1,
    "id": None,
    "links": [],
    "panels": [
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {"color": {"mode": "palette-classic"}, "custom": {"fillOpacity": 20}},
                "overrides": [],
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "id": 1,
            "title": "Store / Query Throughput (ops/sec)",
            "type": "timeseries",
            "targets": [
                {"expr": "rate(tetramem_store_total[5m])", "legendFormat": "store/sec"},
                {"expr": "rate(tetramem_query_total[5m])", "legendFormat": "query/sec"},
                {"expr": "rate(tetramem_associate_total[5m])", "legendFormat": "associate/sec"},
            ],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [{"color": "green", "value": None}, {"color": "red", "value": 80}],
                    },
                },
                "overrides": [],
            },
            "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0},
            "id": 2,
            "title": "Active Memory Nodes",
            "type": "stat",
            "targets": [{"expr": "tetramem_node_count", "legendFormat": "nodes"}],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 1},
                            {"color": "red", "value": 10},
                        ],
                    },
                },
                "overrides": [],
            },
            "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0},
            "id": 3,
            "title": "Error Rate (errors/sec)",
            "type": "stat",
            "targets": [
                {"expr": "sum(rate(tetramem_errors_total[5m]))", "legendFormat": "errors/sec"}
            ],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}}, "overrides": []},
            "gridPos": {"h": 4, "w": 12, "x": 12, "y": 4},
            "id": 4,
            "title": "Self-Organize & Associate Rate",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "rate(tetramem_self_organize_total[5m])",
                    "legendFormat": "self-organize/sec",
                },
                {"expr": "rate(tetramem_associate_total[5m])", "legendFormat": "associate/sec"},
            ],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {"color": {"mode": "palette-classic"}, "custom": {"fillOpacity": 30}},
                "overrides": [],
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            "id": 5,
            "title": "Weight Distribution Histogram",
            "type": "timeseries",
            "targets": [
                {"expr": "tetramem_weight_distribution_bucket", "legendFormat": "{{le}}"},
            ],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {"color": {"fixedColor": "orange", "mode": "fixed"}},
                "overrides": [],
            },
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
            "id": 6,
            "title": "Errors by Operation",
            "type": "timeseries",
            "targets": [
                {"expr": "rate(tetramem_errors_total[5m])", "legendFormat": "{{operation}}"},
            ],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 50000},
                            {"color": "red", "value": 100000},
                        ]
                    },
                    "unit": "none",
                },
                "overrides": [],
            },
            "gridPos": {"h": 4, "w": 8, "x": 0, "y": 16},
            "id": 7,
            "title": "Memory Node Count (Gauge)",
            "type": "gauge",
            "targets": [{"expr": "tetramem_node_count", "legendFormat": "nodes"}],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}}, "overrides": []},
            "gridPos": {"h": 4, "w": 8, "x": 8, "y": 16},
            "id": 8,
            "title": "Cumulative Store Operations",
            "type": "timeseries",
            "targets": [{"expr": "tetramem_store_total", "legendFormat": "total stores"}],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}}, "overrides": []},
            "gridPos": {"h": 4, "w": 8, "x": 16, "y": 16},
            "id": 9,
            "title": "Cumulative Query Operations",
            "type": "timeseries",
            "targets": [{"expr": "tetramem_query_total", "legendFormat": "total queries"}],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {"color": {"mode": "palette-classic"}, "custom": {"fillOpacity": 20}},
                "overrides": [],
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 20},
            "id": 10,
            "title": "Store vs Query Latency Comparison",
            "type": "timeseries",
            "targets": [
                {"expr": "rate(tetramem_store_total[1m])", "legendFormat": "store rate"},
                {"expr": "rate(tetramem_query_total[1m])", "legendFormat": "query rate"},
                {"expr": "rate(tetramem_associate_total[1m])", "legendFormat": "associate rate"},
                {
                    "expr": "rate(tetramem_self_organize_total[1m])",
                    "legendFormat": "self-organize rate",
                },
            ],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"fixedColor": "purple", "mode": "fixed"},
                    "custom": {"stacking": {"mode": "normal", "group": "A"}},
                },
                "overrides": [],
            },
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 20},
            "id": 11,
            "title": "Error Breakdown (Stacked)",
            "type": "timeseries",
            "targets": [
                {"expr": "tetramem_errors_total", "legendFormat": "{{operation}}"},
            ],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 0.01},
                        ],
                    },
                    "unit": "ops/sec",
                },
                "overrides": [],
            },
            "gridPos": {"h": 4, "w": 24, "x": 0, "y": 28},
            "id": 12,
            "title": "Aggregate Error Rate",
            "type": "stat",
            "targets": [
                {"expr": "sum(rate(tetramem_errors_total[5m]))", "legendFormat": "error rate"}
            ],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "unit": "s",
                },
                "overrides": [],
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 32},
            "id": 13,
            "title": "Store/Query Latency (p50, p95, p99)",
            "type": "timeseries",
            "targets": [
                {"expr": "histogram_quantile(0.5, rate(tetramem_store_latency_seconds_bucket[5m]))", "legendFormat": "store p50"},
                {"expr": "histogram_quantile(0.95, rate(tetramem_store_latency_seconds_bucket[5m]))", "legendFormat": "store p95"},
                {"expr": "histogram_quantile(0.99, rate(tetramem_store_latency_seconds_bucket[5m]))", "legendFormat": "store p99"},
                {"expr": "histogram_quantile(0.5, rate(tetramem_query_latency_seconds_bucket[5m]))", "legendFormat": "query p50"},
                {"expr": "histogram_quantile(0.95, rate(tetramem_query_latency_seconds_bucket[5m]))", "legendFormat": "query p95"},
            ],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 2.0},
                            {"color": "red", "value": 4.0},
                        ],
                    },
                },
                "overrides": [],
            },
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 32},
            "id": 14,
            "title": "Persistent Entropy",
            "type": "stat",
            "targets": [
                {"expr": "tetramem_persistent_entropy", "legendFormat": "entropy"},
            ],
        },
        {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "unit": "ops/sec",
                },
                "overrides": [],
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 40},
            "id": 15,
            "title": "Integration & Dream Cycles",
            "type": "timeseries",
            "targets": [
                {"expr": "rate(tetramem_integration_total[5m])", "legendFormat": "integrations/sec"},
                {"expr": "rate(tetramem_dream_cycle_total[5m])", "legendFormat": "dreams/sec"},
            ],
        },
    ],
    "refresh": "10s",
    "schemaVersion": 39,
    "tags": ["tetramem", "memory", "topology", "production"],
    "templating": {
        "list": [
            {
                "current": {"selected": False, "text": "Prometheus", "value": "Prometheus"},
                "hide": 0,
                "includeAll": False,
                "name": "DS_PROMETHEUS",
                "options": [],
                "query": "prometheus",
                "type": "datasource",
            }
        ]
    },
    "time": {"from": "now-1h", "to": "now"},
    "timepicker": {},
    "timezone": "utc",
    "title": "TetraMem-XL Production Dashboard",
    "uid": "tetramem-xl-prod",
    "version": 1,
    "overwrite": True,
}


ALERT_RULES = [
    {
        "alert": "TetraMemEntropySpike",
        "expr": "tetramem_persistent_entropy > 4.0",
        "for": "2m",
        "labels": {"severity": "warning"},
        "annotations": {"summary": "Persistent entropy spike > 4.0", "description": "Entropy has exceeded safe threshold for 2 minutes."},
    },
    {
        "alert": "TetraMemHighErrorRate",
        "expr": "sum(rate(tetramem_errors_total[5m])) > 0.02",
        "for": "1m",
        "labels": {"severity": "critical"},
        "annotations": {"summary": "Error rate > 2%", "description": "System error rate exceeds 2% threshold."},
    },
    {
        "alert": "TetraMemStoreLatencyHigh",
        "expr": "histogram_quantile(0.99, rate(tetramem_store_latency_seconds_bucket[5m])) > 0.1",
        "for": "5m",
        "labels": {"severity": "warning"},
        "annotations": {"summary": "Store p99 latency > 100ms", "description": "Store latency degradation detected."},
    },
    {
        "alert": "TetraMemQueryLatencyHigh",
        "expr": "histogram_quantile(0.99, rate(tetramem_query_latency_seconds_bucket[5m])) > 0.05",
        "for": "5m",
        "labels": {"severity": "warning"},
        "annotations": {"summary": "Query p99 latency > 50ms", "description": "Query latency degradation detected."},
    },
]

GRAFANA_ALERT_GROUPS = {
    "groups": [
        {
            "name": "tetramem_alerts",
            "rules": ALERT_RULES,
        }
    ]
}


def get_alert_rules() -> Dict[str, Any]:
    return GRAFANA_ALERT_GROUPS


def get_grafana_dashboard_json() -> str:
    import json

    return json.dumps(GRAFANA_DASHBOARD_TEMPLATE, indent=2)


def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "prometheus": _prometheus_available,
        "ray": get_ray_cluster_status(),
    }
