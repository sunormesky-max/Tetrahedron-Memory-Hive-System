"""
ObserverConfig -- Configuration management for RuntimeObserver.

Supports three config sources (priority order):
  1. Explicit parameters to auto_attach()
  2. JSON config file
  3. Environment variables
  4. Built-in defaults

Usage:
    from tetrahedron_memory.observer_config import auto_attach, write_default_config

    # Zero-touch: first call writes template, subsequent calls use edited config
    observer = auto_attach(field, config_path="./observer_config.json")

    # Explicit: write a default config for the user to customize
    write_default_config("./observer_config.json")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tetramem.observer_config")

DEFAULT_CONFIG = {
    "enabled": True,
    "window_seconds": 300,
    "max_stores_per_minute": 30,
    "queue_max_size": 200,
    "min_events_for_window": 2,
    "enable_behavior": True,
    "enable_performance": True,
    "enable_system": True,
    "log_sources": {
        "python_logging": {
            "enabled": True,
            "level": "WARNING",
            "logger_name": "tetramem",
        },
        "file_tail": [],
    },
    "rules": [
        {
            "category": "noise",
            "level_filter": ["DEBUG"],
            "pattern": "(heartbeat|health.?check|GET /health|ping|pong)",
            "weight": 0.0,
            "immediate": False,
        },
        {
            "category": "error",
            "level_filter": ["ERROR", "CRITICAL"],
            "pattern": None,
            "weight": 2.0,
            "immediate": True,
        },
        {
            "category": "anomaly",
            "pattern": "(timeout|crash|slow|memory.?low|oom|refused|overflow|deadlock|hung)",
            "weight": 1.5,
            "immediate": True,
        },
        {
            "category": "behavior",
            "pattern": "(store|query|dream|organize|cascade|pulse|crystallize|hebbian|bridge|flow)",
            "weight": 0.3,
            "immediate": False,
        },
        {
            "category": "behavior",
            "pattern": "(store|query|dream|organize)",
            "weight": 0.4,
            "immediate": False,
        },
        {
            "category": "agent_reasoning",
            "pattern": "(reasoning|inference|plan|decide|think|conclude)",
            "weight": 0.5,
            "immediate": False,
        },
        {
            "category": "external_activity",
            "pattern": "(http|request|response|api|webhook|callback)",
            "weight": 0.2,
            "immediate": False,
        },
        {
            "category": "performance",
            "pattern": "(latency|throughput|benchmark|elapsed|duration|took\\s+\\d+ms|ms\\b)",
            "weight": 0.5,
            "immediate": False,
        },
        {
            "category": "system",
            "level_filter": ["WARNING"],
            "pattern": None,
            "weight": 0.8,
            "immediate": False,
        },
    ],
}


def write_default_config(path: str = "./observer_config.json") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
    logger.info("Default observer config written to %s", p)
    return str(p)


def load_config(path: str = "./observer_config.json") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        logger.info("Config not found at %s, using defaults", p)
        return dict(DEFAULT_CONFIG)
    try:
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        logger.info("Observer config loaded from %s", p)
        return cfg
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load config from %s: %s, using defaults", p, e)
        return dict(DEFAULT_CONFIG)


def _env_override(cfg: Dict[str, Any]) -> Dict[str, Any]:
    env_map = {
        "TETRAMEM_OBSERVER_ENABLED": ("enabled", lambda v: v.lower() in ("true", "1", "yes")),
        "TETRAMEM_OBSERVER_WINDOW": ("window_seconds", int),
        "TETRAMEM_OBSERVER_MAX_STORES": ("max_stores_per_minute", int),
        "TETRAMEM_OBSERVER_QUEUE_SIZE": ("queue_max_size", int),
        "TETRAMEM_OBSERVER_LOG_PATH": ("_file_tail_path", str),
    }
    for env_key, (cfg_key, converter) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                cfg[cfg_key] = converter(val)
            except (ValueError, TypeError):
                pass
    return cfg


def auto_attach(
    field: Any,
    config_path: str = "./observer_config.json",
    **overrides: Any,
) -> Any:
    """
    Zero-touch observer mounting. Creates config if missing, loads it,
    applies env overrides, constructs and starts the observer.

    Returns the RuntimeObserver instance.
    """
    from .runtime_observer import RuntimeObserver, TetraMemLogHandler

    p = Path(config_path)
    if not p.exists():
        write_default_config(config_path)
        logger.info(
            "First run: default config written to %s. Edit it to customize, then restart.",
            config_path,
        )

    cfg = load_config(config_path)
    cfg = _env_override(cfg)
    cfg.update(overrides)

    observer = RuntimeObserver(
        field=field,
        window_seconds=float(cfg.get("window_seconds", 300)),
        max_stores_per_minute=int(cfg.get("max_stores_per_minute", 30)),
        queue_max_size=int(cfg.get("queue_max_size", 200)),
        min_events_for_window=int(cfg.get("min_events_for_window", 2)),
        enable_behavior=bool(cfg.get("enable_behavior", True)),
        enable_performance=bool(cfg.get("enable_performance", True)),
        enable_system=bool(cfg.get("enable_system", True)),
    )

    rules_raw = cfg.get("rules")
    if rules_raw:
        compiled_rules = []
        for r in rules_raw:
            compiled_rules.append({
                "category": r.get("category", "behavior"),
                "weight": float(r.get("weight", 0.3)),
                "immediate": bool(r.get("immediate", False)),
                "level_filter": set(r["level_filter"]) if "level_filter" in r else None,
                "pattern": __import__("re").compile(r["pattern"], __import__("re").IGNORECASE) if r.get("pattern") else None,
            })
        observer._compiled_rules = compiled_rules
        observer._custom_rules = compiled_rules

    enabled = cfg.get("enabled", True)
    if not enabled:
        observer.set_enabled(False)
        return observer

    observer.start()

    log_sources = cfg.get("log_sources", {})
    py_logging = log_sources.get("python_logging", {})
    if py_logging.get("enabled", True):
        import logging as _logging
        level_name = py_logging.get("level", "WARNING")
        level = getattr(_logging, level_name.upper(), _logging.WARNING)
        logger_name = py_logging.get("logger_name", "tetramem")
        handler = TetraMemLogHandler(observer, level=level)
        _logging.getLogger(logger_name).addHandler(handler)
        if hasattr(field, "_observer_log_handler"):
            field._observer_log_handler = handler

    file_tail_config = log_sources.get("file_tail", [])
    if isinstance(file_tail_config, dict):
        file_tail_config = [file_tail_config]
    for ft in file_tail_config:
        fpath = ft.get("path", cfg.get("_file_tail_path", ""))
        if fpath:
            observer.add_file_tail(
                file_path=fpath,
                poll_interval=float(ft.get("poll_interval", 1.0)),
                module_override=ft.get("module_override"),
            )

    return observer
