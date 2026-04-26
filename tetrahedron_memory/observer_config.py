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
import re
import time
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
    except json.JSONDecodeError as e:
        logger.warning("Config corrupted at %s: %s — auto-repairing with defaults", p, e)
        repaired = dict(DEFAULT_CONFIG)
        repaired["_repair_note"] = f"Auto-repaired from corrupted JSON at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        try:
            backup_path = str(p) + f".corrupted.{int(time.time())}"
            p.rename(backup_path)
            write_default_config(str(p))
            logger.info("Corrupted config backed up to %s, default written to %s", backup_path, p)
        except OSError:
            pass
        return repaired
    except OSError as e:
        logger.warning("Cannot read config at %s: %s, using defaults", p, e)
        return dict(DEFAULT_CONFIG)

    repaired = _validate_and_repair(cfg)
    if repaired.get("_repaired"):
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(repaired, f, indent=2, ensure_ascii=False)
            logger.info("Config auto-repaired and saved to %s", p)
        except OSError:
            pass
    return repaired


def _validate_and_repair(cfg: Dict[str, Any]) -> Dict[str, Any]:
    repairs = []
    if not isinstance(cfg.get("enabled"), bool):
        cfg["enabled"] = DEFAULT_CONFIG["enabled"]
        repairs.append("enabled")
    if not isinstance(cfg.get("window_seconds"), (int, float)) or cfg.get("window_seconds", 0) <= 0:
        cfg["window_seconds"] = DEFAULT_CONFIG["window_seconds"]
        repairs.append("window_seconds")
    if not isinstance(cfg.get("max_stores_per_minute"), int) or cfg.get("max_stores_per_minute", 0) <= 0:
        cfg["max_stores_per_minute"] = DEFAULT_CONFIG["max_stores_per_minute"]
        repairs.append("max_stores_per_minute")
    if not isinstance(cfg.get("log_sources"), dict):
        cfg["log_sources"] = DEFAULT_CONFIG["log_sources"]
        repairs.append("log_sources")
    elif not isinstance(cfg["log_sources"].get("file_tail"), list):
        ft = cfg["log_sources"].get("file_tail")
        if isinstance(ft, dict):
            cfg["log_sources"]["file_tail"] = [ft]
        else:
            cfg["log_sources"]["file_tail"] = []
        repairs.append("log_sources.file_tail")
    if not isinstance(cfg.get("rules"), list):
        cfg["rules"] = DEFAULT_CONFIG["rules"]
        repairs.append("rules")
    else:
        valid_rules = []
        for i, r in enumerate(cfg["rules"]):
            if not isinstance(r, dict):
                continue
            if "category" not in r:
                r["category"] = "behavior"
                repairs.append(f"rules[{i}].category")
            if "weight" not in r or not isinstance(r.get("weight"), (int, float)):
                r["weight"] = 0.3
                repairs.append(f"rules[{i}].weight")
            if r.get("pattern") is not None:
                try:
                    re.compile(r["pattern"])
                except re.error:
                    r["pattern"] = None
                    repairs.append(f"rules[{i}].pattern_invalid")
            valid_rules.append(r)
        cfg["rules"] = valid_rules
    if repairs:
        cfg["_repaired"] = True
        cfg["_repair_details"] = repairs
    return cfg


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
        from .runtime_observer import CLASSIFICATION_RULES
        compiled_rules.extend(CLASSIFICATION_RULES)
        for r in rules_raw:
            compiled_rules.append({
                "category": r.get("category", "behavior"),
                "weight": float(r.get("weight", 0.3)),
                "immediate": bool(r.get("immediate", False)),
                "level_filter": set(r["level_filter"]) if "level_filter" in r else None,
                "pattern": re.compile(r["pattern"], re.IGNORECASE) if r.get("pattern") else None,
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
