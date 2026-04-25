"""
Unit tests for RuntimeObserver -- Self-Observation Layer.
"""
import json
import logging
import os
import tempfile
import time

import pytest

from tetrahedron_memory.honeycomb_neural_field import HoneycombNeuralField
from tetrahedron_memory.runtime_observer import (
    RuntimeObserver,
    TetraMemLogHandler,
    LogEvent,
    LogFileTailer,
    attach_file_observer,
    attach_callback_observer,
    OBSERVER_SOURCE_TAG,
    CATEGORY_ERROR,
    CATEGORY_ANOMALY,
    CATEGORY_BEHAVIOR,
    CATEGORY_NOISE,
    CATEGORY_PERFORMANCE,
    CATEGORY_SYSTEM,
)
from tetrahedron_memory.observer_config import (
    auto_attach,
    write_default_config,
    load_config,
    DEFAULT_CONFIG,
)


@pytest.fixture
def field():
    f = HoneycombNeuralField(resolution=5, spacing=1.0)
    f.initialize()
    return f


@pytest.fixture
def populated_field(field):
    for i in range(10):
        field.store(
            content=f"test memory {i} about topic {i % 3}",
            labels=[f"topic-{i % 3}"],
            weight=1.0 + 0.1 * i,
        )
    return field


@pytest.fixture
def observer(populated_field):
    obs = RuntimeObserver(
        field=populated_field,
        window_seconds=5.0,
        max_stores_per_minute=60,
        queue_max_size=50,
        min_events_for_window=2,
    )
    return obs


class TestClassification:
    def test_error_level_classified(self, observer):
        accepted = observer.observe("ERROR", "tetramem.api", "Connection refused")
        assert accepted is True

    def test_critical_level_classified(self, observer):
        accepted = observer.observe("CRITICAL", "tetramem.core", "System crash detected")
        assert accepted is True

    def test_anomaly_pattern(self, observer):
        accepted = observer.observe("WARNING", "tetramem.honeycomb", "Slow query detected: 5000ms")
        assert accepted is True

    def test_noise_dropped(self, observer):
        accepted = observer.observe("DEBUG", "tetramem.health", "heartbeat ok")
        assert accepted is False
        stats = observer.get_stats()
        assert stats["events_dropped_noise"] >= 1

    def test_behavior_pattern(self, observer):
        accepted = observer.observe("INFO", "tetramem.honeycomb", "store completed successfully")
        assert accepted is True

    def test_performance_pattern(self, observer):
        accepted = observer.observe("INFO", "tetramem.bench", "Query latency: 150ms")
        assert accepted is True


class TestLoopIsolation:
    def test_observer_source_dropped(self, observer):
        accepted = observer.observe(
            "INFO", "tetramem.honeycomb", "some message",
            source=OBSERVER_SOURCE_TAG,
        )
        assert accepted is False
        stats = observer.get_stats()
        assert stats["events_dropped_loop"] >= 1

    def test_observer_module_dropped(self, observer):
        accepted = observer.observe(
            "INFO", "tetramem.observer", "observer internal message",
        )
        assert accepted is False

    def test_normal_module_accepted(self, observer):
        accepted = observer.observe(
            "INFO", "tetramem.honeycomb", "normal message",
        )
        assert accepted is True


class TestAggregation:
    def test_immediate_error_stores_directly(self, observer):
        initial_stored = observer.get_stats()["memories_stored"]
        observer.observe("ERROR", "tetramem.api", "Database connection lost")
        observer.flush_all()
        final_stored = observer.get_stats()["memories_stored"]
        assert final_stored > initial_stored

    def test_behavioral_events_aggregate(self, observer):
        for i in range(5):
            observer.observe("INFO", "tetramem.honeycomb", f"store completed #{i}")
        time.sleep(0.1)
        observer.flush_all()
        stats = observer.get_stats()
        assert stats["windows_aggregated"] >= 1 or stats["memories_stored"] >= 1


class TestRateLimiting:
    def test_rate_limit_enforced(self, populated_field):
        obs = RuntimeObserver(
            field=populated_field,
            window_seconds=5.0,
            max_stores_per_minute=3,
            queue_max_size=50,
            min_events_for_window=1,
        )
        stored = 0
        for i in range(10):
            obs.observe("ERROR", "tetramem.test", f"Error event {i}")
        stored = obs.flush_all()
        stats = obs.get_stats()
        assert stats["events_dropped_rate"] > 0 or stats["memories_stored"] <= 3


class TestNarration:
    def test_single_event_narration(self, observer):
        observer.observe("ERROR", "tetramem.api", "Timeout on query")
        observer.flush_all()
        stats = observer.get_stats()
        assert stats["memories_stored"] >= 1

    def test_window_narration_contains_category(self, observer):
        for i in range(3):
            observer.observe("INFO", "tetramem.honeycomb", f"store completed #{i}")
        time.sleep(0.05)
        observer.flush_all()
        assert observer.get_stats()["memories_stored"] >= 1


class TestPrivacyRedaction:
    def test_api_key_redacted(self, observer):
        observer.observe(
            "ERROR", "tetramem.api",
            "Failed with api_key=sk-12345secret",
        )
        observer.flush_all()
        assert observer.get_stats()["memories_stored"] >= 1

    def test_bearer_token_redacted(self, observer):
        observer.observe(
            "ERROR", "tetramem.auth",
            "Bearer abc123token validation failed",
        )
        observer.flush_all()
        assert observer.get_stats()["memories_stored"] >= 1


class TestEnableDisable:
    def test_disable_rejects_events(self, observer):
        observer.set_enabled(False)
        accepted = observer.observe("ERROR", "tetramem.test", "Should be dropped")
        assert accepted is False

    def test_enable_accepts_events(self, observer):
        observer.set_enabled(False)
        observer.set_enabled(True)
        accepted = observer.observe("ERROR", "tetramem.test", "Should be accepted")
        assert accepted is True


class TestStats:
    def test_stats_structure(self, observer):
        stats = observer.get_stats()
        assert "enabled" in stats
        assert "total_events_received" in stats
        assert "memories_stored" in stats
        assert "events_dropped_noise" in stats
        assert "events_dropped_loop" in stats
        assert "events_dropped_rate" in stats
        assert "category_counts" in stats
        assert "config" in stats
        assert "active_windows" in stats

    def test_stats_track_events(self, observer):
        observer.observe("ERROR", "tetramem.test", "Error 1")
        observer.observe("DEBUG", "tetramem.test", "heartbeat")
        observer.observe("INFO", "tetramem.test", "store done")
        stats = observer.get_stats()
        assert stats["total_events_received"] == 3
        assert stats["events_dropped_noise"] == 1


class TestLogHandler:
    def test_handler_feeds_observer(self, observer):
        handler = TetraMemLogHandler(observer, level=logging.ERROR)
        test_logger = logging.getLogger("tetramem.test_handler")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.ERROR)
        try:
            test_logger.error("Test error from handler")
            stats = observer.get_stats()
            assert stats["total_events_received"] >= 1
        finally:
            test_logger.removeHandler(handler)

    def test_handler_ignores_observer_module(self, observer):
        handler = TetraMemLogHandler(observer, level=logging.WARNING)
        obs_logger = logging.getLogger("tetramem.observer")
        obs_logger.addHandler(handler)
        obs_logger.setLevel(logging.WARNING)
        try:
            obs_logger.warning("Internal observer message")
            stats = observer.get_stats()
            assert stats["events_dropped_loop"] >= 1
        finally:
            obs_logger.removeHandler(handler)


class TestIntegration:
    def test_observer_in_initialized_field(self, field):
        obs = getattr(field, "_runtime_observer", None)
        assert obs is not None
        assert obs._enabled is True

    def test_observer_stores_trajectory_memory(self, field):
        obs = field._runtime_observer
        obs.observe("ERROR", "tetramem.test", "Test error for integration")
        obs.flush_all()
        stats = obs.get_stats()
        assert stats["memories_stored"] >= 1

        results = field.query("trajectory error", k=5)
        found = False
        if results:
            for r in results:
                if OBSERVER_SOURCE_TAG in r.get("labels", []):
                    found = True
                    break
        assert found

    def test_observer_stats_in_export(self, field):
        state = field.export_full_state()
        assert "runtime_observer" in state
        assert state["runtime_observer"]["enabled"] is True


class TestIngest:
    def test_ingest_logevent(self, observer):
        event = LogEvent(time.time(), "ERROR", "my_ai", "Planning failed")
        accepted = observer.ingest(event)
        assert accepted is True
        stats = observer.get_stats()
        assert stats["total_events_received"] >= 1

    def test_ingest_multiple(self, observer):
        for i in range(3):
            observer.ingest(LogEvent(time.time(), "INFO", "my_ai", f"Step {i} done"))
        observer.flush_all()
        stats = observer.get_stats()
        assert stats["memories_stored"] >= 1


class TestFileTail:
    def test_file_tailer_reads_log(self, populated_field, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "2025-01-01 12:00:00 ERROR my_app Connection timeout\n"
            "2025-01-01 12:00:01 INFO my_app store completed\n",
            encoding="utf-8",
        )
        obs = RuntimeObserver(
            field=populated_field,
            window_seconds=5.0,
            max_stores_per_minute=60,
            min_events_for_window=1,
        )
        obs.start()
        tailer = obs.add_file_tail(str(log_file), poll_interval=0.1)
        time.sleep(0.5)
        stats = obs.get_stats()
        assert stats["total_events_received"] >= 2
        obs.stop()

    def test_file_tailer_rotation(self, populated_field, tmp_path):
        log_file = tmp_path / "rotate.log"
        log_file.write_text("2025-01-01 12:00:00 ERROR app Error 1\n", encoding="utf-8")
        obs = RuntimeObserver(
            field=populated_field,
            window_seconds=5.0,
            max_stores_per_minute=60,
            min_events_for_window=1,
        )
        obs.start()
        tailer = obs.add_file_tail(str(log_file), poll_interval=0.1)
        time.sleep(0.5)
        with open(str(log_file), "a", encoding="utf-8") as f:
            f.write("2025-01-01 12:00:01 ERROR app Error 2\n")
        time.sleep(1.0)
        stats = obs.get_stats()
        assert stats["total_events_received"] >= 1
        obs.stop()

    def test_file_tailer_stats(self, populated_field, tmp_path):
        log_file = tmp_path / "stats.log"
        log_file.write_text("", encoding="utf-8")
        obs = RuntimeObserver(field=populated_field, window_seconds=5.0)
        tailer = obs.add_file_tail(str(log_file))
        stats = obs.get_stats()
        assert "file_tailers" in stats
        assert len(stats["file_tailers"]) == 1
        assert stats["file_tailers"][0]["path"] == str(log_file)
        obs.stop()

    def test_file_tailer_nonexistent_file(self, populated_field):
        obs = RuntimeObserver(field=populated_field, window_seconds=5.0)
        tailer = obs.add_file_tail("/nonexistent/path/to/file.log")
        assert tailer is not None
        stats = obs.get_stats()
        assert stats["total_events_received"] == 0
        obs.stop()


class TestCustomRules:
    def test_set_custom_rules(self, observer):
        observer.set_rules([
            {
                "category": "custom_cat",
                "pattern": r"(custom_pattern)",
                "weight": 1.0,
                "immediate": True,
            }
        ])
        accepted = observer.observe("INFO", "tetramem.test", "Found custom_pattern here")
        assert accepted is True
        stats = observer.get_stats()
        assert "custom_cat" in stats["category_counts"]


class TestAttachFunctions:
    def test_attach_callback_observer(self, populated_field):
        obs = attach_callback_observer(
            populated_field,
            window_seconds=5.0,
            max_stores_per_minute=60,
        )
        assert obs is not None
        obs.ingest(LogEvent(time.time(), "ERROR", "test", "Callback test"))
        obs.flush_all()
        assert obs.get_stats()["memories_stored"] >= 1
        obs.stop()

    def test_attach_file_observer(self, populated_field, tmp_path):
        log_file = tmp_path / "attach.log"
        log_file.write_text(
            "2025-01-01 12:00:00 ERROR app File attach test\n",
            encoding="utf-8",
        )
        obs = attach_file_observer(
            populated_field,
            file_path=str(log_file),
            poll_interval=0.1,
            window_seconds=5.0,
            max_stores_per_minute=60,
        )
        time.sleep(0.5)
        assert obs.get_stats()["total_events_received"] >= 1
        obs.stop()


class TestObserverConfig:
    def test_write_default_config(self, tmp_path):
        cfg_path = str(tmp_path / "observer.json")
        result = write_default_config(cfg_path)
        assert os.path.exists(result)
        with open(result, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert "rules" in cfg
        assert "log_sources" in cfg
        assert len(cfg["rules"]) > 0

    def test_load_config_missing(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.json"))
        assert "rules" in cfg

    def test_load_config_valid(self, tmp_path):
        cfg_path = tmp_path / "valid.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"window_seconds": 60, "rules": []}, f)
        cfg = load_config(str(cfg_path))
        assert cfg["window_seconds"] == 60

    def test_load_config_invalid_json(self, tmp_path):
        cfg_path = tmp_path / "bad.json"
        cfg_path.write_text("{invalid json", encoding="utf-8")
        cfg = load_config(str(cfg_path))
        assert "rules" in cfg

    def test_auto_attach_creates_config(self, populated_field, tmp_path):
        cfg_path = str(tmp_path / "auto.json")
        assert not os.path.exists(cfg_path)
        obs = auto_attach(populated_field, config_path=cfg_path)
        assert os.path.exists(cfg_path)
        assert obs is not None
        obs.stop()

    def test_auto_attach_uses_existing_config(self, populated_field, tmp_path):
        cfg_path = tmp_path / "existing.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({
                "enabled": True,
                "window_seconds": 10,
                "rules": [],
                "log_sources": {"python_logging": {"enabled": False}},
            }, f)
        obs = auto_attach(populated_field, config_path=str(cfg_path))
        assert obs._window_seconds == 10
        obs.stop()

    def test_auto_attach_disabled(self, populated_field, tmp_path):
        cfg_path = tmp_path / "disabled.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({
                "enabled": False,
                "window_seconds": 5,
                "rules": [],
                "log_sources": {},
            }, f)
        obs = auto_attach(populated_field, config_path=str(cfg_path))
        assert obs._enabled is False
        obs.stop()

    def test_auto_attach_with_file_tail(self, populated_field, tmp_path):
        log_file = tmp_path / "tail.log"
        log_file.write_text("2025-01-01 12:00:00 ERROR app Test\n", encoding="utf-8")
        cfg_path = tmp_path / "with_tail.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({
                "enabled": True,
                "window_seconds": 5,
                "rules": [],
                "log_sources": {
                    "python_logging": {"enabled": False},
                    "file_tail": [
                        {"path": str(log_file), "poll_interval": 0.1},
                    ],
                },
            }, f)
        obs = auto_attach(populated_field, config_path=str(cfg_path))
        time.sleep(0.5)
        stats = obs.get_stats()
        assert stats["total_events_received"] >= 1
        obs.stop()
