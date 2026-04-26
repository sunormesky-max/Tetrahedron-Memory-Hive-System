"""
RuntimeObserver -- Self-Observation Layer for TetraMem-XL.

Intercepts system runtime events, classifies them by semantic category,
aggregates over sliding windows, generates trajectory narrations, and
injects low-weight "self-observation" memories into the TMHS store.

Input sources:
  - Python logging handler (TetraMemLogHandler) -- in-process capture
  - File tail (LogFileTailer) -- watch external log files
  - Programmatic ingest() -- direct LogEvent injection

Key design constraints:
  - Loop isolation: drops all events originating from the observer itself
  - Rate limiting: hard cap on stores per minute + priority queue overflow
  - Deduplication: same-category events within a window merge into one entry
  - Privacy: redacts known sensitive patterns before storage
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("tetramem.observer")

OBSERVER_SOURCE_TAG = "self-observation"

CATEGORY_ERROR = "error"
CATEGORY_BEHAVIOR = "behavior"
CATEGORY_ANOMALY = "anomaly"
CATEGORY_PERFORMANCE = "performance"
CATEGORY_SYSTEM = "system"
CATEGORY_NOISE = "noise"

CLASSIFICATION_RULES = [
    {
        "category": CATEGORY_ERROR,
        "level_filter": {"ERROR", "CRITICAL"},
        "pattern": None,
        "weight": 2.0,
        "immediate": True,
    },
    {
        "category": CATEGORY_ANOMALY,
        "level_filter": None,
        "pattern": re.compile(
            r"(timeout|crash|slow|memory.?low|oom|refused|overflow|deadlock|hung)",
            re.IGNORECASE,
        ),
        "weight": 1.8,
        "immediate": True,
    },
    {
        "category": CATEGORY_BEHAVIOR,
        "level_filter": None,
        "pattern": re.compile(
            r"(store|query|dream|organize|cascade|pulse|crystallize|hebbian|bridge|flow)",
            re.IGNORECASE,
        ),
        "weight": 0.3,
        "immediate": False,
    },
    {
        "category": CATEGORY_PERFORMANCE,
        "level_filter": None,
        "pattern": re.compile(
            r"(latency|throughput|benchmark|elapsed|duration|took\s+\d+ms|ms\b)",
            re.IGNORECASE,
        ),
        "weight": 0.5,
        "immediate": False,
    },
    {
        "category": CATEGORY_SYSTEM,
        "level_filter": {"WARNING"},
        "pattern": None,
        "weight": 0.8,
        "immediate": False,
    },
    {
        "category": CATEGORY_NOISE,
        "level_filter": {"DEBUG"},
        "pattern": re.compile(r"(heartbeat|health.?check|GET /health|ping|pong)", re.IGNORECASE),
        "weight": 0.0,
        "immediate": False,
    },
]

SENSITIVE_PATTERNS = [
    re.compile(r'(api[_-]?key\s*[:=]\s*["\']?\S+)', re.IGNORECASE),
    re.compile(r'(password\s*[:=]\s*["\']?\S+)', re.IGNORECASE),
    re.compile(r'(token\s*[:=]\s*["\']?\S+)', re.IGNORECASE),
    re.compile(r'(secret\s*[:=]\s*["\']?\S+)', re.IGNORECASE),
    re.compile(r'(Bearer\s+\S+)', re.IGNORECASE),
    re.compile(r'(X-API-Key:\s*\S+)', re.IGNORECASE),
]


@dataclass
class ObservedEvent:
    timestamp: float
    level: str
    module: str
    message: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedWindow:
    category: str
    events: List[ObservedEvent] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def count(self) -> int:
        return len(self.events)

    @property
    def components(self) -> List[str]:
        mods = set()
        for e in self.events:
            parts = e.module.split(".")
            mods.add(parts[-1] if parts else e.module)
        return sorted(mods)

    @property
    def levels(self) -> Dict[str, int]:
        lvls: Dict[str, int] = defaultdict(int)
        for e in self.events:
            lvls[e.level] += 1
        return dict(lvls)

    @property
    def sample_messages(self) -> List[str]:
        if len(self.events) <= 3:
            return [e.message for e in self.events]
        step = max(1, len(self.events) // 3)
        return [self.events[i].message for i in range(0, len(self.events), step)][:3]


@dataclass
class ObserverStats:
    total_events_received: int = 0
    events_dropped_noise: int = 0
    events_dropped_loop: int = 0
    events_dropped_rate: int = 0
    events_dropped_overflow: int = 0
    windows_aggregated: int = 0
    memories_stored: int = 0
    memories_rejected: int = 0
    current_queue_size: int = 0
    uptime_seconds: float = 0.0
    category_counts: Dict[str, int] = field(default_factory=dict)
    file_tailer_active: bool = False
    file_tailer_path: str = ""
    file_tailer_lines_read: int = 0


@dataclass
class LogEvent:
    timestamp: float
    level: str
    module: str
    message: str


DEFAULT_LOG_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
    r"(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+"
    r"(\S+)\s*[:\-]?\s*(.*)$"
)

STRUCTURED_LOG_PATTERNS = [
    re.compile(
        r"^\[(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\s*"
        r"\[(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\]\s*"
        r"\[([^\]]+)\]\s*(.*)$"
    ),
    re.compile(
        r"^(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
        r"\[(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\]\s*"
        r"(\S+)\s*[:\-]?\s*(.*)$"
    ),
    re.compile(
        r"^(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
        r"(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\s+"
        r"(\S+)\s*[:\-]?\s*(.*)$"
    ),
]

LEVEL_ALIASES = {
    "WARN": "WARNING",
    "FATAL": "CRITICAL",
    "TRACE": "DEBUG",
    "NOTICE": "INFO",
}


def _try_parse_json_line(line: str) -> Optional[Tuple[str, str, str, Dict[str, Any]]]:
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None

    level = obj.get("level", obj.get("severity", obj.get("levelname", "INFO")))
    level = LEVEL_ALIASES.get(str(level).upper(), str(level).upper())

    message = obj.get("message", obj.get("msg", obj.get("text", obj.get("m", ""))))
    if not message:
        message = line[:200]

    module = obj.get("logger", obj.get("module", obj.get("name", obj.get("source", ""))))

    ts = 0.0
    ts_val = obj.get("timestamp", obj.get("ts", obj.get("time", obj.get("@timestamp", ""))))
    if isinstance(ts_val, (int, float)):
        ts = float(ts_val)
    elif isinstance(ts_val, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                import datetime
                ts = datetime.datetime.strptime(ts_val[:26], fmt).timestamp()
                break
            except (ValueError, TypeError):
                continue

    extra = {k: v for k, v in obj.items()
             if k not in ("level", "severity", "levelname", "message", "msg", "text", "m",
                          "logger", "module", "name", "source", "timestamp", "ts", "time", "@timestamp")}
    return level, module, str(message), {"json_fields": extra, "original_timestamp": ts} if extra else {}


def _parse_line_auto(line: str) -> Tuple[str, str, str, Dict[str, Any]]:
    json_result = _try_parse_json_line(line)
    if json_result is not None:
        return json_result

    m = DEFAULT_LOG_PATTERN.match(line)
    if m:
        ts_str, level, module, message = m.groups()
        return level.upper(), module, message, {}

    for pat in STRUCTURED_LOG_PATTERNS:
        m = pat.match(line)
        if m:
            ts_str, level, module, message = m.groups()
            level = LEVEL_ALIASES.get(level.upper(), level.upper())
            return level, module, message, {}

    return "INFO", "external", line, {}


class LogFileTailer:
    """
    Tails a log file and feeds parsed lines into a RuntimeObserver.
    Handles rotation by tracking inode and file size.
    """

    def __init__(
        self,
        observer: RuntimeObserver,
        file_path: str,
        poll_interval: float = 1.0,
        log_pattern: Optional[re.Pattern] = None,
        module_override: Optional[str] = None,
    ):
        self._observer = observer
        self._file_path = file_path
        self._poll_interval = poll_interval
        self._log_pattern = log_pattern or DEFAULT_LOG_PATTERN
        self._module_override = module_override
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inode: int = 0
        self._offset: int = 0
        self._lines_read: int = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def lines_read(self) -> int:
        return self._lines_read

    @property
    def file_path(self) -> str:
        return self._file_path

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._tail_loop, daemon=True, name="observer-tailer"
        )
        self._thread.start()
        logger.info("LogFileTailer started: %s", self._file_path)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _tail_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._read_new_lines()
            except Exception as e:
                logger.debug("LogFileTailer read error: %s", e)
            self._stop_event.wait(timeout=self._poll_interval)

    def _read_new_lines(self) -> None:
        path = Path(self._file_path)
        if not path.exists():
            return
        try:
            stat = path.stat()
        except OSError:
            return
        current_inode = stat.st_ino
        current_size = stat.st_size
        if current_inode != self._inode:
            self._inode = current_inode
            self._offset = 0
        if current_size < self._offset:
            self._offset = 0
        if current_size == self._offset:
            return
        try:
            with open(self._file_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._offset)
                for line in f:
                    line = line.rstrip("\n\r")
                    if line:
                        self._parse_and_observe(line)
                self._offset = f.tell()
        except OSError:
            return

    def _parse_and_observe(self, line: str) -> None:
        self._lines_read += 1
        self._observer._stats.file_tailer_lines_read = self._lines_read

        if self._log_pattern is not DEFAULT_LOG_PATTERN:
            m = self._log_pattern.match(line)
            if m:
                ts_str, level, module, message = m.groups()
                module = self._module_override or module
            else:
                level, module, message = "INFO", self._module_override or "external", line
            self._observer.observe(
                level=level,
                module=module,
                message=message,
                source="file-tailer",
            )
            return

        level, module, message, extra = _parse_line_auto(line)
        module = self._module_override or module
        self._observer.observe(
            level=level,
            module=module,
            message=message,
            source="file-tailer",
            metadata=extra if extra else None,
        )


class _ResonanceDetector:
    __slots__ = (
        "_store_times", "_suppressed", "_cooling_until",
        "_threshold_burst", "_threshold_sustained", "_window",
        "_cooling_duration", "_cooling_decay",
    )

    def __init__(
        self,
        burst_threshold: int = 20,
        sustained_threshold: float = 0.70,
        window: float = 10.0,
        cooling_duration: float = 60.0,
        cooling_decay: float = 300.0,
    ):
        self._store_times: deque = deque()
        self._suppressed: int = 0
        self._cooling_until: float = 0.0
        self._threshold_burst = burst_threshold
        self._threshold_sustained = sustained_threshold
        self._window = window
        self._cooling_duration = cooling_duration
        self._cooling_decay = cooling_decay

    def record_store(self) -> None:
        now = time.time()
        self._store_times.append(now)
        while self._store_times and now - self._store_times[0] > self._window:
            self._store_times.popleft()

    def check_resonance(self, events_in_period: int, stores_in_period: int) -> bool:
        if stores_in_period == 0:
            return False
        ratio = stores_in_period / max(events_in_period, 1)
        now = time.time()
        if now < self._cooling_until:
            self._suppressed += 1
            elapsed = now - (self._cooling_until - self._cooling_duration)
            decay_factor = max(0.0, 1.0 - elapsed / self._cooling_decay)
            if decay_factor < 0.1:
                self._cooling_until = 0.0
                return False
            return True
        burst = len(self._store_times) >= self._threshold_burst
        sustained = ratio > self._threshold_sustained and events_in_period > 20
        if burst or sustained:
            self._cooling_until = now + self._cooling_duration
            return True
        return False

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "suppressed_total": self._suppressed,
            "cooling_active": time.time() < self._cooling_until,
            "cooling_remaining": max(0, self._cooling_until - time.time()),
            "recent_stores_in_window": len(self._store_times),
        }


class _AdaptiveRateLimiter:
    __slots__ = (
        "_base_limit", "_current_limit", "_last_adjustment",
        "_adjustment_interval", "_min_limit", "_max_limit",
    )

    def __init__(
        self,
        base_limit: int = 60,
        min_limit: int = 15,
        max_limit: int = 120,
        adjustment_interval: float = 30.0,
    ):
        self._base_limit = base_limit
        self._current_limit = float(base_limit)
        self._last_adjustment = 0.0
        self._adjustment_interval = adjustment_interval
        self._min_limit = min(float(min_limit), float(base_limit))
        self._max_limit = max(float(max_limit), float(base_limit))

    def get_limit(self) -> int:
        return int(self._current_limit)

    def adjust(self, field_stats: Dict[str, Any]) -> None:
        now = time.time()
        if now - self._last_adjustment < self._adjustment_interval:
            return
        self._last_adjustment = now

        stress = 0.0
        reg = field_stats.get("regulation", {})
        if isinstance(reg, dict):
            stress = reg.get("stress", 0.0)

        occupied = field_stats.get("occupied_nodes", 0)
        total = max(field_stats.get("total_nodes", 1), 1)
        occupancy = occupied / total

        dp = field_stats.get("dark_plane", {})
        coherence = dp.get("coherence", 0.5) if isinstance(dp, dict) else 0.5
        pe = dp.get("pe", 0.0) if isinstance(dp, dict) else 0.0

        factor = 1.0
        if stress > 0.7:
            factor *= 0.4
        elif stress > 0.4:
            factor *= 0.7

        if occupancy > 0.8:
            factor *= 0.6
        elif occupancy > 0.5:
            factor *= 0.85

        if coherence < 0.3:
            factor *= 0.5
        elif coherence < 0.5:
            factor *= 0.8

        if pe > 2.5:
            factor *= 0.7

        target = self._base_limit * factor
        self._current_limit = self._current_limit * 0.7 + target * 0.3
        self._current_limit = max(self._min_limit, min(self._max_limit, self._current_limit))

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "base_limit": self._base_limit,
            "current_limit": int(self._current_limit),
            "min_limit": int(self._min_limit),
            "max_limit": int(self._max_limit),
        }


class RuntimeObserver:
    """
    Observes AI runtime behavior through logging events, aggregates them
    into trajectory narrations, and stores them as low-weight self-observation
    memories in the TMHS.

    Pipeline:
        event → classify → (noise drop?) → (loop drop?) →
        aggregate → narrate → (rate limit?) → store(low weight)
    """

    def __init__(
        self,
        field: Any,
        window_seconds: float = 180.0,
        max_stores_per_minute: int = 60,
        queue_max_size: int = 200,
        min_events_for_window: int = 2,
        enable_behavior: bool = True,
        enable_performance: bool = True,
        enable_system: bool = True,
    ):
        self._field = field
        self._window_seconds = window_seconds
        self._max_stores_per_minute = max_stores_per_minute
        self._queue_max_size = queue_max_size
        self._min_events_for_window = min_events_for_window
        self._enable_behavior = enable_behavior
        self._enable_performance = enable_performance
        self._enable_system = enable_system
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._active_windows: Dict[str, AggregatedWindow] = {}
        self._immediate_queue: deque = deque(maxlen=queue_max_size)
        self._store_timestamps: deque = deque()
        self._stats = ObserverStats()
        self._stats.category_counts = defaultdict(int)
        self._enabled = True
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._compiled_rules = CLASSIFICATION_RULES
        self._tailers: List[LogFileTailer] = []
        self._custom_rules: Optional[List[Dict]] = None
        self._custom_classifiers: List[Callable] = []
        self._recent_categories: deque = deque(maxlen=50)
        self._on_store_callbacks: List[Callable] = []
        self._resonance_detector = _ResonanceDetector()
        self._adaptive_limiter = _AdaptiveRateLimiter(base_limit=max_stores_per_minute)
        self._config_path: Optional[str] = None
        self._config_corrupted: bool = False

    def start(self) -> None:
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._stop_event.clear()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="observer-flush"
        )
        self._flush_thread.start()
        for tailer in self._tailers:
            tailer.start()
        logger.info("RuntimeObserver started (window=%.0fs, max_stores/min=%d)",
                     self._window_seconds, self._max_stores_per_minute)

    def stop(self) -> None:
        for tailer in self._tailers:
            tailer.stop()
        self._tailers.clear()
        self._stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None
        logger.info("RuntimeObserver stopped")

    def ingest(self, event: LogEvent) -> bool:
        return self.observe(
            level=event.level,
            module=event.module,
            message=event.message,
            metadata={"original_timestamp": event.timestamp},
        )

    def add_file_tail(
        self,
        file_path: str,
        poll_interval: float = 1.0,
        log_pattern: Optional[re.Pattern] = None,
        module_override: Optional[str] = None,
    ) -> LogFileTailer:
        tailer = LogFileTailer(
            observer=self,
            file_path=file_path,
            poll_interval=poll_interval,
            log_pattern=log_pattern,
            module_override=module_override,
        )
        self._tailers.append(tailer)
        self._stats.file_tailer_active = True
        self._stats.file_tailer_path = file_path
        if self._flush_thread is not None and self._flush_thread.is_alive():
            tailer.start()
        return tailer

    def set_rules(self, rules: List[Dict]) -> None:
        compiled = []
        for r in rules:
            cat = r.get("category", CATEGORY_BEHAVIOR)
            weight = r.get("weight", 0.3)
            immediate = r.get("immediate", False)
            level_filter = set(r["level_filter"]) if "level_filter" in r else None
            pattern = re.compile(r["pattern"], re.IGNORECASE) if "pattern" in r else None
            compiled.append({
                "category": cat,
                "weight": weight,
                "immediate": immediate,
                "level_filter": level_filter,
                "pattern": pattern,
            })
        self._custom_rules = compiled
        self._compiled_rules = compiled

    def observe(
        self,
        level: str,
        module: str,
        message: str,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self._enabled:
            return False

        self._stats.total_events_received += 1

        if source == OBSERVER_SOURCE_TAG or "tetramem.observer" in module:
            self._stats.events_dropped_loop += 1
            return False

        event = ObservedEvent(
            timestamp=time.time(),
            level=level.upper(),
            module=module,
            message=self._redact(message),
            source=source,
            metadata=metadata or {},
        )

        category, weight, immediate = self._classify(event)
        if category == CATEGORY_NOISE:
            self._stats.events_dropped_noise += 1
            return False

        if category == CATEGORY_BEHAVIOR and not self._enable_behavior:
            return False
        if category == CATEGORY_PERFORMANCE and not self._enable_performance:
            return False
        if category == CATEGORY_SYSTEM and not self._enable_system:
            return False

        with self._lock:
            self._stats.category_counts[category] = (
                self._stats.category_counts.get(category, 0) + 1
            )
            if immediate:
                self._immediate_queue.append((event, category, weight))
            else:
                if category not in self._active_windows:
                    self._active_windows[category] = AggregatedWindow(
                        category=category,
                        start_time=event.timestamp,
                    )
                win = self._active_windows[category]
                win.events.append(event)
                win.end_time = event.timestamp

                if (win.end_time - win.start_time) >= self._window_seconds:
                    if win.count >= self._min_events_for_window:
                        self._flush_window(win)
                    del self._active_windows[category]

        return True

    def flush_all(self) -> int:
        stored = 0
        with self._lock:
            for cat, win in list(self._active_windows.items()):
                if win.count >= self._min_events_for_window:
                    if self._flush_window(win):
                        stored += 1
                del self._active_windows[cat]
            while self._immediate_queue:
                event, cat, weight = self._immediate_queue.popleft()
                if self._try_store(
                    self._narrate_single(event, cat), cat, weight, event
                ):
                    stored += 1
        return stored

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            self._stats.current_queue_size = (
                len(self._immediate_queue)
                + sum(w.count for w in self._active_windows.values())
            )
        self._stats.uptime_seconds = time.time() - self._start_time
        return {
            "enabled": self._enabled,
            "total_events_received": self._stats.total_events_received,
            "events_dropped_noise": self._stats.events_dropped_noise,
            "events_dropped_loop": self._stats.events_dropped_loop,
            "events_dropped_rate": self._stats.events_dropped_rate,
            "events_dropped_overflow": self._stats.events_dropped_overflow,
            "windows_aggregated": self._stats.windows_aggregated,
            "memories_stored": self._stats.memories_stored,
            "memories_rejected": self._stats.memories_rejected,
            "current_queue_size": self._stats.current_queue_size,
            "active_windows": {
                cat: {"count": w.count, "components": w.components}
                for cat, w in self._active_windows.items()
            },
            "uptime_seconds": round(self._stats.uptime_seconds, 1),
            "category_counts": dict(self._stats.category_counts),
            "config": {
                "window_seconds": self._window_seconds,
                "max_stores_per_minute": self._max_stores_per_minute,
                "queue_max_size": self._queue_max_size,
            },
            "file_tailers": [
                {
                    "path": t.file_path,
                    "active": t.is_running,
                    "lines_read": t.lines_read,
                }
                for t in self._tailers
            ],
            "custom_rules": self._custom_rules is not None,
            "custom_classifiers": len(self._custom_classifiers),
            "recent_categories": list(self._recent_categories)[-10:],
            "adaptive_limiter": self._adaptive_limiter.stats,
            "resonance": self._resonance_detector.stats,
            "config_path": self._config_path,
            "config_corrupted": self._config_corrupted,
        }

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def register_on_store(self, callback: Callable) -> None:
        self._on_store_callbacks.append(callback)

    def register_classifier(
        self,
        classifier: Callable[[ObservedEvent], Optional[Tuple[str, float, bool]]],
    ) -> None:
        self._custom_classifiers.append(classifier)

    def _classify(
        self, event: ObservedEvent
    ) -> Tuple[str, float, bool]:
        for classifier in self._custom_classifiers:
            try:
                result = classifier(event)
                if result is not None:
                    self._recent_categories.append(result[0])
                    return result
            except Exception:
                continue

        json_fields = event.metadata.get("json_fields", {})
        if json_fields:
            cat_result = self._classify_json_fields(event, json_fields)
            if cat_result is not None:
                self._recent_categories.append(cat_result[0])
                return cat_result

        for rule in self._compiled_rules:
            cat = rule["category"]
            level_ok = True
            if rule["level_filter"]:
                level_ok = event.level in rule["level_filter"]
            pattern_ok = True
            if rule["pattern"]:
                search_text = event.message
                for v in json_fields.values():
                    if isinstance(v, str):
                        search_text += " " + v
                pattern_ok = bool(rule["pattern"].search(search_text))
            if level_ok and pattern_ok:
                self._recent_categories.append(cat)
                return cat, rule["weight"], rule["immediate"]
        if event.level in ("ERROR", "CRITICAL"):
            return CATEGORY_ERROR, 2.0, True
        if event.level == "WARNING":
            return CATEGORY_SYSTEM, 0.8, False
        self._recent_categories.append(CATEGORY_BEHAVIOR)
        return CATEGORY_BEHAVIOR, 0.2, False

    def _classify_json_fields(
        self, event: ObservedEvent, fields: Dict[str, Any]
    ) -> Optional[Tuple[str, float, bool]]:
        status_code = fields.get("status_code", fields.get("statusCode", fields.get("status")))
        if isinstance(status_code, (int, str)):
            try:
                code = int(status_code)
            except (ValueError, TypeError):
                code = 0
            if code >= 500:
                return CATEGORY_ERROR, 2.0, True
            if code >= 400:
                return CATEGORY_ANOMALY, 1.2, True

        duration_ms = fields.get("duration_ms", fields.get("duration", fields.get("elapsed")))
        if isinstance(duration_ms, (int, float)):
            if duration_ms > 5000:
                return CATEGORY_PERFORMANCE, 1.0, False
            if duration_ms > 1000:
                return CATEGORY_PERFORMANCE, 0.5, False

        error_field = fields.get("error", fields.get("error_code", fields.get("errorCode")))
        if error_field:
            return CATEGORY_ERROR, 1.8, True

        request_method = fields.get("method", fields.get("httpMethod"))
        request_path = fields.get("path", fields.get("url", fields.get("uri")))
        if request_method and request_path:
            return CATEGORY_BEHAVIOR, 0.3, False

        return None

    def _redact(self, text: str) -> str:
        for pat in SENSITIVE_PATTERNS:
            text = pat.sub("[REDACTED]", text)
        return text

    def _flush_window(self, window: AggregatedWindow) -> bool:
        if window.count < self._min_events_for_window:
            return False
        narration = self._narrate_window(window)
        weight = self._window_weight(window)
        success = self._try_store(narration, window.category, weight, window.events[0])
        if success:
            self._stats.windows_aggregated += 1
        return success

    def _narrate_window(self, window: AggregatedWindow) -> str:
        duration = window.end_time - window.start_time
        parts = [
            f"[Trajectory:{window.category}] ",
            f"{window.count} events in {duration:.0f}s ",
            f"from {', '.join(window.components)}: ",
        ]
        samples = window.sample_messages[:2]
        for s in samples:
            truncated = s[:120] + "..." if len(s) > 120 else s
            parts.append(f'"{truncated}" ')
        if window.category == CATEGORY_ERROR:
            parts.append("System experienced errors. ")
        elif window.category == CATEGORY_ANOMALY:
            parts.append("Anomalous patterns detected. ")
        elif window.category == CATEGORY_BEHAVIOR:
            parts.append("Behavioral trajectory recorded. ")
        elif window.category == CATEGORY_PERFORMANCE:
            parts.append("Performance metrics observed. ")
        return "".join(parts).strip()

    def _narrate_single(self, event: ObservedEvent, category: str) -> str:
        truncated = event.message[:200] + "..." if len(event.message) > 200 else event.message
        return (
            f"[Trajectory:{category}] "
            f"Immediate {event.level} from {event.module}: "
            f'"{truncated}"'
        )

    def _window_weight(self, window: AggregatedWindow) -> float:
        base_map = {
            CATEGORY_ERROR: 1.5,
            CATEGORY_ANOMALY: 1.2,
            CATEGORY_SYSTEM: 0.5,
            CATEGORY_PERFORMANCE: 0.3,
            CATEGORY_BEHAVIOR: 0.2,
        }
        base = base_map.get(window.category, 0.2)
        level_severity = sum(
            1 for e in window.events if e.level in ("ERROR", "CRITICAL", "WARNING")
        )
        severity_factor = 1.0 + 0.1 * min(level_severity, 10)
        return min(base * severity_factor, 3.0)

    def _try_store(
        self, narration: str, category: str, weight: float, event: ObservedEvent
    ) -> bool:
        now = time.time()
        while self._store_timestamps and now - self._store_timestamps[0] > 60.0:
            self._store_timestamps.popleft()

        try:
            field_stats = self._field.stats() if hasattr(self._field, 'stats') else {}
            if hasattr(self._field, '_regulation') and self._field._regulation is not None:
                field_stats["regulation"] = self._field._regulation.status()
            if hasattr(self._field, '_dark_substrate') and self._field._dark_substrate is not None:
                field_stats["dark_plane"] = self._field._dark_substrate.get_stats()
        except Exception:
            field_stats = {}

        self._adaptive_limiter.adjust(field_stats)
        current_limit = self._adaptive_limiter.get_limit()

        total_events = self._stats.total_events_received
        total_stores = self._stats.memories_stored
        if self._resonance_detector.check_resonance(total_events, total_stores):
            self._stats.events_dropped_rate += 1
            self._stats.memories_rejected += 1
            return False

        if len(self._store_timestamps) >= current_limit:
            self._stats.events_dropped_rate += 1
            self._stats.memories_rejected += 1
            return False

        try:
            memory_id = self._field.store(
                content=narration,
                labels=[category, "meta", "trajectory", OBSERVER_SOURCE_TAG],
                weight=weight,
                metadata={
                    "source": OBSERVER_SOURCE_TAG,
                    "category": category,
                    "observer_event_count": 1,
                    "observer_component": event.module,
                    "observer_level": event.level,
                },
            )
            self._store_timestamps.append(now)
            self._stats.memories_stored += 1
            self._resonance_detector.record_store()
            logger.debug("Observer stored trajectory memory: %s (%.2f)", memory_id, weight)
            for cb in self._on_store_callbacks:
                try:
                    cb(narration, category, weight, {
                        "source": OBSERVER_SOURCE_TAG,
                        "category": category,
                        "observer_component": event.module,
                        "observer_level": event.level,
                    })
                except Exception:
                    pass
            return True
        except Exception as e:
            self._stats.memories_rejected += 1
            logger.warning("Observer failed to store memory: %s", e)
            return False

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=30.0)
            if self._stop_event.is_set():
                break
            with self._lock:
                for cat in list(self._active_windows.keys()):
                    win = self._active_windows[cat]
                    elapsed = time.time() - win.start_time
                    if elapsed >= self._window_seconds:
                        if win.count >= self._min_events_for_window:
                            self._flush_window(win)
                        elif win.count >= 1:
                            evt = win.events[0]
                            self._try_store(
                                self._narrate_single(evt, cat), cat,
                                self._window_weight(win), evt,
                            )
                        del self._active_windows[cat]
                while self._immediate_queue:
                    event, cat, weight = self._immediate_queue.popleft()
                    self._try_store(
                        self._narrate_single(event, cat), cat, weight, event
                    )


class TetraMemLogHandler(logging.Handler):
    """
    Logging handler that feeds log records into the RuntimeObserver.
    Attach to any logger to bridge Python logging → observer pipeline.
    """

    def __init__(self, observer: RuntimeObserver, level: int = logging.WARNING):
        super().__init__(level)
        self._observer = observer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            source = getattr(record, "source", None)
            metadata = {}
            for attr in ("funcName", "lineno", "thread", "threadName"):
                val = getattr(record, attr, None)
                if val is not None:
                    metadata[attr] = val
            if hasattr(record, "extra"):
                metadata.update(record.extra)
            self._observer.observe(
                level=record.levelname,
                module=record.name,
                message=self.format(record),
                source=source,
                metadata=metadata,
            )
        except Exception:
            pass


def attach_file_observer(
    field: Any,
    file_path: str,
    poll_interval: float = 1.0,
    module_override: Optional[str] = None,
    **kwargs: Any,
) -> RuntimeObserver:
    """
    Convenience: create a RuntimeObserver attached to a log file.
    Starts both the observer flush thread and the file tailer.
    """
    observer = RuntimeObserver(field=field, **kwargs)
    observer.start()
    observer.add_file_tail(
        file_path=file_path,
        poll_interval=poll_interval,
        module_override=module_override,
    )
    return observer


def attach_callback_observer(
    field: Any,
    **kwargs: Any,
) -> RuntimeObserver:
    """
    Convenience: create a RuntimeObserver for programmatic ingest().
    No file tailing, no log handler. Use observer.ingest(LogEvent(...)).
    """
    observer = RuntimeObserver(field=field, **kwargs)
    observer.start()
    return observer
