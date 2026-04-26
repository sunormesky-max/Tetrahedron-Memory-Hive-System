from __future__ import annotations

import asyncio
import json
import logging
import os
import signal as sig_module
import time
import threading
from typing import Any, Dict, List, Optional

from tetrahedron_memory.honeycomb_neural_field import HoneycombNeuralField
from tetrahedron_memory.auth import AuthManager
from tetrahedron_memory.persistence_engine import PersistenceEngine
from tetrahedron_memory.system_ops import SystemOperationManager
from tetrahedron_memory.agent_loop import AgentMemoryLoop
from tetrahedron_memory.insight_aggregator import InsightAggregator
from tetrahedron_memory.phase_transition_honeycomb import HoneycombPhaseTransition
from tetrahedron_memory.enterprise import VersionControl, QuotaManager, BackupManager
from tetrahedron_memory.audit_log import AuditLog
from tetrahedron_memory.observability import SimpleMetrics

log = logging.getLogger("tetramem.api")


def _resolve_node(field, node_id: str) -> str:
    if node_id in field._nodes:
        return node_id
    matches = [nid for nid in field._nodes if nid.startswith(node_id)]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(f"Ambiguous ID prefix: {node_id}")
    return node_id


class AppState:
    def __init__(self) -> None:
        self.field: Optional[HoneycombNeuralField] = None
        self.state_lock: Optional[threading.RLock] = None
        self.persistence: Optional[PersistenceEngine] = None
        self.proactive_engine_stop: Optional[threading.Event] = None
        self.loading_complete: bool = False
        self.start_time: float = 0.0
        self.system_ops: Optional[SystemOperationManager] = None
        self.agent_hb_lock: Optional[threading.Lock] = None
        self.agent_heartbeats: Dict[str, float] = {}
        self.insight_aggregator: Optional[InsightAggregator] = None
        self.phase_detector: Optional[HoneycombPhaseTransition] = None
        self.agent_loop: Optional[AgentMemoryLoop] = None
        self.event_subscribers: List = []
        self.event_subscriber_lock: Optional[threading.Lock] = None
        self._event_cooldowns: Dict[str, float] = {}
        self._event_cooldown_duration: Dict[str, float] = {
            "self_check_alert": 600,
            "self_organize_update": 600,
            "agent_idle": 600,
            "sessions_expired": 1800,
            "insight_high_priority": 1800,
            "auto_recovery": 600,
            "auto_recovery_failed": 300,
            "degradation_alert": 900,
            "evolution_cycle_completed": 60,
            "proactive_triggered": 60,
        }
        self.storage_dir: str = ""
        self.auth_manager: Optional[AuthManager] = None
        self.quota_manager: Optional[QuotaManager] = None
        self.audit_log: Optional[AuditLog] = None
        self.backup_manager: Optional[BackupManager] = None
        self.version_control: Optional[VersionControl] = None
        self.metrics: Optional[SimpleMetrics] = None

    def emit_event(self, event_name: str, data: dict) -> None:
        now = time.time()
        min_interval = self._event_cooldowns.get(event_name, 0)
        if now < min_interval:
            return
        self._event_cooldowns[event_name] = now + self._event_cooldown_duration.get(event_name, 300)
        msg = json.dumps({"event": event_name, **data})
        with self.event_subscriber_lock:
            stale = [q for q in self.event_subscribers if q.full()]
            for q in stale:
                try:
                    self.event_subscribers.remove(q)
                except ValueError:
                    pass
            for q in self.event_subscribers:
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

    def log_op(self, op: str, detail: dict = None) -> None:
        detail = detail or {}
        log.info("[OP] %s %s", op, json.dumps(detail, default=str, ensure_ascii=False)[:200])

    def initialize(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        )

        self.storage_dir = os.environ.get("TETRAMEM_STORAGE", "./tetramem_data_v2")
        os.makedirs(self.storage_dir, exist_ok=True)

        self.start_time = time.time()
        self.state_lock = threading.RLock()
        self.loading_complete = False
        self.proactive_engine_stop = threading.Event()

        resolution = int(os.environ.get("TETRAMEM_RESOLUTION", "5"))
        spacing = float(os.environ.get("TETRAMEM_SPACING", "1.0"))
        self.field = HoneycombNeuralField(resolution=resolution, spacing=spacing)
        self.field.initialize()

        persist_dir = os.path.join(self.storage_dir, "persistence")
        self.persistence = PersistenceEngine(
            storage_dir=persist_dir,
            checkpoint_interval=int(os.environ.get("TETRAMEM_CHECKPOINT_INTERVAL", "300")),
        )
        self.persistence.open()

        latest = self.persistence.recover()
        if latest:
            try:
                self.field.import_full_state(latest)
                n_nodes = len(latest.get("nodes", {}))
                log.info("Restored %d nodes from checkpoint", n_nodes)
                print(f"[TetraMem v8.0] Restored {n_nodes} nodes from checkpoint")
            except Exception as e:
                log.error("Checkpoint restore failed: %s", e)
                print(f"[TetraMem v8.0] Checkpoint restore failed: {e}")

        self.auth_manager = AuthManager(secret_key=os.environ.get("TETRAMEM_SECRET_KEY"))
        self.quota_manager = QuotaManager()
        self.audit_log = AuditLog()
        self.backup_manager = BackupManager(self.storage_dir)
        self.version_control = VersionControl()

        self.system_ops = SystemOperationManager(
            field=self.field, persistence=self.persistence, storage_dir=self.storage_dir
        )

        self.agent_hb_lock = threading.Lock()
        self.agent_heartbeats = {}
        self.insight_aggregator = InsightAggregator()
        self.agent_loop = AgentMemoryLoop(self.field)
        self.phase_detector = HoneycombPhaseTransition()

        self.event_subscribers = []
        self.event_subscriber_lock = threading.Lock()

        self.metrics = SimpleMetrics()

        self.field.start_pulse_engine()

        threading.Thread(target=self._proactive_loop, daemon=True, name="proactive").start()
        threading.Thread(target=self._session_cleanup_loop, daemon=True, name="session-cleanup").start()
        threading.Thread(target=self._insight_loop, daemon=True, name="insight").start()
        threading.Thread(target=self._system_health_loop, daemon=True, name="system-health").start()
        threading.Thread(target=self._auto_checkpoint_loop, daemon=True, name="auto-checkpoint").start()

        self.loading_complete = True
        print(f"[TetraMem v8.0] Initialized: resolution={resolution}, spacing={spacing}")

    def _proactive_loop(self) -> None:
        cycle = 0
        while not self.proactive_engine_stop.is_set():
            self.proactive_engine_stop.wait(30)
            if self.proactive_engine_stop.is_set() or not self.loading_complete:
                continue
            cycle += 1
            try:
                if cycle % 5 == 0:
                    s = state.self_check_status()
                    anomalies = s.get("latest_check", {}).get("anomalies_found", 0)
                    if anomalies > 0:
                        self.emit_event("self_check_alert", {
                            "anomalies": anomalies,
                            "repairs": s.get("latest_check", {}).get("repairs_succeeded", 0),
                        })

                if cycle % 12 == 0:
                    with self.state_lock:
                        so = self.field.self_organize_stats()
                    if so.get("active_shortcuts", 0) > 0:
                        self.emit_event("self_organize_update", {
                            "clusters": so.get("active_clusters", 0),
                            "shortcuts": so.get("active_shortcuts", 0),
                            "entropy": so.get("latest_entropy"),
                        })

                if cycle % 30 == 0:
                    now = time.time()
                    with self.agent_hb_lock:
                        idle_agents = [aid for aid, ts in list(self.agent_heartbeats.items()) if now - ts > 60]
                        for aid in idle_agents:
                            if now - self.agent_heartbeats.get(aid, 0) > 300:
                                self.agent_heartbeats.pop(aid, None)
                                continue
                        stale = [aid for aid, ts in list(self.agent_heartbeats.items()) if now - ts > 300]
                        for aid in stale:
                            self.agent_heartbeats.pop(aid, None)
                    for aid in idle_agents:
                        if aid not in self.agent_heartbeats:
                            continue
                        self.emit_event("agent_idle", {"agent_id": aid, "idle_seconds": int(now - self.agent_heartbeats.get(aid, 0))})
            except Exception as e:
                log.error("Proactive loop error: %s", e)

    def _session_cleanup_loop(self) -> None:
        while not self.proactive_engine_stop.is_set():
            self.proactive_engine_stop.wait(120)
            if self.proactive_engine_stop.is_set() or not self.loading_complete:
                continue
            try:
                with self.state_lock:
                    result = self.field.session_cleanup(max_age=3600)
                if result.get("expired", 0) > 0:
                    self.emit_event("sessions_expired", result)
            except Exception as e:
                log.error("Session cleanup error: %s", e)

    def _insight_loop(self) -> None:
        while not self.proactive_engine_stop.is_set():
            self.proactive_engine_stop.wait(120)
            if self.proactive_engine_stop.is_set() or not self.loading_complete:
                continue
            if self.insight_aggregator is None:
                continue
            try:
                insights = self.insight_aggregator.collect()
                for ins in insights:
                    if ins.get("priority", 0) >= 7:
                        self.emit_event("insight_high_priority", {
                            "type": ins.get("type"),
                            "title": ins.get("title"),
                            "priority": ins.get("priority"),
                            "action": ins.get("action"),
                        })
            except Exception as e:
                log.error("Insight loop error: %s", e)

    def _system_health_loop(self) -> None:
        while not self.proactive_engine_stop.is_set():
            self.proactive_engine_stop.wait(120)
            if self.proactive_engine_stop.is_set() or not self.loading_complete:
                continue
            if self.system_ops is None:
                continue
            try:
                report = self.system_ops.run_health_check()
                for issue in report.get("issues", []):
                    if issue in ("stale_checkpoint", "wal_too_large", "pulse_dead", "orphan_nodes", "memory_high", "disk_low"):
                        recovered = self.system_ops.auto_recover(issue)
                        if recovered:
                            self.emit_event("auto_recovery", {"issue": issue, "success": True})
                        else:
                            self.emit_event("auto_recovery_failed", {"issue": issue, "success": False})
                if self.system_ops.degradation_level > 0:
                    self.emit_event("degradation_alert", {
                        "level": self.system_ops.degradation_level,
                        "issues": report.get("issues", []),
                    })
                self.system_ops.check_scheduled_backups()
            except Exception as e:
                log.error("System health loop error: %s", e)

    def _auto_checkpoint_loop(self) -> None:
        while not self.proactive_engine_stop.is_set():
            self.proactive_engine_stop.wait(60)
            if self.proactive_engine_stop.is_set() or not self.loading_complete:
                continue
            if self.persistence is not None and self.persistence.should_checkpoint():
                try:
                    with self.state_lock:
                        state = self.field.export_full_state()
                    self.persistence.checkpoint(state)
                    log.info("Auto-checkpoint: %d memories", len(state.get("nodes", {})))
                except Exception as e:
                    log.error("Auto-checkpoint failed: %s", e)

    def emergency_save(self, signum=None, frame=None) -> None:
        print(f"[TetraMem v8.0] Signal {signum} received — emergency save starting")
        if not self.loading_complete or self.field is None or self.persistence is None:
            return
        try:
            with self.state_lock:
                state = self.field.export_full_state()
            self.persistence.checkpoint(state)
            n = len(state.get("nodes", {}))
            print(f"[TetraMem v8.0] Emergency save completed: {n} memories")
        except Exception as e:
            print(f"[TetraMem v8.0] Emergency save failed: {e}")
        finally:
            if self.persistence is not None:
                self.persistence.close()

    def register_signal_handlers(self) -> None:
        for s in (sig_module.SIGTERM, sig_module.SIGINT):
            try:
                sig_module.signal(s, self.emergency_save)
            except (OSError, ValueError):
                pass
