"""
System Operation Manager — long-term operation hardening.
Scheduled backups, health monitoring, auto-recovery, data integrity.
Pure Python. No external engines.
"""
from __future__ import annotations

import glob as _glob
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField
    from .persistence_engine import PersistenceEngine


class SystemOperationManager:

    BACKUP_DIR_NAME = "scheduled_backups"
    BACKUP_INDEX_FILE = "_schedule_index.json"

    _log = logging.getLogger("tetramem.system_ops")

    def __init__(
        self,
        field: "HoneycombNeuralField",
        persistence: "PersistenceEngine",
        storage_dir: str,
    ):
        self._field = field
        self._persistence = persistence
        self._storage_dir = storage_dir
        self._backup_schedule = {
            "hourly": {"interval": 3600, "keep": 24},
            "daily": {"interval": 86400, "keep": 30},
            "weekly": {"interval": 604800, "keep": 12},
        }
        self._last_backup_time: Dict[str, float] = {
            level: 0.0 for level in self._backup_schedule
        }
        self._health_history: List[Dict] = []
        self._max_health_history = 200
        self._degradation_level = 0
        self._last_health_check: Optional[Dict] = None
        self._recovery_attempts: Dict[str, int] = {}
        self._recovery_successes: Dict[str, int] = {}

        self._backup_dir = os.path.join(storage_dir, self.BACKUP_DIR_NAME)
        os.makedirs(self._backup_dir, exist_ok=True)
        self._backup_index_path = os.path.join(
            self._backup_dir, self.BACKUP_INDEX_FILE
        )
        self._backup_index: List[Dict] = self._load_backup_index()

        self._rollback_points: Dict[str, Dict] = {}
        self._periodic_backup_thread: Optional[threading.Thread] = None
        self._periodic_backup_stop = threading.Event()
        self._periodic_backup_interval: float = 0.0

    @property
    def degradation_level(self) -> int:
        return self._degradation_level

    @property
    def last_health_check(self) -> Optional[Dict]:
        return self._last_health_check

    def run_health_check(self) -> Dict[str, Any]:
        checks: Dict[str, Dict[str, Any]] = {}
        issues: List[str] = []
        actions_taken: List[str] = []

        checks["persistence"] = self._check_persistence()
        if checks["persistence"]["status"] != "ok":
            issues.append("persistence")

        checks["lattice_integrity"] = self._check_lattice_integrity()
        if checks["lattice_integrity"]["status"] != "ok":
            issues.append("orphan_nodes")

        checks["memory_pressure"] = self._check_memory_pressure()
        if checks["memory_pressure"]["status"] != "ok":
            if checks["memory_pressure"].get("level") == "critical":
                issues.append("memory_high")

        checks["disk_space"] = self._check_disk_space()
        if checks["disk_space"]["status"] != "ok":
            issues.append("disk_low")

        checks["pulse_engine"] = self._check_pulse_engine()
        if checks["pulse_engine"]["status"] != "ok":
            issues.append("pulse_dead")

        checks["background_threads"] = self._check_background_threads()
        if checks["background_threads"]["status"] != "ok":
            issues.append("threads_dead")

        checks["wal_size"] = self._check_wal_size()
        if checks["wal_size"]["status"] != "ok":
            issues.append("wal_too_large")

        checks["checkpoint_freshness"] = self._check_checkpoint_freshness()
        if checks["checkpoint_freshness"]["status"] != "ok":
            issues.append("stale_checkpoint")

        prev_level = self._degradation_level
        self._degradation_level = self._compute_degradation(checks)
        if self._degradation_level > prev_level:
            actions_taken.append(
                f"degradation_escalated:{prev_level}->{self._degradation_level}"
            )

        if self._degradation_level == 3:
            try:
                self._force_emergency_backup()
                actions_taken.append("emergency_backup_created")
            except Exception as e:
                self._log.error("Emergency backup failed: %s", e)

        result = {
            "status": (
                "ok"
                if self._degradation_level == 0
                else (
                    "degraded"
                    if self._degradation_level == 1
                    else (
                        "critical"
                        if self._degradation_level == 2
                        else "emergency"
                    )
                )
            ),
            "checks": checks,
            "degradation_level": self._degradation_level,
            "issues": issues,
            "actions_taken": actions_taken,
            "timestamp": time.time(),
        }

        self._last_health_check = result
        self._health_history.append(result)
        if len(self._health_history) > self._max_health_history:
            self._health_history = self._health_history[
                -self._max_health_history // 2 :
            ]

        return result

    def auto_recover(self, issue: str) -> bool:
        self._recovery_attempts[issue] = (
            self._recovery_attempts.get(issue, 0) + 1
        )
        try:
            if issue == "stale_checkpoint":
                return self._recover_stale_checkpoint()
            elif issue == "wal_too_large":
                return self._recover_wal_too_large()
            elif issue == "pulse_dead":
                return self._recover_pulse_dead()
            elif issue == "orphan_nodes":
                return self._recover_orphan_nodes()
            elif issue == "memory_high":
                return self._recover_memory_high()
            elif issue == "disk_low":
                return self._recover_disk_low()
            return False
        except Exception as e:
            self._log.error("Auto-recovery failed for %s: %s", issue, e)
            return False

    def create_scheduled_backup(self, level: str = "daily") -> str:
        if level not in self._backup_schedule:
            level = "daily"

        schedule = self._backup_schedule[level]
        backup_id = hashlib.sha256(
            f"{time.time()}{level}".encode()
        ).hexdigest()[:16]
        ts = time.time()
        backup_path = os.path.join(
            self._backup_dir, f"{level}_{backup_id}"
        )
        os.makedirs(backup_path, exist_ok=True)

        with self._field.field_lock():
            state = self._field.export_full_state()

        meta = {
            "backup_id": backup_id,
            "level": level,
            "timestamp": ts,
            "node_count": len(state.get("nodes", {})),
        }
        state["_backup_meta"] = meta

        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json", dir=backup_path, prefix=".state_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    self._log.debug("fsync failed for backup %s", backup_id)
            final_path = os.path.join(backup_path, "state.json")
            os.replace(tmp_path, final_path)
        except Exception:
            self._log.error("Failed to write backup %s", backup_id, exc_info=True)
            try:
                os.unlink(tmp_path)
            except OSError:
                self._log.debug("cleanup failed for %s", tmp_path)
            raise

        entry = {
            "id": backup_id,
            "level": level,
            "ts": ts,
            "path": backup_path,
            "node_count": meta["node_count"],
        }
        self._backup_index.append(entry)
        self._last_backup_time[level] = ts
        self._prune_backups(level, schedule["keep"])
        self._save_backup_index()

        return backup_id

    def restore_backup(self, backup_id: str) -> bool:
        entry = None
        for e in self._backup_index:
            if e["id"] == backup_id:
                entry = e
                break
        if entry is None:
            return False

        state_path = os.path.join(entry["path"], "state.json")
        if not os.path.exists(state_path):
            return False

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            self._log.error("Failed to load backup %s: %s", backup_id, e)
            return False

        meta = state.get("_backup_meta", {})
        expected_nodes = meta.get("node_count", -1)
        actual_nodes = len(state.get("nodes", {}))
        if expected_nodes >= 0 and actual_nodes != expected_nodes:
            return False

        with self._field.field_lock():
            self._field.import_full_state(state)
        return True

    def verify_data_integrity(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "checks": {},
            "issues": [],
            "status": "ok",
        }

        with self._field.field_lock():
            nodes_in_memory = {
                nid
                for nid, n in self._field.all_nodes_items()
                if n.is_occupied
            }
            results["checks"]["memory_node_count"] = len(nodes_in_memory)

            checkpoint = self._persistence.recover()
            if checkpoint:
                nodes_in_checkpoint = set(checkpoint.get("nodes", {}).keys())
                results["checks"]["checkpoint_node_count"] = len(
                    nodes_in_checkpoint
                )
                only_in_memory = nodes_in_memory - nodes_in_checkpoint
                only_in_checkpoint = nodes_in_checkpoint - nodes_in_memory
                if only_in_memory:
                    results["issues"].append(
                        f"{len(only_in_memory)} nodes in memory but not in checkpoint"
                    )
                    results["checks"]["memory_only_count"] = len(
                        only_in_memory
                    )
                if only_in_checkpoint:
                    results["issues"].append(
                        f"{len(only_in_checkpoint)} nodes in checkpoint but not in memory"
                    )
                    results["checks"]["checkpoint_only_count"] = len(
                        only_in_checkpoint
                    )
            else:
                results["checks"]["checkpoint_node_count"] = 0

            seen_ids: Dict[str, int] = {}
            duplicate_ids = []
            for nid in nodes_in_memory:
                short = nid[:8]
                seen_ids[short] = seen_ids.get(short, 0) + 1
                if seen_ids[short] > 1:
                    duplicate_ids.append(short)
            if duplicate_ids:
                results["issues"].append(
                    f"{len(duplicate_ids)} duplicate ID prefixes detected"
                )
                results["checks"]["duplicate_ids"] = duplicate_ids[:5]
            results["checks"]["duplicate_check"] = (
                "ok" if not duplicate_ids else "duplicates_found"
            )

            invalid_edges = 0
            for n1, n2, etype in self._field.edges_items():
                if not self._field.node_exists(n1) or not self._field.node_exists(n2):
                    invalid_edges += 1
            if invalid_edges > 0:
                results["issues"].append(
                    f"{invalid_edges} edges with invalid endpoints"
                )
            results["checks"]["invalid_edges"] = invalid_edges

            content_hashes = {}
            hash_mismatches = 0
            for nid in nodes_in_memory:
                node = self._field.node_get(nid)
                if node and node.content:
                    h = hashlib.sha256(
                        node.content.encode("utf-8")
                    ).hexdigest()[:12]
                    content_hashes[nid] = h
            results["checks"][
                "content_hash_sample"
            ] = hash_mismatches

            centroid_issues = 0
            spacing = self._field.get_spacing()
            resolution = self._field.get_resolution()
            max_extent = spacing * resolution * 2
            for nid in nodes_in_memory:
                node = self._field.node_get(nid)
                if node is not None:
                    pos = node.position
                    if hasattr(pos, "__len__") and len(pos) == 3:
                        for coord in pos:
                            val = float(coord)
                            if abs(val) > max_extent * 2:
                                centroid_issues += 1
                                break
            if centroid_issues > 0:
                results["issues"].append(
                    f"{centroid_issues} nodes with centroids outside lattice bounds"
                )
            results["checks"]["centroid_bounds_issues"] = centroid_issues

        if results["issues"]:
            results["status"] = "issues_found"
        return results

    def get_degradation_status(self) -> Dict[str, Any]:
        level = self._degradation_level
        features: Dict[str, Any] = {
            "pulse_engine": True,
            "dream_engine": True,
            "new_stores": True,
            "self_organize": True,
            "background_maintenance": True,
        }
        if level >= 1:
            features["pulse_engine"] = "reduced_frequency"
            features["dream_engine"] = False
        if level >= 2:
            features["new_stores"] = False
            features["self_organize"] = False
        if level >= 3:
            features["pulse_engine"] = False
            features["background_maintenance"] = False

        return {
            "level": level,
            "level_name": (
                "normal"
                if level == 0
                else (
                    "degraded"
                    if level == 1
                    else ("critical" if level == 2 else "emergency")
                )
            ),
            "features": features,
            "last_health_check": self._last_health_check,
            "health_check_count": len(self._health_history),
            "recovery_stats": {
                "attempts": dict(self._recovery_attempts),
                "successes": dict(self._recovery_successes),
            },
            "backup_count": len(self._backup_index),
            "last_backup_times": dict(self._last_backup_time),
        }

    def check_scheduled_backups(self) -> List[str]:
        now = time.time()
        created = []
        for level, schedule in self._backup_schedule.items():
            last = self._last_backup_time.get(level, 0)
            if now - last >= schedule["interval"]:
                try:
                    bid = self.create_scheduled_backup(level)
                    created.append(f"{level}:{bid}")
                except Exception as exc:
                    self._log.warning("Scheduled backup failed for level %s: %s", level, exc)
        return created

    def create_rollback_point(self, label: str) -> Dict[str, Any]:
        ts = time.time()
        with self._field.field_lock():
            state = self._field.export_full_state()

        state_hash = hashlib.sha256(
            json.dumps(state, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]

        entry = {
            "label": label,
            "timestamp": ts,
            "state_hash": state_hash,
            "node_count": len(state.get("nodes", {})),
            "state": state,
        }

        self._rollback_points[label] = entry

        rollback_dir = os.path.join(self._backup_dir, "rollback_points")
        os.makedirs(rollback_dir, exist_ok=True)
        safe_label = "".join(c for c in label if c.isalnum() or c in ("_", "-"))[:64]
        if not safe_label:
            safe_label = state_hash
        rollback_path = os.path.join(rollback_dir, f"{safe_label}.json")
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json", dir=rollback_dir, prefix=".rb_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    self._log.debug("fsync failed for rollback point %s", label)
            os.replace(tmp_path, rollback_path)
        except Exception:
            self._log.error("Failed to write rollback point %s", label, exc_info=True)
            try:
                os.unlink(tmp_path)
            except OSError:
                self._log.debug("cleanup failed for %s", tmp_path)
            raise

        return {
            "label": label,
            "timestamp": ts,
            "state_hash": state_hash,
            "node_count": entry["node_count"],
        }

    def rollback_to(self, label: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "label": label,
            "rolled_back": False,
            "error": None,
        }

        entry = self._rollback_points.get(label)
        if entry is None:
            rollback_dir = os.path.join(self._backup_dir, "rollback_points")
            safe_label = "".join(c for c in label if c.isalnum() or c in ("_", "-"))[:64]
            rollback_path = os.path.join(rollback_dir, f"{safe_label}.json")
            if os.path.exists(rollback_path):
                try:
                    with open(rollback_path, "r", encoding="utf-8") as f:
                        entry = json.load(f)
                    self._rollback_points[label] = entry
                except Exception as e:
                    self._log.warning("Failed to load rollback point %s: %s", label, e)
                    result["error"] = f"Failed to load rollback point: {e}"
                    return result
            else:
                result["error"] = f"Rollback point '{label}' not found"
                return result

        state = entry.get("state")
        if state is None:
            result["error"] = "Rollback point contains no state data"
            return result

        with self._field.field_lock():
            before_count = sum(
                1 for n in self._field.all_nodes_values() if n.is_occupied
            )
            self._field.import_full_state(state)
            after_count = sum(
                1 for n in self._field.all_nodes_values() if n.is_occupied
            )

        if self._persistence.is_dirty():
            try:
                with self._field.field_lock():
                    current_state = self._field.export_full_state()
                self._persistence.checkpoint(current_state)
            except Exception:
                self._log.error("Post-rollback checkpoint failed for %s", label, exc_info=True)

        result["rolled_back"] = True
        result["nodes_before"] = before_count
        result["nodes_after"] = after_count
        result["rolled_back_to_timestamp"] = entry.get("timestamp")
        result["state_hash"] = entry.get("state_hash")
        return result

    def schedule_periodic_backup(self, interval_seconds: float) -> Dict[str, Any]:
        if interval_seconds < 60:
            interval_seconds = 60

        self.stop_periodic_backup()

        self._periodic_backup_interval = interval_seconds
        self._periodic_backup_stop.clear()
        self._periodic_backup_thread = threading.Thread(
            target=self._periodic_backup_loop,
            name="periodic-backup",
            daemon=True,
        )
        self._periodic_backup_thread.start()

        return {
            "scheduled": True,
            "interval_seconds": interval_seconds,
            "message": f"Periodic backup scheduled every {interval_seconds}s",
        }

    def stop_periodic_backup(self) -> Dict[str, Any]:
        was_running = (
            self._periodic_backup_thread is not None
            and self._periodic_backup_thread.is_alive()
        )
        self._periodic_backup_stop.set()
        if self._periodic_backup_thread is not None:
            self._periodic_backup_thread.join(timeout=10)
            self._periodic_backup_thread = None

        return {
            "stopped": was_running,
            "message": (
                "Periodic backup stopped"
                if was_running
                else "No periodic backup was running"
            ),
        }

    def get_backup_history(self, n: int = 20) -> List[Dict[str, Any]]:
        history = sorted(
            self._backup_index,
            key=lambda e: e.get("ts", 0),
            reverse=True,
        )
        result = []
        for entry in history[:n]:
            item = {
                "backup_id": entry.get("id", ""),
                "level": entry.get("level", "unknown"),
                "timestamp": entry.get("ts", 0),
                "age_seconds": round(time.time() - entry.get("ts", time.time()), 1),
                "node_count": entry.get("node_count", 0),
                "path": entry.get("path", ""),
                "exists": os.path.exists(
                    os.path.join(entry.get("path", ""), "state.json")
                ),
            }
            state_path = os.path.join(entry.get("path", ""), "state.json")
            if os.path.exists(state_path):
                try:
                    size = os.path.getsize(state_path)
                    item["file_size_bytes"] = size
                    item["file_size_mb"] = round(size / (1024 * 1024), 2)
                except OSError:
                    self._log.debug("cleanup failed for %s", state_path)
            result.append(item)
        return result

    def _periodic_backup_loop(self):
        while not self._periodic_backup_stop.wait(
            timeout=self._periodic_backup_interval
        ):
            try:
                self.create_scheduled_backup("daily")
            except Exception as exc:
                self._log.warning("Periodic backup failed: %s", exc)

    def _check_persistence(self) -> Dict[str, Any]:
        try:
            p = self._persistence
            if p._closed:
                return {"status": "degraded", "error": "persistence closed"}
            if p._wal_fd is None:
                return {"status": "degraded", "error": "WAL not open"}
            return {
                "status": "ok",
                "dirty_ops": p.dirty_ops,
                "is_dirty": p.is_dirty(),
            }
        except Exception as e:
            self._log.warning("Persistence check failed: %s", e)
            return {"status": "degraded", "error": str(e)}

    def _check_lattice_integrity(self) -> Dict[str, Any]:
        try:
            orphans = 0
            with self._field.field_lock():
                for nid, node in self._field.all_nodes_items():
                    if not node.is_occupied:
                        continue
                    has_occupied_neighbor = False
                    for fnid in node.face_neighbors:
                        fn = self._field.node_get(fnid)
                        if fn and fn.is_occupied:
                            has_occupied_neighbor = True
                            break
                    if not has_occupied_neighbor:
                        orphans += 1
            occupied = sum(
                1
                for n in self._field.all_nodes_values()
                if n.is_occupied
            )
            orphan_rate = orphans / max(1, occupied)
            return {
                "status": "ok" if orphan_rate < 0.5 else "degraded",
                "orphan_count": orphans,
                "occupied_count": occupied,
                "orphan_rate": round(orphan_rate, 3),
            }
        except Exception as e:
            self._log.warning("Lattice integrity check failed: %s", e)
            return {"status": "degraded", "error": str(e)}

    def _check_memory_pressure(self) -> Dict[str, Any]:
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            level = (
                "normal"
                if mem_mb < 1024
                else ("warning" if mem_mb < 2048 else "critical")
            )
            return {
                "status": "ok" if level == "normal" else "degraded",
                "level": level,
                "rss_mb": round(mem_mb, 1),
            }
        except ImportError:
            self._log.debug("optional dep not available: psutil")
        try:
            import resource as _resource

            usage = _resource.getrusage(_resource.RUSAGE_SELF)
            mem_mb = usage.ru_maxrss / 1024.0
            level = (
                "normal"
                if mem_mb < 1024
                else ("warning" if mem_mb < 2048 else "critical")
            )
            return {
                "status": "ok" if level == "normal" else "degraded",
                "level": level,
                "rss_mb": round(mem_mb, 1),
            }
        except Exception:
            self._log.debug("memory monitoring via resource module unavailable")
        return {"status": "ok", "note": "memory monitoring unavailable"}

    def _check_disk_space(self) -> Dict[str, Any]:
        try:
            if os.name == "nt":
                import ctypes

                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    self._storage_dir,
                    None,
                    None,
                    ctypes.pointer(free_bytes),
                )
                free_gb = free_bytes.value / (1024**3)
            else:
                usage = os.statvfs(self._storage_dir)
                free_gb = (
                    usage.f_bavail * usage.f_frsize
                ) / (1024**3)
            return {
                "status": "ok" if free_gb > 1.0 else "degraded",
                "free_gb": round(free_gb, 2),
            }
        except Exception:
            self._log.warning("Disk space check failed for %s", self._storage_dir, exc_info=True)
            return {"status": "ok", "note": "disk monitoring unavailable"}

    def _check_pulse_engine(self) -> Dict[str, Any]:
        try:
            stats = self._field.stats()
            running = stats.get("pulse_engine_running", False)
            return {
                "status": "ok" if running else "degraded",
                "running": running,
                "pulse_count": stats.get("pulse_count", 0),
            }
        except Exception as e:
            self._log.warning("Pulse engine check failed: %s", e)
            return {"status": "degraded", "error": str(e)}

    def _check_background_threads(self) -> Dict[str, Any]:
        try:
            alive_daemons = sum(
                1
                for t in threading.enumerate()
                if t.is_alive() and t.daemon
            )
            return {
                "status": "ok" if alive_daemons >= 1 else "degraded",
                "daemon_threads_alive": alive_daemons,
                "total_threads": threading.active_count(),
            }
        except Exception as e:
            self._log.warning("Background threads check failed: %s", e)
            return {"status": "degraded", "error": str(e)}

    def _check_wal_size(self) -> Dict[str, Any]:
        try:
            wal_path = self._persistence._wal_path
            if not os.path.exists(wal_path):
                return {"status": "ok", "wal_size_bytes": 0}
            size = os.path.getsize(wal_path)
            max_size = getattr(
                self._persistence, "MAX_WAL_BYTES", 10 * 1024 * 1024
            )
            return {
                "status": "ok" if size < max_size else "degraded",
                "wal_size_bytes": size,
                "wal_size_mb": round(size / (1024 * 1024), 2),
                "max_bytes": max_size,
            }
        except Exception as e:
            self._log.warning("WAL size check failed: %s", e)
            return {"status": "degraded", "error": str(e)}

    def _check_checkpoint_freshness(self) -> Dict[str, Any]:
        try:
            last_ckpt = getattr(
                self._persistence, "_last_checkpoint_ts", 0
            )
            age = time.time() - last_ckpt
            max_age = getattr(
                self._persistence, "_checkpoint_interval", 300
            ) * 3
            return {
                "status": "ok" if age < max_age else "degraded",
                "last_checkpoint_age_seconds": round(age, 1),
                "max_acceptable_age": max_age,
            }
        except Exception as e:
            self._log.warning("Checkpoint freshness check failed: %s", e)
            return {"status": "degraded", "error": str(e)}

    def _compute_degradation(self, checks: Dict[str, Dict]) -> int:
        degraded_count = sum(
            1
            for c in checks.values()
            if c.get("status") != "ok"
        )
        if degraded_count == 0:
            return 0
        if degraded_count <= 2:
            return 1
        if degraded_count <= 4:
            return 2
        return 3

    def _force_emergency_backup(self) -> str:
        return self.create_scheduled_backup("daily")

    def _recover_stale_checkpoint(self) -> bool:
        try:
            with self._field.field_lock():
                state = self._field.export_full_state()
            self._persistence.checkpoint(state)
            self._recovery_successes["stale_checkpoint"] = (
                self._recovery_successes.get("stale_checkpoint", 0) + 1
            )
            return True
        except Exception:
            self._log.error("Recovery failed for stale_checkpoint", exc_info=True)
            return False

    def _recover_wal_too_large(self) -> bool:
        try:
            with self._field.field_lock():
                state = self._field.export_full_state()
            self._persistence.checkpoint(state)
            self._recovery_successes["wal_too_large"] = (
                self._recovery_successes.get("wal_too_large", 0) + 1
            )
            return True
        except Exception:
            self._log.error("Recovery failed for wal_too_large", exc_info=True)
            return False

    def _recover_pulse_dead(self) -> bool:
        try:
            self._field.start_pulse_engine()
            stats = self._field.stats()
            if stats.get("pulse_engine_running", False):
                self._recovery_successes["pulse_dead"] = (
                    self._recovery_successes.get("pulse_dead", 0) + 1
                )
                return True
            return False
        except Exception:
            self._log.error("Recovery failed for pulse_dead", exc_info=True)
            return False

    def _recover_orphan_nodes(self) -> bool:
        try:
            self._field.run_self_organize()
            self._recovery_successes["orphan_nodes"] = (
                self._recovery_successes.get("orphan_nodes", 0) + 1
            )
            return True
        except Exception:
            self._log.error("Recovery failed for orphan_nodes", exc_info=True)
            return False

    def _recover_memory_high(self) -> bool:
        try:
            with self._field.field_lock():
                self._field.global_decay()
            if self._persistence.is_dirty():
                with self._field.field_lock():
                    state = self._field.export_full_state()
                self._persistence.checkpoint(state)
            self._recovery_successes["memory_high"] = (
                self._recovery_successes.get("memory_high", 0) + 1
            )
            return True
        except Exception:
            self._log.error("Recovery failed for memory_high", exc_info=True)
            return False

    def _recover_disk_low(self) -> bool:
        pruned = False
        for level, schedule in self._backup_schedule.items():
            self._prune_backups(level, max(1, schedule["keep"] // 2))
            pruned = True
        if self._persistence.is_dirty():
            try:
                with self._field.field_lock():
                    state = self._field.export_full_state()
                self._persistence.checkpoint(state)
                pruned = True
            except Exception:
                self._log.warning("Checkpoint during disk-low recovery failed", exc_info=True)
        if pruned:
            self._recovery_successes["disk_low"] = (
                self._recovery_successes.get("disk_low", 0) + 1
            )
        return pruned

    def _prune_backups(self, level: str, keep: int):
        level_entries = [
            e for e in self._backup_index if e.get("level") == level
        ]
        other_entries = [
            e for e in self._backup_index if e.get("level") != level
        ]
        if len(level_entries) <= keep:
            return
        to_remove = level_entries[: len(level_entries) - keep]
        for entry in to_remove:
            try:
                shutil.rmtree(entry["path"], ignore_errors=True)
            except Exception:
                self._log.warning("Failed to remove backup directory %s", entry.get("path", "?"))
            if entry in self._backup_index:
                self._backup_index.remove(entry)

    def _load_backup_index(self) -> List[Dict]:
        if os.path.exists(self._backup_index_path):
            try:
                with open(
                    self._backup_index_path, "r", encoding="utf-8"
                ) as f:
                    return json.load(f)
            except Exception:
                self._log.warning("Failed to load backup index from %s", self._backup_index_path, exc_info=True)
                return []
        return []

    def _save_backup_index(self):
        try:
            with open(
                self._backup_index_path, "w", encoding="utf-8"
            ) as f:
                json.dump(
                    self._backup_index,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            self._log.error("Failed to save backup index to %s", self._backup_index_path, exc_info=True)
