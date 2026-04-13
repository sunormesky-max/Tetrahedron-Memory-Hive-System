import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("tetramem.consistency")


@dataclass
class VersionedNode:
    node_id: str
    version: int
    bucket_id: str
    timestamp: float
    checksum: str
    operation: str = "store"


@dataclass
class ConflictRecord:
    node_id: str
    local_version: VersionedNode
    remote_version: VersionedNode
    detected_at: float
    resolution: str = "pending"
    resolved_at: float = 0.0


class VectorClock:
    def __init__(self, bucket_ids: List[str]):
        self._clock: Dict[str, int] = {bid: 0 for bid in bucket_ids}
        self._lock = threading.RLock()

    def increment(self, bucket_id: str) -> int:
        with self._lock:
            self._clock[bucket_id] = self._clock.get(bucket_id, 0) + 1
            return self._clock[bucket_id]

    def get(self, bucket_id: str) -> int:
        with self._lock:
            return self._clock.get(bucket_id, 0)

    def merge(self, other: "VectorClock") -> None:
        with self._lock:
            other_snap = other.snapshot()
            for bid, val in other_snap.items():
                self._clock[bid] = max(self._clock.get(bid, 0), val)

    def happens_before(self, other: "VectorClock") -> bool:
        other_snap = other.snapshot()
        with self._lock:
            at_least_one_less = False
            for bid in self._clock:
                mine = self._clock[bid]
                theirs = other_snap.get(bid, 0)
                if mine > theirs:
                    return False
                if mine < theirs:
                    at_least_one_less = True
            return at_least_one_less

    def is_concurrent(self, other: "VectorClock") -> bool:
        return not self.happens_before(other) and not other.happens_before(self)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._clock)

    def add_bucket(self, bucket_id: str) -> None:
        with self._lock:
            if bucket_id not in self._clock:
                self._clock[bucket_id] = 0


class CompensationLog:
    def __init__(self, max_entries: int = 1000):
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._next_id = 0

    def record(self, operation: str, bucket_id: str, params: Dict[str, Any], error: str) -> str:
        with self._lock:
            entry_id = f"comp_{self._next_id}"
            self._next_id += 1
            self._entries.append(
                {
                    "id": entry_id,
                    "operation": operation,
                    "bucket_id": bucket_id,
                    "params": params,
                    "error": error,
                    "timestamp": time.time(),
                    "resolved": False,
                    "retries": 0,
                }
            )
            if len(self._entries) > self._max_entries:
                self._entries = [e for e in self._entries if not e["resolved"]][
                    -self._max_entries :
                ]
            return entry_id

    def get_pending(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [e for e in self._entries if not e["resolved"]]

    def mark_resolved(self, entry_id: str) -> None:
        with self._lock:
            for e in self._entries:
                if e["id"] == entry_id:
                    e["resolved"] = True
                    break

    def retry_all(self, executor: Callable) -> List[Dict[str, Any]]:
        results = []
        with self._lock:
            pending = [e for e in self._entries if not e["resolved"]]
        for entry in pending:
            try:
                executor(entry["operation"], entry["params"])
                self.mark_resolved(entry["id"])
                results.append({"id": entry["id"], "status": "resolved"})
            except Exception as exc:
                with self._lock:
                    for e in self._entries:
                        if e["id"] == entry["id"]:
                            e["retries"] += 1
                            e["last_error"] = str(exc)
                            break
                results.append({"id": entry["id"], "status": "failed", "error": str(exc)})
        return results

    def clear_resolved(self) -> int:
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if not e["resolved"]]
            return before - len(self._entries)


class ConsistencyManager:
    def __init__(self, bucket_ids: List[str]):
        self._vector_clock = VectorClock(bucket_ids)
        self._compensation = CompensationLog()
        self._bucket_locks: Dict[str, threading.RLock] = {
            bid: threading.RLock() for bid in bucket_ids
        }
        self._version_store: Dict[str, List[VersionedNode]] = {}
        self._lock = threading.RLock()
        self._conflict_history: List[ConflictRecord] = []
        self._auto_resolve_enabled = True
        self._max_conflict_history = 500
        self._max_versions_per_node = 20

    def acquire_lock(self, bucket_ids: List[str], timeout: float = 5.0) -> bool:
        sorted_ids = sorted(set(bucket_ids))
        acquired: List[str] = []
        for bid in sorted_ids:
            lock = self._bucket_locks.get(bid)
            if lock is None:
                lock = threading.RLock()
                self._bucket_locks[bid] = lock
            success = lock.acquire(timeout=timeout)
            if not success:
                for rbid in reversed(acquired):
                    self._bucket_locks[rbid].release()
                return False
            acquired.append(bid)
        return True

    def release_lock(self, bucket_ids: List[str]) -> None:
        sorted_ids = sorted(set(bucket_ids))
        for bid in reversed(sorted_ids):
            lock = self._bucket_locks.get(bid)
            if lock is not None:
                lock.release()

    def record_version(
        self,
        node_id: str,
        bucket_id: str,
        content: str,
        operation: str = "store",
    ) -> VersionedNode:
        with self._lock:
            existing_list = self._version_store.get(node_id, [])
            version = (existing_list[-1].version + 1) if existing_list else 1
            checksum = hashlib.md5(content.encode()).hexdigest()
            vn = VersionedNode(
                node_id=node_id,
                version=version,
                bucket_id=bucket_id,
                timestamp=time.time(),
                checksum=checksum,
                operation=operation,
            )
            self._version_store.setdefault(node_id, []).append(vn)
            if len(self._version_store[node_id]) > self._max_versions_per_node:
                self._version_store[node_id] = self._version_store[node_id][
                    -self._max_versions_per_node :
                ]
            self._vector_clock.increment(bucket_id)

            if operation in ("store", "dream", "integrate"):
                self._check_and_resolve_conflict(node_id)

            return vn

    def check_version(self, node_id: str, expected_version: int) -> bool:
        with self._lock:
            versions = self._version_store.get(node_id, [])
            return any(vn.version == expected_version for vn in versions)

    def validate_before_write(
        self,
        node_id: str,
        bucket_id: str,
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            versions = self._version_store.get(node_id, [])
            if not versions:
                return {"valid": True, "reason": "new_node"}

            latest = max(versions, key=lambda v: v.version)

            if expected_version is not None and latest.version != expected_version:
                conflict = ConflictRecord(
                    node_id=node_id,
                    local_version=latest,
                    remote_version=VersionedNode(
                        node_id=node_id,
                        version=expected_version,
                        bucket_id=bucket_id,
                        timestamp=time.time(),
                        checksum="",
                        operation="conflict_check",
                    ),
                    detected_at=time.time(),
                )
                self._record_conflict(conflict)

                if self._auto_resolve_enabled:
                    resolution = self._auto_resolve_conflict(conflict)
                    return {
                        "valid": resolution["proceed"],
                        "reason": resolution["reason"],
                        "conflict": True,
                        "latest_version": latest.version,
                        "resolution": resolution["action"],
                    }

                return {
                    "valid": False,
                    "reason": "version_mismatch",
                    "conflict": True,
                    "expected": expected_version,
                    "actual": latest.version,
                }

            return {"valid": True, "reason": "version_match", "latest_version": latest.version}

    def detect_conflicts(self) -> List[Tuple[VersionedNode, VersionedNode]]:
        with self._lock:
            conflicts = []
            for nid, versions in self._version_store.items():
                bucket_map: Dict[str, VersionedNode] = {}
                for vn in versions:
                    if vn.bucket_id in bucket_map:
                        if vn.version > bucket_map[vn.bucket_id].version:
                            bucket_map[vn.bucket_id] = vn
                    else:
                        bucket_map[vn.bucket_id] = vn
                if len(bucket_map) > 1:
                    by_time = sorted(bucket_map.values(), key=lambda v: v.timestamp)
                    conflicts.append((by_time[0], by_time[-1]))
            return conflicts

    def read_repair(
        self,
        node_id: str,
        source_bucket: str,
        target_buckets: List[str],
    ) -> Dict[str, Any]:
        with self._lock:
            versions = self._version_store.get(node_id, [])
            if not versions:
                return {"repaired": False, "reason": "no_versions"}

            latest = max(versions, key=lambda v: v.version)
            repaired = []
            if latest.bucket_id == source_bucket:
                for tb in target_buckets:
                    self._vector_clock.increment(tb)
                    repaired.append(tb)

            return {
                "repaired": len(repaired) > 0,
                "source_bucket": source_bucket,
                "repaired_buckets": repaired,
                "latest_version": latest.version,
            }

    def read_repair_multi(
        self,
        bucket_id: str,
        known_buckets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            stale = self.get_staleness(bucket_id)
            if not stale:
                return {"repaired": 0, "stale_found": 0}

            targets = known_buckets or list({s["source_bucket"] for s in stale})
            total_repaired = 0
            for entry in stale:
                result = self.read_repair(
                    node_id=entry["node_id"],
                    source_bucket=entry["source_bucket"],
                    target_buckets=[bucket_id],
                )
                if result["repaired"]:
                    total_repaired += 1

            return {
                "repaired": total_repaired,
                "stale_found": len(stale),
                "bucket_id": bucket_id,
            }

    def get_staleness(self, bucket_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            clock_snap = self._vector_clock.snapshot()
            bucket_clock = clock_snap.get(bucket_id, 0)
            stale = []
            for nid, versions in self._version_store.items():
                latest = max(versions, key=lambda v: v.version)
                source_clock = clock_snap.get(latest.bucket_id, 0)
                if latest.bucket_id != bucket_id and source_clock > bucket_clock:
                    stale.append(
                        {
                            "node_id": latest.node_id,
                            "source_bucket": latest.bucket_id,
                            "source_version": latest.version,
                            "source_clock": source_clock,
                        }
                    )
            return stale

    def add_bucket(self, bucket_id: str) -> None:
        with self._lock:
            self._bucket_locks.setdefault(bucket_id, threading.RLock())
            self._vector_clock.add_bucket(bucket_id)

    def get_conflict_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "node_id": c.node_id,
                    "local_version": c.local_version.version,
                    "remote_version": c.remote_version.version,
                    "resolution": c.resolution,
                    "detected_at": c.detected_at,
                }
                for c in self._conflict_history[-limit:]
            ]

    def compensate_operation(
        self,
        operation: str,
        bucket_id: str,
        params: Dict[str, Any],
        error: str,
    ) -> str:
        entry_id = self._compensation.record(operation, bucket_id, params, error)
        logger.warning(
            "Compensation recorded: %s op=%s bucket=%s error=%s",
            entry_id,
            operation,
            bucket_id,
            error,
        )
        return entry_id

    def retry_pending_compensations(self, executor: Callable) -> List[Dict[str, Any]]:
        results = self._compensation.retry_all(executor)
        resolved = sum(1 for r in results if r["status"] == "resolved")
        logger.info("Compensation retry: %d/%d resolved", resolved, len(results))
        return results

    def get_health(self) -> Dict[str, Any]:
        with self._lock:
            conflicts = self.detect_conflicts()
            pending = self._compensation.get_pending()
            return {
                "enabled": True,
                "total_versioned_nodes": len(self._version_store),
                "active_conflicts": len(conflicts),
                "pending_compensations": len(pending),
                "conflict_history_size": len(self._conflict_history),
                "auto_resolve_enabled": self._auto_resolve_enabled,
                "vector_clock": self._vector_clock.snapshot(),
            }

    def _check_and_resolve_conflict(self, node_id: str) -> None:
        versions = self._version_store.get(node_id, [])
        if len(versions) < 2:
            return

        bucket_map: Dict[str, VersionedNode] = {}
        for vn in versions:
            if vn.bucket_id not in bucket_map or vn.version > bucket_map[vn.bucket_id].version:
                bucket_map[vn.bucket_id] = vn

        if len(bucket_map) < 2:
            return

        sorted_versions = sorted(bucket_map.values(), key=lambda v: (v.version, v.timestamp))
        conflict = ConflictRecord(
            node_id=node_id,
            local_version=sorted_versions[-1],
            remote_version=sorted_versions[-2],
            detected_at=time.time(),
        )
        self._record_conflict(conflict)

        if self._auto_resolve_enabled:
            self._auto_resolve_conflict(conflict)

    def _record_conflict(self, conflict: ConflictRecord) -> None:
        self._conflict_history.append(conflict)
        if len(self._conflict_history) > self._max_conflict_history:
            self._conflict_history = self._conflict_history[-self._max_conflict_history :]
        logger.info(
            "Conflict detected: node=%s local_v=%d remote_v=%d",
            conflict.node_id,
            conflict.local_version.version,
            conflict.remote_version.version,
        )

    def _auto_resolve_conflict(self, conflict: ConflictRecord) -> Dict[str, Any]:
        local = conflict.local_version
        remote = conflict.remote_version

        if local.version > remote.version:
            action = "keep_latest"
            conflict.resolution = "resolved_latest_wins"
            conflict.resolved_at = time.time()
            return {"proceed": True, "reason": "local_is_latest", "action": action}

        if local.version == remote.version:
            if local.timestamp >= remote.timestamp:
                action = "keep_latest_timestamp"
                conflict.resolution = "resolved_timestamp_wins"
                conflict.resolved_at = time.time()
                return {"proceed": True, "reason": "local_is_newer", "action": action}
            else:
                conflict.resolution = "resolved_remote_newer"
                conflict.resolved_at = time.time()
                return {
                    "proceed": False,
                    "reason": "remote_is_newer",
                    "action": "retry_with_remote",
                }

        conflict.resolution = "resolved_version_wins"
        conflict.resolved_at = time.time()
        return {
            "proceed": False,
            "reason": "remote_has_higher_version",
            "action": "fetch_and_merge",
        }

    @property
    def compensation_log(self) -> CompensationLog:
        return self._compensation

    @property
    def vector_clock(self) -> VectorClock:
        return self._vector_clock
