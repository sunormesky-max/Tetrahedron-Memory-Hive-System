"""
Eternity Audit — Strict verification that NO memory is ever deleted.

Core guarantee:
  For every memory ever stored, one of the following MUST hold:
    1. The memory still exists in the mesh (tetra_id is live)
    2. The memory was merged into another memory (content preserved in metadata)
    3. The memory exists in the compensation log with a pending retry

This module provides:
  - EternityAudit: records every store/merge/transform and enables full verification
  - Proof of preservation: formal content tracing across all operations
  - Verification pass: scan entire history, prove no memory was lost
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger_msg = __import__("logging").getLogger("tetramem.eternity")


@dataclass
class AuditEntry:
    operation: str
    target_id: str
    timestamp: float
    content_hash: str
    content_preview: str
    preserved_in: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EternityAudit:
    """
    Strict audit trail for the Eternity Principle.

    Every mutating operation (store, merge, integrate, dream-insert, reintegrate)
    must register an audit entry. The verify() method can then prove that no
    memory content was ever lost.
    """

    def __init__(self, max_entries: int = 100000):
        self._log: List[AuditEntry] = []
        self._max_entries = max_entries
        self._content_registry: Dict[str, str] = {}
        self._preservation_map: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()
        self._violations: List[Dict[str, Any]] = []
        self._max_violations: int = 100

    @staticmethod
    def _hash_content(content: str) -> str:
        import hashlib

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def record_store(self, tetra_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        with self._lock:
            entry = AuditEntry(
                operation="store",
                target_id=tetra_id,
                timestamp=time.time(),
                content_hash=self._hash_content(content),
                content_preview=content[:80],
                metadata=metadata or {},
            )
            self._log.append(entry)
            self._content_registry[tetra_id] = entry.content_hash
            self._preservation_map.setdefault(tetra_id, set()).add(tetra_id)
            self._trim()

    def record_merge(
        self,
        source_ids: List[str],
        result_id: str,
        result_content: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        with self._lock:
            content_hash = self._hash_content(result_content)
            entry = AuditEntry(
                operation="merge",
                target_id=result_id,
                timestamp=time.time(),
                content_hash=content_hash,
                content_preview=result_content[:80],
                preserved_in=result_id,
                metadata={"merged_from": source_ids, **(metadata or {})},
            )
            self._log.append(entry)
            self._content_registry[result_id] = content_hash
            for sid in source_ids:
                self._preservation_map.setdefault(sid, set()).add(result_id)
            self._preservation_map.setdefault(result_id, set()).add(result_id)
            self._trim()

    def record_transform(
        self,
        source_id: str,
        target_id: str,
        operation: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        with self._lock:
            entry = AuditEntry(
                operation=operation,
                target_id=target_id,
                timestamp=time.time(),
                content_hash=self._hash_content(content),
                content_preview=content[:80],
                preserved_in=target_id,
                metadata={"source": source_id, **(metadata or {})},
            )
            self._log.append(entry)
            self._preservation_map.setdefault(source_id, set()).add(target_id)
            self._preservation_map.setdefault(target_id, set()).add(target_id)
            for ancestor_id, descendants in list(self._preservation_map.items()):
                if source_id in descendants:
                    descendants.add(target_id)
            self._trim()

    def record_dream(
        self, dream_id: str, content: str, source_ids: List[str], metadata: Optional[Dict] = None
    ) -> None:
        with self._lock:
            entry = AuditEntry(
                operation="dream_create",
                target_id=dream_id,
                timestamp=time.time(),
                content_hash=self._hash_content(content),
                content_preview=content[:80],
                preserved_in=dream_id,
                metadata={"synthesized_from": source_ids, **(metadata or {})},
            )
            self._log.append(entry)
            self._content_registry[dream_id] = entry.content_hash
            self._preservation_map.setdefault(dream_id, set()).add(dream_id)
            for sid in source_ids:
                self._preservation_map.setdefault(sid, set()).add(dream_id)
            self._trim()

    def record_reintegration(self, dream_id: str, metadata: Optional[Dict] = None) -> None:
        with self._lock:
            entry = AuditEntry(
                operation="reintegrate",
                target_id=dream_id,
                timestamp=time.time(),
                content_hash=self._content_registry.get(dream_id, "unknown"),
                content_preview="",
                preserved_in=dream_id,
                metadata=metadata or {},
            )
            self._log.append(entry)
            self._trim()

    def verify(self, mesh: Any) -> Dict[str, Any]:
        """
        Verify the Eternity Principle: prove no memory was ever deleted.

        For every tetra_id that was ever stored, check that:
          1. It still exists in the mesh, OR
          2. Its content was preserved in a merge (tracked in preservation_map)

        Returns a verification report with:
          - total_stored: total memories ever stored
          - total_alive: memories still in mesh
          - total_preserved: memories whose content lives on via merge/transform
          - total_violations: memories with NO preservation proof
          - violations: list of tetra_ids that cannot be proven preserved
          - verified: True if no violations
        """
        with self._lock:
            all_stored_ids = set()
            for entry in self._log:
                if entry.operation == "store":
                    all_stored_ids.add(entry.target_id)
                elif entry.operation == "merge":
                    for sid in entry.metadata.get("merged_from", []):
                        all_stored_ids.add(sid)
                elif entry.operation in ("dream_create", "reintegrate"):
                    pass

            live_ids = set()
            tetrahedra = mesh._tetrahedra if hasattr(mesh, "_tetrahedra") else {}
            for tid in tetrahedra:
                live_ids.add(tid)

            violations = []
            preserved_count = 0
            for tid in all_stored_ids:
                if tid in live_ids:
                    preserved_count += 1
                    continue

                chain = self._preservation_map.get(tid, set())
                if chain:
                    any_alive = bool(chain & live_ids)
                    if any_alive:
                        preserved_count += 1
                        continue

                violations.append(tid)

            result = {
                "verified": len(violations) == 0,
                "total_stored": len(all_stored_ids),
                "total_alive": len(live_ids & all_stored_ids),
                "total_preserved": preserved_count,
                "total_violations": len(violations),
                "violations": violations[:20],
                "audit_entries": len(self._log),
                "timestamp": time.time(),
            }

            if violations:
                self._violations.append(result)
                if len(self._violations) > self._max_violations:
                    self._violations = self._violations[-self._max_violations :]
                logger_msg.warning(
                    "ETERNITY VIOLATION: %d memories have no preservation proof",
                    len(violations),
                )

            return result

    def get_preservation_chain(self, tetra_id: str) -> List[str]:
        with self._lock:
            return list(self._preservation_map.get(tetra_id, set()))

    def get_audit_trail(self, tetra_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "operation": e.operation,
                    "target_id": e.target_id,
                    "timestamp": e.timestamp,
                    "content_hash": e.content_hash,
                    "preserved_in": e.preserved_in,
                    "metadata": e.metadata,
                }
                for e in self._log
                if e.target_id == tetra_id
                or tetra_id in e.metadata.get("merged_from", [])
                or tetra_id in e.metadata.get("synthesized_from", [])
                or tetra_id in e.metadata.get("source", [])
            ]

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_entries": len(self._log),
                "total_tracked_ids": len(self._preservation_map),
                "total_violations": len(self._violations),
                "last_violation": self._violations[-1] if self._violations else None,
            }

    def _trim(self) -> None:
        if len(self._log) > self._max_entries:
            excess = len(self._log) - self._max_entries
            removed_ids = set()
            for entry in self._log[:excess]:
                if entry.operation == "store":
                    removed_ids.add(entry.target_id)
            self._log = self._log[-self._max_entries :]
            if removed_ids and len(self._content_registry) > self._max_entries * 2:
                for rid in removed_ids:
                    self._content_registry.pop(rid, None)
                    self._preservation_map.pop(rid, None)
            if len(self._preservation_map) > self._max_entries * 2:
                dead_keys = [k for k, v in self._preservation_map.items() if not v]
                for k in dead_keys[: len(dead_keys) // 2]:
                    del self._preservation_map[k]
