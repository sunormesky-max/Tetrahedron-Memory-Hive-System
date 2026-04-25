from __future__ import annotations

import hashlib
import random
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .pcnn_types import PulseType

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField


class SessionRecord:
    __slots__ = ("role", "content", "timestamp", "memory_id", "metadata")

    def __init__(self, role: str, content: str, memory_id: str = None, metadata: Dict = None):
        self.role = role
        self.content = content
        self.timestamp = time.time()
        self.memory_id = memory_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content[:200],
            "timestamp": self.timestamp,
            "memory_id": self.memory_id[:12] if self.memory_id else None,
            "metadata": self.metadata,
        }


class Session:
    def __init__(self, session_id: str, agent_id: str, metadata: Dict = None):
        self.session_id = session_id
        self.agent_id = agent_id
        self.created_at = time.time()
        self.last_active = time.time()
        self.records: List[SessionRecord] = []
        self._max_records = 500
        self.metadata = metadata or {}
        self.ephemeral_ids: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "record_count": len(self.records),
            "metadata": self.metadata,
        }


class SessionManager:
    """
    Conversation memory management — distinguishes ephemeral (temporary)
    from permanent memories. Ephemeral memories are session-scoped context
    that can be consolidated into permanent memories when the session ends.
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
        self._max_sessions = 50

    def create_session(self, agent_id: str, metadata: Dict = None) -> str:
        session_id = hashlib.sha256(f"{agent_id}:{time.time()}:{random.random()}".encode()).hexdigest()[:16]
        session = Session(session_id, agent_id, metadata)

        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_active)
                self.consolidate_session(oldest_id)
                del self._sessions[oldest_id]

            self._sessions[session_id] = session

        return session_id

    def add_to_session(self, session_id: str, role: str, content: str,
                       metadata: Dict = None) -> Dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": "session not found"}

        field = self._field
        memory_id = None
        if content and len(content.strip()) > 0:
            ephemeral_labels = ["__ephemeral__", f"__session_{session_id[:8]}__"]
            if metadata and metadata.get("labels"):
                ephemeral_labels.extend(metadata["labels"])

            with field._lock:
                memory_id = field.store(
                    content=content,
                    labels=ephemeral_labels,
                    weight=0.5,
                    metadata={"session_id": session_id, "role": role, "ephemeral": True},
                )

        record = SessionRecord(role, content, memory_id, metadata)

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": "session not found"}
            session.records.append(record)
            if len(session.records) > session._max_records:
                session.records = session.records[-session._max_records:]
            session.last_active = time.time()
            if memory_id:
                session.ephemeral_ids.append(memory_id)
                if len(session.ephemeral_ids) > session._max_records:
                    session.ephemeral_ids = session.ephemeral_ids[-session._max_records:]

        return {"added": True, "memory_id": memory_id[:12] if memory_id else None}

    def recall_session(self, session_id: str, n: int = 20) -> Dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": "session not found"}
            records = session.records[-n:]
            return {
                "session_id": session_id,
                "records": [r.to_dict() for r in records],
                "total_records": len(session.records),
                "agent_id": session.agent_id,
            }

    def consolidate_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": "session not found"}

        field = self._field
        consolidated = 0
        promoted = 0

        with field._lock:
            for mid in session.ephemeral_ids:
                node = field._nodes.get(mid)
                if node is None:
                    continue

                if node.weight >= 0.8:
                    node.labels = [l for l in node.labels if not l.startswith("__session_") and l != "__ephemeral__"]
                    node.labels.append("__consolidated__")
                    node.weight = min(10.0, node.weight + 0.3)
                    if "ephemeral" in node.metadata:
                        del node.metadata["ephemeral"]
                    field._emit_pulse(mid, strength=0.5, pulse_type=PulseType.REINFORCING)
                    promoted += 1
                else:
                    node.base_activation = max(node.base_activation, 0.02)
                    consolidated += 1

        with self._lock:
            session.metadata["consolidated_at"] = time.time()
            session.metadata["promoted"] = promoted
            session.metadata["soft_kept"] = consolidated

        return {
            "session_id": session_id,
            "total_ephemeral": len(session.ephemeral_ids),
            "promoted_to_permanent": promoted,
            "soft_kept": consolidated,
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [s.to_dict() for s in sorted(self._sessions.values(), key=lambda s: -s.last_active)]

    def expire_sessions(self, max_age: int = 3600) -> Dict[str, Any]:
        now = time.time()
        expired = []
        with self._lock:
            to_remove = []
            for sid, session in self._sessions.items():
                if now - session.last_active > max_age:
                    self.consolidate_session(sid)
                    to_remove.append(sid)
                    expired.append(sid)
            for sid in to_remove:
                del self._sessions[sid]

        return {"expired_sessions": len(expired), "session_ids": [s[:12] for s in expired]}

    def session_cleanup(self, max_age: int = 3600) -> Dict[str, Any]:
        return self.expire_sessions(max_age)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return {
                **session.to_dict(),
                "recent_records": [r.to_dict() for r in session.records[-10:]],
            }
