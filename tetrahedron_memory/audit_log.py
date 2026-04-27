import time
import logging
import threading
from typing import Any, Dict, List, Optional
from contextlib import contextmanager


class AuditLog:
    def __init__(self, max_entries: int = 10000):
        self._entries: List[Dict] = []
        self._max = max_entries
        self._lock = threading.Lock()

    def log(self, tenant_id: str, action: str, data: dict = None, node_id: str = None):
        entry = {
            "ts": time.time(),
            "tenant_id": tenant_id,
            "action": action,
            "data": data or {},
        }
        if node_id:
            entry["node_id"] = node_id
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max:
                self._entries = self._entries[-self._max:]

    def query(self, tenant_id: str = None, action: str = None,
              since: float = None, limit: int = 100) -> List[Dict]:
        with self._lock:
            results = list(self._entries)
        if tenant_id:
            results = [e for e in results if e.get("tenant_id") == tenant_id]
        if action:
            results = [e for e in results if e.get("action") == action]
        if since:
            results = [e for e in results if e.get("ts", 0) >= since]
        return results[-limit:]

    def get_stats(self) -> Dict:
        with self._lock:
            total = len(self._entries)
        return {"total_entries": total, "max_entries": self._max}
