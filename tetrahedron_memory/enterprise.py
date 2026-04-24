import copy
import time
import json
import os
import threading
from typing import Any, Dict, List, Optional


class VersionControl:
    def __init__(self, max_versions: int = 10):
        self._versions: Dict[str, List[Dict]] = {}
        self._max = max_versions
        self._lock = threading.Lock()

    def record_version(self, node_id: str, content, weight: float,
                       labels=None, metadata=None):
        version = {
            "ts": time.time(),
            "content": content,
            "weight": weight,
            "labels": list(labels) if labels else [],
            "metadata": dict(metadata) if metadata else {},
        }
        with self._lock:
            if node_id not in self._versions:
                self._versions[node_id] = []
            self._versions[node_id].append(version)
            if len(self._versions[node_id]) > self._max:
                self._versions[node_id] = self._versions[node_id][-self._max:]

    def get_versions(self, node_id: str) -> List[Dict]:
        with self._lock:
            return list(self._versions.get(node_id, []))

    def get_latest(self, node_id: str) -> Optional[Dict]:
        with self._lock:
            versions = self._versions.get(node_id, [])
            return versions[-1] if versions else None


class QuotaManager:
    def __init__(self, default_quota: int = 100000):
        self._quotas: Dict[str, int] = {}
        self._usage: Dict[str, int] = {}
        self._default = default_quota
        self._lock = threading.Lock()

    def check_allowed(self, tenant_id: str) -> bool:
        with self._lock:
            quota = self._quotas.get(tenant_id, self._default)
            usage = self._usage.get(tenant_id, 0)
            return usage < quota

    def increment(self, tenant_id: str):
        with self._lock:
            self._usage[tenant_id] = self._usage.get(tenant_id, 0) + 1

    def decrement(self, tenant_id: str):
        with self._lock:
            if tenant_id in self._usage:
                self._usage[tenant_id] = max(0, self._usage[tenant_id] - 1)

    def set_quota(self, tenant_id: str, quota: int):
        with self._lock:
            self._quotas[tenant_id] = quota

    def get_usage(self, tenant_id: str) -> Dict:
        with self._lock:
            return {
                "tenant_id": tenant_id,
                "usage": self._usage.get(tenant_id, 0),
                "quota": self._quotas.get(tenant_id, self._default),
            }


class TTLManager:
    def __init__(self):
        self._ttls: Dict[str, float] = {}
        self._lock = threading.Lock()

    def set_ttl(self, node_id: str, ttl_seconds: float):
        with self._lock:
            if ttl_seconds > 0:
                self._ttls[node_id] = time.time() + ttl_seconds
            else:
                self._ttls.pop(node_id, None)

    def get_expired(self) -> List[str]:
        now = time.time()
        with self._lock:
            return [nid for nid, exp in self._ttls.items() if now >= exp]

    def remove(self, node_id: str):
        with self._lock:
            self._ttls.pop(node_id, None)


class BackupManager:
    def __init__(self, storage_dir: str):
        self._storage_dir = os.path.join(storage_dir, "backups")
        os.makedirs(self._storage_dir, exist_ok=True)
        self._max_backups = 10

    def create_backup(self, state: dict, label: str = "manual") -> str:
        ts = int(time.time() * 1000)
        filename = f"backup_{label}_{ts}.json"
        path = os.path.join(self._storage_dir, filename)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, default=str)
            os.replace(tmp, path)
            self._rotate()
            return filename
        except Exception as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

    def restore_backup(self, backup_id: str) -> Optional[dict]:
        path = os.path.join(self._storage_dir, backup_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def list_backups(self) -> List[Dict]:
        backups = []
        for f in os.listdir(self._storage_dir):
            if f.startswith("backup_") and f.endswith(".json"):
                path = os.path.join(self._storage_dir, f)
                try:
                    stat = os.stat(path)
                    backups.append({
                        "id": f,
                        "size": stat.st_size,
                        "created": stat.st_mtime,
                    })
                except OSError:
                    pass
        backups.sort(key=lambda b: b["created"], reverse=True)
        return backups

    def delete_backup(self, backup_id: str) -> bool:
        path = os.path.join(self._storage_dir, backup_id)
        if os.path.exists(path):
            try:
                os.remove(path)
                return True
            except OSError:
                pass
        return False

    def _rotate(self):
        backups = []
        for f in os.listdir(self._storage_dir):
            if f.startswith("backup_") and f.endswith(".json"):
                path = os.path.join(self._storage_dir, f)
                try:
                    backups.append((os.path.getmtime(path), path))
                except OSError:
                    pass
        if len(backups) <= self._max_backups:
            return
        backups.sort()
        for _, path in backups[:-self._max_backups]:
            try:
                os.remove(path)
            except OSError:
                pass
