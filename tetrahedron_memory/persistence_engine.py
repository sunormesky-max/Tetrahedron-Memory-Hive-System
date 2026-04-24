import json
import os
import time
import threading
import gzip
import io
from typing import Any, Dict, List, Optional


class PersistenceEngine:
    MAX_WAL_BYTES = 10 * 1024 * 1024

    def __init__(self, storage_dir: str, checkpoint_interval: float = 300):
        self._storage_dir = storage_dir
        self._checkpoint_interval = checkpoint_interval
        os.makedirs(storage_dir, exist_ok=True)
        self._wal_path = os.path.join(storage_dir, "wal.log")
        self._wal_fd = None
        self._closed = False
        self.dirty_ops = 0
        self._last_checkpoint_ts = 0.0
        self._lock = threading.Lock()
        self._max_checkpoints = 3

    def open(self):
        self._wal_fd = open(self._wal_path, "a", encoding="utf-8")
        self._closed = False

    def close(self):
        with self._lock:
            if self._wal_fd and not self._closed:
                try:
                    self._wal_fd.close()
                except Exception:
                    pass
            self._closed = True
            self._wal_fd = None

    def log_operation(self, op: str, data: Optional[dict] = None):
        with self._lock:
            if self._closed or not self._wal_fd:
                return
            entry = {"op": op, "ts": time.time(), "data": data or {}}
            try:
                line = json.dumps(entry, ensure_ascii=False, default=str)
                self._wal_fd.write(line + "\n")
                self._wal_fd.flush()
                self.dirty_ops += 1
            except Exception:
                pass

    def is_dirty(self) -> bool:
        return self.dirty_ops > 0

    def should_checkpoint(self) -> bool:
        if self.dirty_ops >= 100:
            return True
        if self._last_checkpoint_ts > 0 and time.time() - self._last_checkpoint_ts >= self._checkpoint_interval:
            return self.dirty_ops > 0
        return False

    def checkpoint(self, state: dict):
        with self._lock:
            ts = int(time.time() * 1000)
            use_gzip = len(state.get("nodes", {})) > 500
            path = None
            tmp_path = None
            try:
                if use_gzip:
                    path = os.path.join(self._storage_dir, f"checkpoint_{ts}.json.gz")
                    tmp_path = path + ".tmp"
                    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                        json.dump(state, f, ensure_ascii=False, default=str)
                else:
                    path = os.path.join(self._storage_dir, f"checkpoint_{ts}.json")
                    tmp_path = path + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(state, f, ensure_ascii=False, default=str)
                os.replace(tmp_path, path)
                self.dirty_ops = 0
                self._last_checkpoint_ts = time.time()
                if self._wal_fd and not self._closed:
                    self._wal_fd.close()
                    with open(self._wal_path, "w", encoding="utf-8") as f:
                        f.write("")
                    self._wal_fd = open(self._wal_path, "a", encoding="utf-8")
                self._rotate_checkpoints()
            except Exception as e:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    def recover(self) -> Optional[dict]:
        latest = self._find_latest_checkpoint()
        if latest:
            state = self._load_checkpoint(latest)
            if state is not None:
                self._replay_wal(state)
                return state
        wal_state = self._replay_wal_only()
        return wal_state

    def _find_latest_checkpoint(self) -> Optional[str]:
        checkpoints = []
        for f in os.listdir(self._storage_dir):
            if f.startswith("checkpoint_") and (f.endswith(".json") or f.endswith(".json.gz")):
                checkpoints.append(f)
        if not checkpoints:
            return None
        checkpoints.sort(reverse=True)
        return os.path.join(self._storage_dir, checkpoints[0])

    def _load_checkpoint(self, path: str) -> Optional[dict]:
        try:
            if path.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return None

    def _replay_wal(self, state: dict):
        if not os.path.exists(self._wal_path):
            return
        try:
            with open(self._wal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        op = entry.get("op", "")
                        data = entry.get("data", {})
                        if op == "store":
                            nodes = state.get("nodes", {})
                            nid = data.get("id")
                            if nid and nid not in nodes:
                                nodes[nid] = data
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    def _replay_wal_only(self) -> Optional[dict]:
        if not os.path.exists(self._wal_path):
            return None
        state = {"nodes": {}, "hebbian_edges": [], "crystals": [], "config": {}}
        self._replay_wal(state)
        return state if state["nodes"] else None

    def _rotate_checkpoints(self):
        checkpoints = []
        for f in os.listdir(self._storage_dir):
            if f.startswith("checkpoint_") and (f.endswith(".json") or f.endswith(".json.gz")):
                checkpoints.append(os.path.join(self._storage_dir, f))
        if len(checkpoints) <= self._max_checkpoints:
            return
        checkpoints.sort(key=lambda p: os.path.getmtime(p))
        for old in checkpoints[:-self._max_checkpoints]:
            try:
                os.remove(old)
            except OSError:
                pass

    def get_config(self) -> dict:
        config_path = os.path.join(self._storage_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"resolution": 5, "spacing": 1.0}

    def save_config(self, config: dict):
        config_path = os.path.join(self._storage_dir, "config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f)
        except Exception:
            pass
