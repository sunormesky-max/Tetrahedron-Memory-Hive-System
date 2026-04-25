"""
Distributed Architecture + Memory Offloading + Multi-Instance for TetraMem-XL.

Pure Python — no external services or engines.
Uses gossip protocol for peer discovery and HTTP for inter-instance communication.
"""

import gzip
import json
import logging
import math
import os
import threading
import time
import urllib.request
import urllib.error
from concurrent import futures as _futures
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("tetramem.distributed")


class MeshPartition:
    """A partition of the BCC lattice that can be owned by one instance."""

    def __init__(self, partition_id: str, bounds: tuple, node_ids: List[str]):
        self.id = partition_id
        self.bounds = bounds
        self.node_ids: Set[str] = set(node_ids)
        self.owner_instance: Optional[str] = None
        self.access_mode: str = "local"

    def contains_position(self, pos) -> bool:
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = self.bounds
        arr = np.asarray(pos, dtype=np.float32)
        return bool(
            xmin <= arr[0] <= xmax
            and ymin <= arr[1] <= ymax
            and zmin <= arr[2] <= zmax
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bounds": self.bounds,
            "node_ids": sorted(self.node_ids),
            "owner_instance": self.owner_instance,
            "access_mode": self.access_mode,
        }

    @classmethod
    def from_dict(cls, d) -> "MeshPartition":
        p = cls(d["id"], tuple(tuple(b) for b in d["bounds"]), d.get("node_ids", []))
        p.owner_instance = d.get("owner_instance")
        p.access_mode = d.get("access_mode", "local")
        return p


class MemoryOffloader:
    """
    Memory offloading mechanism for large-scale systems.
    Moves cold (low activation) memories to compressed disk storage.
    They remain in the lattice topology but are 'hibernated'.
    """

    def __init__(self, storage_dir: str, hot_ratio: float = 0.7):
        self._storage_dir = storage_dir
        self._hot_ratio = hot_ratio
        self._cold_store: Dict[str, Dict] = {}
        self._access_log: Dict[str, float] = {}
        self._reload_count: int = 0
        self._total_offloaded: int = 0
        self._bytes_saved: int = 0
        self._lock = threading.Lock()
        os.makedirs(storage_dir, exist_ok=True)

    def should_offload(self, node) -> bool:
        now = time.time()
        last_access = self._access_log.get(node.id, node.creation_time or 0)
        age = now - last_access
        if not node.is_occupied:
            return False
        if node.weight >= 1.0:
            return False
        if node.activation >= 0.1:
            return False
        if node.weight >= 0.5:
            return False
        if age < 3600:
            return False
        return True

    def record_access(self, node_id: str):
        with self._lock:
            self._access_log[node_id] = time.time()

    def offload(self, node_id: str, node_data: dict):
        with self._lock:
            compressed = self._compress(node_data)
            cold_path = os.path.join(self._storage_dir, f"{node_id}.gz")
            try:
                with gzip.open(cold_path, "wb") as f:
                    f.write(compressed)
            except Exception as e:
                logger.error("Failed to offload node %s: %s", node_id[:8], e)
                return
            raw_size = len(json.dumps(node_data).encode("utf-8"))
            self._cold_store[node_id] = {
                "path": cold_path,
                "size": len(compressed),
                "offloaded_at": time.time(),
            }
            self._total_offloaded += 1
            self._bytes_saved += max(0, raw_size - len(compressed))
            logger.debug("Offloaded node %s (saved %d bytes)", node_id[:8], max(0, raw_size - len(compressed)))

    def reload(self, node_id: str) -> Optional[dict]:
        with self._lock:
            meta = self._cold_store.get(node_id)
            if meta is None:
                return None
            try:
                with gzip.open(meta["path"], "rb") as f:
                    data = json.loads(f.read().decode("utf-8"))
            except Exception as e:
                logger.error("Failed to reload node %s: %s", node_id[:8], e)
                return None
            del self._cold_store[node_id]
            try:
                os.remove(meta["path"])
            except OSError:
                logger.debug("Could not remove cold-store file %s", meta["path"], exc_info=True)
            self._reload_count += 1
            self._access_log[node_id] = time.time()
            return data

    def batch_offload(self, node_ids: List[str], node_data_map: Dict[str, dict]) -> Dict[str, bool]:
        results: Dict[str, bool] = {}

        def _do_offload(nid: str) -> Tuple[str, bool]:
            data = node_data_map.get(nid)
            if data is None:
                return (nid, False)
            try:
                self.offload(nid, data)
                return (nid, True)
            except Exception as e:
                logger.error("batch_offload failed for %s: %s", nid[:8], e)
                return (nid, False)

        with _futures.ThreadPoolExecutor(max_workers=min(8, len(node_ids) or 1)) as executor:
            future_map = {
                executor.submit(_do_offload, nid): nid
                for nid in node_ids
            }
            for future in _futures.as_completed(future_map):
                try:
                    nid, ok = future.result()
                    results[nid] = ok
                except Exception as e:
                    nid = future_map[future]
                    results[nid] = False
                    logger.error("batch_offload exception for %s: %s", nid[:8], e)

        logger.info(
            "batch_offload: %d/%d succeeded",
            sum(1 for v in results.values() if v),
            len(node_ids),
        )
        return results

    def batch_reload(self, node_ids: List[str]) -> Dict[str, Optional[dict]]:
        results: Dict[str, Optional[dict]] = {}

        def _do_reload(nid: str) -> Tuple[str, Optional[dict]]:
            try:
                data = self.reload(nid)
                return (nid, data)
            except Exception as e:
                logger.error("batch_reload failed for %s: %s", nid[:8], e)
                return (nid, None)

        with _futures.ThreadPoolExecutor(max_workers=min(8, len(node_ids) or 1)) as executor:
            future_map = {
                executor.submit(_do_reload, nid): nid
                for nid in node_ids
            }
            for future in _futures.as_completed(future_map):
                try:
                    nid, data = future.result()
                    results[nid] = data
                except Exception as e:
                    nid = future_map[future]
                    results[nid] = None
                    logger.error("batch_reload exception for %s: %s", nid[:8], e)

        reloaded_count = sum(1 for v in results.values() if v is not None)
        logger.info(
            "batch_reload: %d/%d succeeded",
            reloaded_count,
            len(node_ids),
        )
        return results

    def is_hibernated(self, node_id: str) -> bool:
        return node_id in self._cold_store

    def get_hibernated_ids(self) -> Set[str]:
        return set(self._cold_store.keys())

    def get_stats(self) -> dict:
        with self._lock:
            hot_count = 0
            cold_count = len(self._cold_store)
            total = hot_count + cold_count
            return {
                "hot_count": hot_count,
                "cold_count": cold_count,
                "total_offloaded": self._total_offloaded,
                "reload_count": self._reload_count,
                "bytes_saved": self._bytes_saved,
                "hot_ratio": self._hot_ratio,
                "cold_files": len(self._cold_store),
            }

    def _compress(self, data: dict) -> bytes:
        raw = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        import io
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
            gz.write(raw)
        return buf.getvalue()


class DistributedCoordinator:
    """
    Coordinates multiple TetraMem instances for horizontal scaling.
    Uses a simple gossip protocol (no external services).
    """

    def __init__(self, instance_id: str, instance_url: str, port: int = 8000):
        self._instance_id = instance_id
        self._instance_url = instance_url
        self._port = port
        self._peers: Dict[str, Dict] = {}
        self._partitions: List[MeshPartition] = []
        self._label_ownership: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._heartbeat_interval = 10.0
        self._peer_timeout = 45.0
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def create_partitions(self, resolution: int, spacing: float, num_partitions: int = 4):
        with self._lock:
            self._partitions.clear()
            bound = resolution * spacing
            if num_partitions <= 1:
                p = MeshPartition(
                    f"{self._instance_id}_p0",
                    ((-bound, bound), (-bound, bound), (-bound, bound)),
                    [],
                )
                p.owner_instance = self._instance_url
                self._partitions.append(p)
                return
            axis_cuts = int(round(num_partitions ** (1.0 / 3.0)))
            axis_cuts = max(1, axis_cuts)
            step = (2 * bound) / axis_cuts
            idx = 0
            for ix in range(axis_cuts):
                for iy in range(axis_cuts):
                    for iz in range(axis_cuts):
                        x0 = -bound + ix * step
                        y0 = -bound + iy * step
                        z0 = -bound + iz * step
                        partition = MeshPartition(
                            f"{self._instance_id}_p{idx}",
                            (
                                (x0, x0 + step),
                                (y0, y0 + step),
                                (z0, z0 + step),
                            ),
                            [],
                        )
                        partition.owner_instance = self._instance_url
                        self._partitions.append(partition)
                        idx += 1

    def register_peer(self, peer_url: str) -> bool:
        try:
            req = urllib.request.Request(
                f"{peer_url}/api/v1/health",
                headers={"Content-Type": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") != "ok":
                    return False
        except Exception as e:
            logger.warning("Failed to register peer %s: %s", peer_url, e)
            return False
        with self._lock:
            self._peers[peer_url] = {
                "url": peer_url,
                "last_heartbeat": time.time(),
                "partitions": [],
                "status": "alive",
            }
        logger.info("Registered peer: %s", peer_url)
        return True

    def start_heartbeat(self):
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="distributed-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self):
        self._stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)

    def _heartbeat_loop(self):
        while not self._stop_event.wait(timeout=self._heartbeat_interval):
            self.heartbeat()

    def heartbeat(self) -> dict:
        status = self.get_cluster_status()
        with self._lock:
            for peer_url, peer_info in list(self._peers.items()):
                try:
                    payload = json.dumps({
                        "instance_id": self._instance_id,
                        "url": self._instance_url,
                        "partitions": [p.to_dict() for p in self._partitions],
                        "timestamp": time.time(),
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        f"{peer_url}/api/v1/distributed/gossip",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        resp.read()
                    peer_info["last_heartbeat"] = time.time()
                    peer_info["status"] = "alive"
                except Exception:
                    logger.warning("Gossip heartbeat failed to %s", peer_url, exc_info=True)
                    peer_info["status"] = "unreachable"
                    if time.time() - peer_info.get("last_heartbeat", 0) > self._peer_timeout:
                        peer_info["status"] = "dead"
        return status

    def receive_gossip(self, gossip_data: dict) -> dict:
        with self._lock:
            peer_url = gossip_data.get("url", "")
            if not peer_url:
                return {"status": "error", "message": "missing url"}
            self._peers[peer_url] = {
                "url": peer_url,
                "last_heartbeat": time.time(),
                "partitions": gossip_data.get("partitions", []),
                "status": "alive",
            }
            for pdata in gossip_data.get("partitions", []):
                p = MeshPartition.from_dict(pdata)
                p.owner_instance = peer_url
                for label in pdata.get("labels_owned", []):
                    self._label_ownership[label] = peer_url
        return {"status": "ok", "instance_id": self._instance_id}

    def route_query(self, query: str, labels: List[str]) -> str:
        with self._lock:
            for label in (labels or []):
                owner = self._label_ownership.get(label)
                if owner:
                    return owner
            for partition in self._partitions:
                if partition.access_mode == "local":
                    continue
                if partition.owner_instance and partition.owner_instance != self._instance_url:
                    return partition.owner_instance
        return self._instance_url

    def transfer_partition(self, partition_id: str, target_url: str) -> bool:
        partition = None
        with self._lock:
            for p in self._partitions:
                if p.id == partition_id:
                    partition = p
                    break
            if partition is None:
                return False
        try:
            payload = json.dumps({
                "partition": partition.to_dict(),
                "source_instance": self._instance_url,
                "timestamp": time.time(),
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{target_url}/api/v1/distributed/receive-partition",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            with self._lock:
                partition.owner_instance = target_url
                partition.access_mode = "remote"
            logger.info("Transferred partition %s to %s", partition_id, target_url)
            return True
        except Exception as e:
            logger.error("Failed to transfer partition %s: %s", partition_id, e)
            return False

    def receive_partition(self, transfer_data: dict) -> bool:
        pdata = transfer_data.get("partition", {})
        partition = MeshPartition.from_dict(pdata)
        with self._lock:
            partition.owner_instance = self._instance_url
            partition.access_mode = "local"
            self._partitions.append(partition)
            for label in pdata.get("labels_owned", []):
                self._label_ownership[label] = self._instance_url
        logger.info("Received partition %s from %s", partition.id, transfer_data.get("source_instance"))
        return True

    def propagate_store(self, content: str, labels: Optional[List[str]] = None,
                        weight: float = 1.0, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        payload_data = {
            "content": content,
            "labels": labels or [],
            "weight": weight,
            "metadata": metadata or {},
            "source_instance": self._instance_id,
            "source_url": self._instance_url,
            "timestamp": time.time(),
        }
        results: Dict[str, Any] = {
            "source": self._instance_id,
            "peers_notified": 0,
            "peers_failed": 0,
            "peer_results": {},
        }
        with self._lock:
            alive_peers = [
                (url, info) for url, info in self._peers.items()
                if info.get("status") == "alive"
            ]
        for peer_url, peer_info in alive_peers:
            try:
                encoded = json.dumps(payload_data).encode("utf-8")
                req = urllib.request.Request(
                    f"{peer_url}/api/v1/distributed/merge",
                    data=encoded,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                results["peers_notified"] += 1
                results["peer_results"][peer_url] = resp_data
            except Exception as e:
                results["peers_failed"] += 1
                results["peer_results"][peer_url] = {"error": str(e)}
                logger.warning(
                    "propagate_store: failed to notify peer %s: %s",
                    peer_url, e,
                )
        logger.info(
            "propagate_store: notified %d/%d peers",
            results["peers_notified"],
            len(alive_peers),
        )
        return results

    def merge_incoming(self, state_fragment: Dict) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "merged_nodes": 0,
            "merged_labels": 0,
            "merged_partitions": 0,
            "skipped": [],
        }
        with self._lock:
            source = state_fragment.get("source_instance", "unknown")
            for label in state_fragment.get("labels", []):
                if label and label not in self._label_ownership:
                    self._label_ownership[label] = source
                    result["merged_labels"] += 1

            for pdata in state_fragment.get("partitions", []):
                existing = None
                for p in self._partitions:
                    if p.id == pdata.get("id"):
                        existing = p
                        break
                if existing is not None:
                    new_ids = set(pdata.get("node_ids", []))
                    before = len(existing.node_ids)
                    existing.node_ids.update(new_ids)
                    if len(existing.node_ids) > before:
                        result["merged_partitions"] += 1
                else:
                    new_partition = MeshPartition.from_dict(pdata)
                    new_partition.owner_instance = source
                    self._partitions.append(new_partition)
                    result["merged_partitions"] += 1

            for nid, ndata in state_fragment.get("nodes", {}).items():
                result["merged_nodes"] += 1

            for gossip_entry in state_fragment.get("gossip", []):
                peer_url = gossip_entry.get("url", "")
                if peer_url:
                    self._peers[peer_url] = {
                        "url": peer_url,
                        "last_heartbeat": time.time(),
                        "partitions": gossip_entry.get("partitions", []),
                        "status": "alive",
                    }

        logger.info(
            "merge_incoming from %s: nodes=%d, labels=%d, partitions=%d",
            state_fragment.get("source_instance", "?"),
            result["merged_nodes"],
            result["merged_labels"],
            result["merged_partitions"],
        )
        return result

    def get_partition_stats(self) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for partition in self._partitions:
                p_stats: Dict[str, Any] = {
                    "id": partition.id,
                    "bounds": partition.bounds,
                    "access_mode": partition.access_mode,
                    "owner_instance": partition.owner_instance,
                    "node_count": len(partition.node_ids),
                    "is_local": partition.owner_instance == self._instance_url,
                }
                if partition.bounds:
                    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = partition.bounds
                    volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
                    p_stats["volume"] = round(volume, 4)
                stats[partition.id] = p_stats
        return stats

    def get_cluster_status(self) -> dict:
        with self._lock:
            alive_peers = sum(1 for p in self._peers.values() if p["status"] == "alive")
            dead_peers = sum(1 for p in self._peers.values() if p["status"] == "dead")
            return {
                "instance_id": self._instance_id,
                "instance_url": self._instance_url,
                "peers": {
                    "total": len(self._peers),
                    "alive": alive_peers,
                    "dead": dead_peers,
                    "details": {
                        url: {
                            "status": info["status"],
                            "last_heartbeat": info["last_heartbeat"],
                        }
                        for url, info in self._peers.items()
                    },
                },
                "partitions": [p.to_dict() for p in self._partitions],
                "label_ownership": dict(self._label_ownership),
            }
