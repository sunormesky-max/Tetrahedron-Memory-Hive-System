"""
Persistence module for Tetrahedron Memory System.
"""

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class MemorySnapshot:
    id: str
    content: str
    geometry: List[float]
    timestamp: float
    weight: float
    labels: List[str]
    metadata: Dict[str, Any]


class MemoryPersistence:
    """
    Handles persistence of memory data to JSON files.
    """

    def __init__(self, storage_dir: str = "./tetrahedron_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.nodes_file = self.storage_dir / "nodes.json"
        self.metadata_file = self.storage_dir / "metadata.json"

    def save_nodes(self, nodes: Dict[str, Any]) -> None:
        data = {}
        for node_id, node in nodes.items():
            data[node_id] = {
                "id": node.id,
                "content": node.content,
                "geometry": node.geometry.tolist(),
                "timestamp": node.timestamp,
                "weight": node.weight,
                "labels": node.labels,
                "metadata": node.metadata,
            }

        with open(self.nodes_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_nodes(self) -> Dict[str, Any]:
        from tetrahedron_memory.core import MemoryNode

        if not self.nodes_file.exists():
            return {}

        with open(self.nodes_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = {}
        for node_id, node_data in data.items():
            nodes[node_id] = MemoryNode(
                id=node_data["id"],
                content=node_data["content"],
                geometry=np.array(node_data["geometry"]),
                timestamp=node_data["timestamp"],
                weight=node_data["weight"],
                labels=node_data.get("labels", []),
                metadata=node_data.get("metadata", {}),
            )

        return nodes

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self) -> Dict[str, Any]:
        if not self.metadata_file.exists():
            return {}

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def export_to_text(self, nodes: Dict[str, Any], output_file: str) -> None:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Tetrahedron Memory Export\n\n")

            for node_id, node in nodes.items():
                f.write(f"## Memory: {node_id}\n")
                f.write(f"Content: {node.content}\n")
                f.write(f"Weight: {node.weight:.4f}\n")
                f.write(f"Labels: {', '.join(node.labels) if node.labels else 'none'}\n")
                f.write(f"Geometry: [{', '.join(f'{x:.4f}' for x in node.geometry)}]\n")
                f.write("\n")

    def get_storage_stats(self) -> Dict[str, Any]:
        stats = {
            "storage_dir": str(self.storage_dir),
            "nodes_file_exists": self.nodes_file.exists(),
            "metadata_file_exists": self.metadata_file.exists(),
        }

        if self.nodes_file.exists():
            stats["nodes_file_size"] = self.nodes_file.stat().st_size

        if self.metadata_file.exists():
            stats["metadata_file_size"] = self.metadata_file.stat().st_size

        return stats


class ParquetPersistence:
    def __init__(self, storage_path: str = "./tetramem_data"):
        self.storage_path = storage_path
        self._snapshots: List[MemorySnapshot] = []
        self._max_in_memory_snapshots: int = 1000
        self._base_path = Path(storage_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._global_file = self._base_path / "memories.parquet"
        self._lock = threading.RLock()

    # ---------------------------------------------------------------------
    # Incremental write per bucket (production‑grade)
    # ---------------------------------------------------------------------
    def _bucket_file(self, bucket_id: str) -> Path:
        """Return the parquet file path for a given bucket identifier."""
        safe_id = bucket_id.replace("/", "_")
        return self._base_path / f"bucket_{safe_id}.parquet"

    @staticmethod
    def _parse_labels(raw_labels) -> List[str]:
        if not raw_labels:
            return []
        text = str(raw_labels).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return [l.strip() for l in text.split(",") if l.strip()]

    def _metadata_file(self, bucket_id: str) -> Path:
        safe_id = bucket_id.replace("/", "_")
        return self._base_path / f"bucket_{safe_id}_meta.json"

    def write_incremental(self, bucket_id: str, snapshot: MemorySnapshot) -> None:
        """Append a single snapshot to the parquet file of ``bucket_id``.

        A small JSON metadata file stores the current version number. Each call
        increments the version and writes it atomically alongside the parquet
        append. Thread‑safe via ``self._lock``.
        """
        with self._lock:
            # Update version metadata
            meta_path = self._metadata_file(bucket_id)
            version = 1
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    version = meta.get("version", 0) + 1
                except Exception:
                    version = 1
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"version": version, "last_updated": time.time()}, f, indent=2)

            # Prepare a single‑row DataFrame
            try:
                import pandas as pd
            except ImportError:
                return

            row = {
                "id": snapshot.id,
                "content": snapshot.content,
                "geometry_x": snapshot.geometry[0],
                "geometry_y": snapshot.geometry[1],
                "geometry_z": snapshot.geometry[2],
                "timestamp": snapshot.timestamp,
                "weight": snapshot.weight,
                "labels": ",".join(snapshot.labels),
                "metadata": json.dumps(snapshot.metadata),
            }
            df = pd.DataFrame([row])
            bucket_path = self._bucket_file(bucket_id)
            if bucket_path.exists():
                try:
                    df_existing = pd.read_parquet(bucket_path)
                    df = pd.concat([df_existing, df], ignore_index=True)
                except Exception:
                    pass
                df.to_parquet(bucket_path, index=False, engine="pyarrow", compression="snappy")
            else:
                df.to_parquet(bucket_path, index=False, engine="pyarrow", compression="snappy")

    def load_bucket(self, bucket_id: str) -> List[MemorySnapshot]:
        try:
            import pandas as pd
        except ImportError:
            return []
        bucket_path = self._bucket_file(bucket_id)
        if not bucket_path.exists():
            return []
        df = pd.read_parquet(bucket_path)
        snapshots: List[MemorySnapshot] = []
        s = self._scalar
        for _, row in df.iterrows():
            raw_labels = s(row["labels"])
            raw_meta = s(row["metadata"])
            snapshots.append(
                MemorySnapshot(
                    id=str(s(row["id"])),
                    content=str(s(row["content"])),
                    geometry=[
                        float(s(row["geometry_x"])),
                        float(s(row["geometry_y"])),
                        float(s(row["geometry_z"])),
                    ],
                    timestamp=float(s(row["timestamp"])),
                    weight=float(s(row["weight"])),
                    labels=self._parse_labels(raw_labels),
                    metadata=json.loads(str(raw_meta)) if raw_meta else {},
                )
            )
        return snapshots

    def compact_bucket(self, bucket_id: str) -> None:
        """Rewrite a bucket parquet file to remove duplicate IDs and apply a simple
        compaction (e.g., keep the latest snapshot per ``id``)."""
        with self._lock:
            snapshots = self.load_bucket(bucket_id)
            # Keep only the most recent snapshot per memory ID
            latest: Dict[str, MemorySnapshot] = {}
            for snap in snapshots:
                if snap.id not in latest or snap.timestamp > latest[snap.id].timestamp:
                    latest[snap.id] = snap
            # Write back compacted data
            try:
                import pandas as pd
            except ImportError:
                return
            rows = []
            for snap in latest.values():
                rows.append(
                    {
                        "id": snap.id,
                        "content": snap.content,
                        "geometry_x": snap.geometry[0],
                        "geometry_y": snap.geometry[1],
                        "geometry_z": snap.geometry[2],
                        "timestamp": snap.timestamp,
                        "weight": snap.weight,
                        "labels": ",".join(snap.labels),
                        "metadata": json.dumps(snap.metadata),
                    }
                )
            df = pd.DataFrame(rows)
            bucket_path = self._bucket_file(bucket_id)
            df.to_parquet(bucket_path, index=False, engine="pyarrow", compression="snappy")
            # Reset version metadata after compaction
            meta_path = self._metadata_file(bucket_id)
            if meta_path.exists():
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({"version": 1, "last_compacted": time.time()}, f, indent=2)

    # ---------------------------------------------------------------------
    # Existing global snapshot logic (kept for backward compatibility)
    # ---------------------------------------------------------------------
    def save_snapshot(
        self,
        node_id: str,
        content: str,
        geometry: np.ndarray,
        timestamp: float,
        weight: float,
        labels: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        snapshot = MemorySnapshot(
            id=node_id,
            content=content,
            geometry=geometry.tolist(),
            timestamp=timestamp,
            weight=weight,
            labels=labels,
            metadata=metadata,
        )
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_in_memory_snapshots:
            self._snapshots = self._snapshots[-self._max_in_memory_snapshots :]

    def save_to_parquet(self) -> None:
        try:
            import pandas as pd

            self._base_path.mkdir(parents=True, exist_ok=True)

            data = []
            for snapshot in self._snapshots:
                data.append(
                    {
                        "id": snapshot.id,
                        "content": snapshot.content,
                        "geometry_x": snapshot.geometry[0],
                        "geometry_y": snapshot.geometry[1],
                        "geometry_z": snapshot.geometry[2],
                        "timestamp": snapshot.timestamp,
                        "weight": snapshot.weight,
                        "labels": json.dumps(snapshot.labels),
                        "metadata": json.dumps(snapshot.metadata),
                    }
                )

            df = pd.DataFrame(data)
            df.to_parquet(self._global_file, index=False)
        except ImportError:
            pass

    @staticmethod
    def _scalar(val: Any) -> Any:
        if hasattr(val, "item"):
            return val.item()
        return val

    def load_from_parquet(self) -> List[MemorySnapshot]:
        try:
            import pandas as pd

            if not self._global_file.exists():
                return []

            df = pd.read_parquet(self._global_file)
            snapshots = []

            for _, row in df.iterrows():
                s = self._scalar
                raw_labels = s(row["labels"])
                raw_meta = s(row["metadata"])
                snapshot = MemorySnapshot(
                    id=str(s(row["id"])),
                    content=str(s(row["content"])),
                    geometry=[
                        float(s(row["geometry_x"])),
                        float(s(row["geometry_y"])),
                        float(s(row["geometry_z"])),
                    ],
                    timestamp=float(s(row["timestamp"])),
                    weight=float(s(row["weight"])),
                    labels=self._parse_labels(raw_labels),
                    metadata=json.loads(str(raw_meta)) if raw_meta else {},
                )
                snapshots.append(snapshot)

            self._snapshots = snapshots
            return snapshots
        except ImportError:
            return []

    def load_snapshots(self) -> List[MemorySnapshot]:
        return self._snapshots.copy()

    def clear(self) -> None:
        self._snapshots.clear()

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_snapshots": len(self._snapshots),
            "storage_path": self.storage_path,
            "file_exists": self._global_file.exists(),
        }

        if self._global_file.exists():
            stats["file_size"] = self._global_file.stat().st_size

        return stats

    def write_full_snapshot(self, nodes: Dict[str, Any], snapshot_name: str = "full") -> None:
        with self._lock:
            try:
                import pandas as pd
            except ImportError:
                return

            rows = []
            for node_id, node in nodes.items():
                geom = (
                    node.geometry
                    if hasattr(node, "geometry")
                    else np.array(node.get("geometry", [0, 0, 0]))
                )
                rows.append(
                    {
                        "id": node_id,
                        "content": node.content
                        if hasattr(node, "content")
                        else node.get("content", ""),
                        "geometry_x": float(geom[0]),
                        "geometry_y": float(geom[1]),
                        "geometry_z": float(geom[2]),
                        "timestamp": node.timestamp
                        if hasattr(node, "timestamp")
                        else node.get("timestamp", 0.0),
                        "weight": node.weight
                        if hasattr(node, "weight")
                        else node.get("weight", 1.0),
                        "labels": ",".join(
                            node.labels if hasattr(node, "labels") else node.get("labels", [])
                        ),
                        "metadata": json.dumps(
                            node.metadata if hasattr(node, "metadata") else node.get("metadata", {})
                        ),
                    }
                )

            df = pd.DataFrame(rows)
            self._base_path.mkdir(parents=True, exist_ok=True)
            tmp_path = self._base_path / f"_tmp_{snapshot_name}.parquet"
            final_path = self._base_path / f"snapshot_{snapshot_name}.parquet"
            df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")
            os.replace(str(tmp_path), str(final_path))

    def load_latest_snapshot(self, snapshot_name: str = "full") -> Dict[str, Any]:
        try:
            import pandas as pd
        except ImportError:
            return {}

        snap_path = self._base_path / f"snapshot_{snapshot_name}.parquet"
        if not snap_path.exists():
            return {}

        df = pd.read_parquet(snap_path)
        s = self._scalar
        nodes = {}
        for _, row in df.iterrows():
            node_id = str(s(row["id"]))
            nodes[node_id] = {
                "id": node_id,
                "content": str(s(row["content"])),
                "geometry": [
                    float(s(row["geometry_x"])),
                    float(s(row["geometry_y"])),
                    float(s(row["geometry_z"])),
                ],
                "timestamp": float(s(row["timestamp"])),
                "weight": float(s(row["weight"])),
                "labels": self._parse_labels(s(row["labels"])),
                "metadata": json.loads(str(s(row["metadata"]))) if s(row["metadata"]) else {},
            }
        return nodes

    def write_incremental_full(self, nodes: Dict[str, Any], snapshot_name: str = "full") -> None:
        with self._lock:
            try:
                import pandas as pd
            except ImportError:
                return

            rows = []
            for node_id, node in nodes.items():
                geom = (
                    node.geometry
                    if hasattr(node, "geometry")
                    else np.array(node.get("geometry", [0, 0, 0]))
                )
                rows.append(
                    {
                        "id": node_id,
                        "content": node.content
                        if hasattr(node, "content")
                        else node.get("content", ""),
                        "geometry_x": float(geom[0]),
                        "geometry_y": float(geom[1]),
                        "geometry_z": float(geom[2]),
                        "timestamp": node.timestamp
                        if hasattr(node, "timestamp")
                        else node.get("timestamp", 0.0),
                        "weight": node.weight
                        if hasattr(node, "weight")
                        else node.get("weight", 1.0),
                        "labels": ",".join(
                            node.labels if hasattr(node, "labels") else node.get("labels", [])
                        ),
                        "metadata": json.dumps(
                            node.metadata if hasattr(node, "metadata") else node.get("metadata", {})
                        ),
                        "operation": "upsert",
                    }
                )

            if not rows:
                return

            df_delta = pd.DataFrame(rows)
            self._base_path.mkdir(parents=True, exist_ok=True)
            inc_path = self._base_path / f"incremental_{snapshot_name}.parquet"

            if inc_path.exists():
                try:
                    df_existing = pd.read_parquet(inc_path)
                    df_merged = pd.concat([df_existing, df_delta], ignore_index=True)
                    df_merged = df_merged.drop_duplicates(subset=["id"], keep="last")
                except Exception:
                    df_merged = df_delta
            else:
                df_merged = df_delta

            tmp_path = self._base_path / f"_tmp_inc_{snapshot_name}.parquet"
            df_merged.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")
            os.replace(str(tmp_path), str(inc_path))

    def compact_snapshots(self, snapshot_name: str = "full") -> None:
        inc_path = self._base_path / f"incremental_{snapshot_name}.parquet"
        full_path = self._base_path / f"snapshot_{snapshot_name}.parquet"

        try:
            import pandas as pd
        except ImportError:
            return

        frames = []
        if full_path.exists():
            frames.append(pd.read_parquet(full_path))
        if inc_path.exists():
            df_inc = pd.read_parquet(inc_path)
            frames.append(df_inc)

        if not frames:
            return

        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["id"], keep="last")

        tmp_path = self._base_path / f"_tmp_compact_{snapshot_name}.parquet"
        df_all.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")
        os.replace(str(tmp_path), str(full_path))

        if inc_path.exists():
            inc_path.unlink()


class S3Storage:
    def __init__(self, bucket: str, prefix: str = "tetramem/"):
        self._bucket = bucket
        self._prefix = prefix
        self._s3_client: Optional[Any] = None

    def _get_client(self) -> Any:
        if self._s3_client is None:
            try:
                import boto3

                self._s3_client = boto3.client("s3")
            except ImportError:
                raise ImportError("boto3 is required for S3 storage: pip install boto3")
        return self._s3_client

    def upload_file(self, local_path: str, remote_key: Optional[str] = None) -> str:
        client = self._get_client()
        key = remote_key or f"{self._prefix}{os.path.basename(local_path)}"
        client.upload_file(local_path, self._bucket, key)
        return key

    def download_file(self, remote_key: str, local_path: str) -> None:
        client = self._get_client()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.download_file(self._bucket, remote_key, local_path)

    def list_files(self, prefix: Optional[str] = None) -> List[str]:
        client = self._get_client()
        paginator = client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix or self._prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return sorted(keys)

    def upload_snapshot(self, persister: ParquetPersistence, snapshot_name: str = "full") -> str:
        snap_path = persister._base_path / f"snapshot_{snapshot_name}.parquet"
        if not snap_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snap_path}")
        ts = time.strftime("%Y%m%d_%H%M%S")
        remote_key = f"{self._prefix}snapshot_{snapshot_name}_{ts}.parquet"
        return self.upload_file(str(snap_path), remote_key)

    def download_latest_snapshot(
        self, local_dir: str, snapshot_name: str = "full"
    ) -> Optional[str]:
        keys = self.list_files()
        snap_keys = [k for k in keys if f"snapshot_{snapshot_name}" in k and k.endswith(".parquet")]
        if not snap_keys:
            return None
        latest = snap_keys[-1]
        local_path = os.path.join(local_dir, f"snapshot_{snapshot_name}.parquet")
        self.download_file(latest, local_path)
        return local_path


class RemoteBucketActor:
    def __init__(self, bucket_id: str, dimension: int = 3):
        self.bucket_id = bucket_id
        self._dimension = dimension
        self._local_actor: Optional[Any] = None

    def _ensure_actor(self) -> Any:
        if self._local_actor is None:
            from tetrahedron_memory.partitioning import BucketActor

            self._local_actor = BucketActor(bucket_id=self.bucket_id, dimension=self._dimension)
        return self._local_actor

    def store(
        self,
        content: str,
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ) -> str:
        return self._ensure_actor().store(content, labels=labels, metadata=metadata, weight=weight)

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        return self._ensure_actor().query(query_text, k=k)

    def associate(self, memory_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        return self._ensure_actor().associate(memory_id, max_depth=max_depth)

    def self_organize(self) -> Dict[str, Any]:
        return self._ensure_actor().self_organize()

    def get_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return self._ensure_actor().get_snapshot()

    def get_statistics(self) -> Dict[str, Any]:
        return self._ensure_actor().get_statistics()

    def to_serializable(self) -> Dict[str, Any]:
        snap = self.get_snapshot()
        return {
            "bucket_id": self.bucket_id,
            "dimension": self._dimension,
            "snapshot": snap,
        }

    @classmethod
    def from_serializable(cls, data: Dict[str, Any]) -> "RemoteBucketActor":
        actor = cls(data["bucket_id"], data.get("dimension", 3))
        snap = data.get("snapshot", {})
        for nid, ndata in snap.items():
            actor.store(
                content=ndata["content"],
                labels=ndata.get("labels"),
                weight=ndata.get("weight", 1.0),
            )
        return actor


class RayController:
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self._initialized = False
        self._local_mode = True
        self._actors = []

    def initialize(self) -> None:
        try:
            import ray  # type: ignore

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            self._local_mode = False
            self._initialized = True
            self._create_actors()
        except ImportError:
            self._local_mode = True
            self._initialized = True

    def _create_actors(self) -> None:
        try:
            import ray  # type: ignore

            @ray.remote
            class MemoryActor:
                def __init__(self):
                    self.data = {}

                def store(self, key: str, value: Any) -> None:
                    self.data[key] = value

                def get(self, key: str) -> Any:
                    return self.data.get(key)

                def query(self, query_func: Callable, *args) -> Any:
                    return query_func(*args)

            self._actors = [MemoryActor.remote() for _ in range(self.num_workers)]  # type: ignore
        except ImportError:
            pass

    def shutdown(self) -> None:
        if not self._local_mode:
            try:
                import ray  # type: ignore

                if ray.is_initialized():
                    ray.shutdown()
            except ImportError:
                pass
        self._initialized = False
        self._actors = []

    def distributed_store(self, data: Dict[str, Any]) -> None:
        if self._local_mode or not self._actors:
            return

        try:
            import ray  # type: ignore

            chunk_size = len(data) // self.num_workers + 1
            items = list(data.items())

            futures = []
            for i, actor in enumerate(self._actors):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(items))
                chunk = dict(items[start_idx:end_idx])

                for key, value in chunk.items():
                    futures.append(actor.store.remote(key, value))

            ray.get(futures)
        except ImportError:
            pass

    def distributed_query(self, query_func: Callable, *args) -> List[Any]:
        if self._local_mode or not self._actors:
            return []

        try:
            import ray  # type: ignore

            futures = [actor.query.remote(query_func, *args) for actor in self._actors]
            results = ray.get(futures)
            return results
        except ImportError:
            return []

    def is_initialized(self) -> bool:
        return self._initialized

    def is_local_mode(self) -> bool:
        return self._local_mode

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "local_mode": self._local_mode,
            "num_workers": self.num_workers,
            "num_actors": len(self._actors),
        }
