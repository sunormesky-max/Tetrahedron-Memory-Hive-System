"""
TetraMesh partitioning: route tetrahedra into spatial buckets.

Each bucket contains a local TetraMesh. The router distributes new
tetrahedra by centroid proximity and handles cross-bucket queries
via Ghost Cells at bucket boundaries.
"""

import hashlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .partitioning import BoundingBox, GhostCell
from .tetra_mesh import TetraMesh

logger = logging.getLogger("tetramem.router")


class TetraBucket:
    def __init__(self, bucket_id: str, bounds: BoundingBox):
        self.id = bucket_id
        self.bounds = bounds
        self.mesh = TetraMesh(time_lambda=0.001)
        self.ghost_cells: Dict[str, GhostCell] = {}

    def store(self, content: str, seed_point: np.ndarray, **kwargs) -> str:
        return self.mesh.store(content=content, seed_point=seed_point, **kwargs)

    def query(self, query_point: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        return self.mesh.query_topological(query_point, k=k)

    def get_tetrahedron(self, tid: str) -> Optional[Any]:
        return self.mesh.get_tetrahedron(tid)

    def tetrahedra_count(self) -> int:
        return len(self.mesh.tetrahedra)


class TetraMeshRouter:
    def __init__(
        self,
        max_tetra_per_bucket: int = 2000,
        ghost_ttl: float = 3600.0,
        boundary_width: float = 0.15,
        consistency: Optional[Any] = None,
        persistence: Optional[Any] = None,
    ):
        self._buckets: Dict[str, TetraBucket] = {}
        self._bounds_map: Dict[str, BoundingBox] = {}
        self._sizes: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._max_per_bucket = max_tetra_per_bucket
        self._ghost_ttl = ghost_ttl
        self._boundary_width = boundary_width
        self._next_idx = 0
        self._consistency = consistency
        self._persistence = persistence

    def initialize(self, initial_bounds: Optional[BoundingBox] = None) -> None:
        bounds = initial_bounds or BoundingBox(
            np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])
        )
        self._create_bucket(bounds)

    def route_store(
        self, seed_point: np.ndarray, content: str, **kwargs
    ) -> Tuple[str, str]:
        with self._lock:
            bid = self._find_bucket(seed_point)
            if bid is None:
                bid = self._create_bucket(
                    BoundingBox(seed_point - 2.0, seed_point + 2.0)
                )

            if self._sizes.get(bid, 0) >= self._max_per_bucket:
                bid = self._split_and_route(bid, seed_point)

            bucket = self._buckets[bid]
            tid = bucket.store(content, seed_point, **kwargs)
            self._sizes[bid] = self._sizes.get(bid, 0) + 1

            if self._consistency is not None:
                try:
                    self._consistency.record_version(tid, bid, content)
                except Exception:
                    pass

            if self._persistence is not None:
                try:
                    from .persistence import MemorySnapshot
                    snap = MemorySnapshot(
                        id=tid,
                        content=content,
                        geometry=seed_point.tolist() if hasattr(seed_point, 'tolist') else list(seed_point),
                        timestamp=time.time(),
                        weight=kwargs.get("weight", 1.0),
                        labels=kwargs.get("labels") or [],
                        metadata=kwargs.get("metadata") or {},
                    )
                    self._persistence.write_incremental(bid, snap)
                except Exception:
                    pass

            self._update_ghosts(bid, seed_point, tid, content,
                                labels=kwargs.get("labels"))

            return bid, tid

    def route_query(
        self, query_point: np.ndarray, k: int = 5
    ) -> List[Tuple[str, float, str]]:
        with self._lock:
            if self._consistency is not None:
                try:
                    conflicts = self._consistency.detect_conflicts()
                    for vn1, vn2 in conflicts:
                        self._consistency.read_repair(
                            vn1.node_id, vn1.bucket_id, [vn2.bucket_id]
                        )
                except Exception:
                    pass

            results: List[Tuple[str, float, str]] = []

            scored_buckets = []
            for bid, bounds in self._bounds_map.items():
                center = bounds.center
                dist = float(np.linalg.norm(query_point - center))
                scored_buckets.append((dist, bid))
            scored_buckets.sort()
            top_buckets = [bid for _, bid in scored_buckets[:3]]

            for bid in top_buckets:
                bucket = self._buckets[bid]
                local = bucket.query(query_point, k=k)
                for tid, score in local:
                    results.append((tid, score, bid))

            results.sort(key=lambda x: x[1])
            return results[:k]

    def cross_bucket_associate(
        self, bucket_id: str, tetra_id: str, max_depth: int = 2
    ) -> List[Tuple[str, float, str, str]]:
        with self._lock:
            bucket = self._buckets.get(bucket_id)
            if bucket is None:
                return []

            local = bucket.mesh.associate_topological(tetra_id, max_depth)
            results = [(tid, s, ct, bucket_id) for tid, s, ct in local]

            for ghost_id, ghost in bucket.ghost_cells.items():
                if ghost.is_expired:
                    continue
                source_bid = ghost.source_bucket_id
                if source_bid and source_bid in self._buckets:
                    source_bucket = self._buckets[source_bid]
                    tetra = source_bucket.get_tetrahedron(ghost_id)
                    if tetra:
                        results.append((
                            ghost_id,
                            0.2,
                            "ghost_cell",
                            source_bid,
                        ))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

    def get_bucket(self, bucket_id: str) -> Optional[TetraBucket]:
        return self._buckets.get(bucket_id)

    def get_all_bucket_ids(self) -> List[str]:
        return list(self._buckets.keys())

    def get_statistics(self) -> Dict[str, Any]:
        total_tetra = sum(b.tetrahedra_count() for b in self._buckets.values())
        total_ghosts = sum(len(b.ghost_cells) for b in self._buckets.values())
        return {
            "total_buckets": len(self._buckets),
            "total_tetrahedra": total_tetra,
            "total_ghost_cells": total_ghosts,
            "bucket_sizes": {bid: b.tetrahedra_count() for bid, b in self._buckets.items()},
        }

    def _find_bucket(self, point: np.ndarray) -> Optional[str]:
        best_bid = None
        best_dist = float("inf")
        for bid, bounds in self._bounds_map.items():
            if bounds.contains(point):
                center = bounds.center
                dist = float(np.linalg.norm(point - center))
                if dist < best_dist:
                    best_dist = dist
                    best_bid = bid
        return best_bid

    def _create_bucket(self, bounds: BoundingBox) -> str:
        bid = f"bucket_{self._next_idx}"
        self._next_idx += 1
        bucket = TetraBucket(bid, bounds)
        self._buckets[bid] = bucket
        self._bounds_map[bid] = bounds
        self._sizes[bid] = 0
        if self._consistency is not None:
            try:
                self._consistency.add_bucket(bid)
            except Exception:
                pass
        return bid

    def _split_and_route(
        self, bid: str, point: np.ndarray
    ) -> str:
        bounds = self._bounds_map[bid]
        axis_sizes = bounds.size
        split_axis = int(np.argmax(axis_sizes))
        mid = (bounds.min_bounds[split_axis] + bounds.max_bounds[split_axis]) / 2.0

        min1 = bounds.min_bounds.copy()
        max1 = bounds.max_bounds.copy()
        max1[split_axis] = mid
        bounds1 = BoundingBox(min1, max1)

        min2 = bounds.min_bounds.copy()
        min2[split_axis] = mid
        max2 = bounds.max_bounds.copy()
        bounds2 = BoundingBox(min2, max2)

        old_bucket = self._buckets.pop(bid)
        old_data = []
        for tid, tetra in old_bucket.mesh.tetrahedra.items():
            old_data.append((tid, tetra.content, tetra.centroid, tetra))
        del self._bounds_map[bid]
        self._sizes.pop(bid, None)

        if bounds1.contains(point):
            new_bid = self._create_bucket(bounds1)
        else:
            new_bid = self._create_bucket(bounds2)

        for tid, content, centroid, tetra in old_data:
            target_bounds = bounds1 if bounds1.contains(centroid) else bounds2
            target_bid = None
            for b_id, b in self._bounds_map.items():
                if b.contains(centroid):
                    target_bid = b_id
                    break
            if target_bid is None:
                target_bid = self._create_bucket(target_bounds)
            self._buckets[target_bid].store(
                content=content,
                seed_point=centroid,
                labels=tetra.labels,
                metadata=tetra.metadata,
                weight=tetra.weight,
            )
            self._sizes[target_bid] = self._sizes.get(target_bid, 0) + 1

        return new_bid

    def _update_ghosts(
        self, bid: str, point: np.ndarray, tid: str, content: str,
        labels: Optional[List[str]] = None,
    ) -> None:
        bucket = self._buckets[bid]
        bounds = self._bounds_map[bid]
        width = bounds.size * self._boundary_width
        inner = BoundingBox(bounds.min_bounds + width, bounds.max_bounds - width)

        if not inner.contains(point):
            for other_bid, other_bounds in self._bounds_map.items():
                if other_bid == bid:
                    continue
                if other_bounds.contains(point):
                    ghost = GhostCell(
                        node_id=tid,
                        source_bucket_id=other_bid,
                        geometry=point.copy(),
                        weight=1.0,
                        content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
                        labels=labels or [],
                        ttl=self._ghost_ttl,
                    )
                    other_bucket = self._buckets.get(other_bid)
                    if other_bucket is not None:
                        other_bucket.ghost_cells[tid] = ghost
                    break
