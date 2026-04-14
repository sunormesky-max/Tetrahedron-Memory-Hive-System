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



    def navigate_cross_bucket(
        self, start_bucket_id: str, start_tetra_id: str,
        max_hops: int = 20, cross_bucket_limit: int = 3,
    ) -> List[Tuple[str, float, str, str]]:
        """Navigate topology across bucket boundaries via ghost cells.

        Starts with local topology navigation, then uses ghost cells
        to bridge into neighboring buckets.

        Returns: [(tetra_id, score, connection_type, bucket_id)]
        """
        with self._lock:
            results = []
            visited = set()
            visited.add(start_tetra_id)

            start_bucket = self._buckets.get(start_bucket_id)
            if start_bucket is None:
                return []

            local_nav = start_bucket.mesh.navigate_topology(
                start_tetra_id, max_steps=max_hops
            )
            for tid, conn_type, hop in local_nav:
                if tid not in visited:
                    visited.add(tid)
                    score = 1.0 / (1.0 + hop * 0.2)
                    results.append((tid, score, conn_type, start_bucket_id))

            cross_count = 0
            for ghost_id, ghost in start_bucket.ghost_cells.items():
                if cross_count >= cross_bucket_limit:
                    break
                if ghost.is_expired:
                    continue
                if ghost_id in visited:
                    continue

                source_bid = ghost.source_bucket_id
                source_bucket = self._buckets.get(source_bid)
                if source_bucket is None:
                    continue

                tetra = source_bucket.get_tetrahedron(ghost_id)
                if tetra is None:
                    continue

                visited.add(ghost_id)
                score = ghost.weight * 0.5 / (1.0 + ghost.access_count * 0.05)
                results.append((ghost_id, score, "ghost_bridge", source_bid))
                cross_count += 1

                remote_nav = source_bucket.mesh.navigate_topology(
                    ghost_id, max_steps=max_hops // 2
                )
                for tid, conn_type, hop in remote_nav:
                    if tid not in visited:
                        visited.add(tid)
                        s = score * (1.0 / (1.0 + hop * 0.3))
                        results.append((tid, s, "remote_" + conn_type, source_bid))

            results.sort(key=lambda x: -x[1])
            return results

    def distributed_dream(self, walk_steps: int = 12) -> Dict[str, Any]:
        """Run dream cycles across all buckets, then cross-pollinate.

        Phase 1: Each bucket runs a local dream cycle.
        Phase 2: Cross-bucket synthesis using ghost cell bridges.
        """
        with self._lock:
            stats = {
                "phase": "idle",
                "buckets_processed": 0,
                "local_dreams_created": 0,
                "cross_dreams_created": 0,
                "total_tetra_before": 0,
                "total_tetra_after": 0,
            }

            total_before = sum(b.tetrahedra_count() for b in self._buckets.values())
            stats["total_tetra_before"] = total_before

            if len(self._buckets) < 1:
                stats["phase"] = "no_buckets"
                return stats

            from .tetra_dream import TetraDreamCycle

            local_dreams = 0
            for bid, bucket in self._buckets.items():
                if bucket.tetrahedra_count() < 3:
                    continue
                dream = TetraDreamCycle(
                    bucket.mesh, walk_steps=walk_steps, dream_weight=0.5,
                )
                dream_stats = dream.trigger_now()
                local_dreams += dream_stats.get("dreams_created", 0)
                stats["buckets_processed"] += 1

            stats["local_dreams_created"] = local_dreams

            cross_dreams = 0
            bucket_ids = list(self._buckets.keys())
            for i in range(len(bucket_ids)):
                for j in range(i + 1, min(len(bucket_ids), i + 3)):
                    bid_a = bucket_ids[i]
                    bid_b = bucket_ids[j]
                    bucket_a = self._buckets[bid_a]
                    bucket_b = self._buckets[bid_b]

                    if bucket_a.tetrahedra_count() < 2 or bucket_b.tetrahedra_count() < 2:
                        continue

                    shared_ghosts = 0
                    for gid, ghost in bucket_a.ghost_cells.items():
                        if ghost.source_bucket_id == bid_b and not ghost.is_expired:
                            shared_ghosts += 1

                    if shared_ghosts > 0:
                        tetras_a = list(bucket_a.mesh.tetrahedra.values())
                        tetras_b = list(bucket_b.mesh.tetrahedra.values())
                        if tetras_a and tetras_b:
                            import hashlib as _hl
                            import time as _t
                            best_a = max(tetras_a, key=lambda t: t.weight)
                            best_b = max(tetras_b, key=lambda t: t.weight)
                            shared_lbl = set(best_a.labels) & set(best_b.labels)
                            shared_lbl.discard("__dream__")
                            shared_lbl.discard("__system__")
                            lbl_str = ", ".join(shared_lbl) if shared_lbl else "cross"
                            content = "[cross-dream:" + lbl_str + "] " + best_a.content[:40] + " | " + best_b.content[:40]
                            bridge = (best_a.centroid + best_b.centroid) / 2.0
                            bridge += np.random.normal(0, 0.01, size=3)

                            target = bucket_a if bucket_a.tetrahedra_count() <= bucket_b.tetrahedra_count() else bucket_b
                            target.mesh.store(
                                content=content,
                                seed_point=bridge,
                                labels=list(shared_lbl) + ["__dream__", "__cross__"],
                                metadata={
                                    "type": "cross_bucket_dream",
                                    "sources": [bid_a, bid_b],
                                    "ghost_bridges": shared_ghosts,
                                },
                                weight=0.5,
                            )
                            cross_dreams += 1

            stats["cross_dreams_created"] = cross_dreams

            total_after = sum(b.tetrahedra_count() for b in self._buckets.values())
            stats["total_tetra_after"] = total_after
            stats["phase"] = "complete"
            return stats

    def distributed_self_org(self, max_iterations: int = 5) -> Dict[str, Any]:
        """Run self-organization across all buckets in parallel.

        Each bucket self-organizes independently, then cross-bucket
        ghost cells are refreshed.
        """
        with self._lock:
            stats = {
                "phase": "idle",
                "buckets_processed": 0,
                "total_actions": 0,
                "ghost_refreshes": 0,
            }

            from .tetra_self_org import TetraSelfOrganizer

            total_actions = 0
            for bid, bucket in self._buckets.items():
                if bucket.tetrahedra_count() < 3:
                    continue
                org = TetraSelfOrganizer(bucket.mesh, max_iterations=max_iterations)
                org_stats = org.run()
                total_actions += org_stats.get("total_actions", 0)
                stats["buckets_processed"] += 1

            stats["total_actions"] = total_actions

            ghost_refreshes = 0
            for bid in list(self._buckets.keys()):
                bucket = self._buckets[bid]
                expired = [gid for gid, g in bucket.ghost_cells.items() if g.is_expired]
                for gid in expired:
                    del bucket.ghost_cells[gid]
                    ghost_refreshes += 1

            stats["ghost_refreshes"] = ghost_refreshes
            stats["phase"] = "complete"
            return stats

    def auto_balance(self) -> Dict[str, Any]:
        """Check bucket sizes and split oversized ones."""
        with self._lock:
            splits = 0
            for bid in list(self._sizes.keys()):
                if self._sizes.get(bid, 0) >= self._max_per_bucket:
                    self._split_and_route(bid, self._bounds_map[bid].center)
                    splits += 1
            return {"splits": splits, "buckets": len(self._buckets)}


    def invalidate_ghost_for(self, tetra_id: str, new_version: int) -> int:
        """Invalidate ghost cells referencing a mutated tetrahedron.

        Called after integration/weight-change/label-change to mark
        ghost cells in other buckets as stale.
        """
        invalidated = 0
        with self._lock:
            for bid, bucket in self._buckets.items():
                ghost = bucket.ghost_cells.get(tetra_id)
                if ghost is not None:
                    ghost.source_version = new_version
                    invalidated += 1
        return invalidated

    def verify_ghost_cells(self, bucket_id: Optional[str] = None) -> Dict[str, Any]:
        """Verify ghost cells against source buckets, refresh or remove stale ones.

        Under high load, ghost cells can drift from their source. This method
        re-verifies by checking source tetrahedra still exist and versions match.
        """
        stats = {"verified": 0, "refreshed": 0, "removed": 0, "stale": 0}
        with self._lock:
            target_ids = [bucket_id] if bucket_id else list(self._buckets.keys())
            for bid in target_ids:
                bucket = self._buckets.get(bid)
                if bucket is None:
                    continue
                to_remove = []
                for gid, ghost in bucket.ghost_cells.items():
                    stats["verified"] += 1
                    if ghost.is_expired:
                        to_remove.append(gid)
                        continue
                    if not ghost.needs_verification:
                        continue
                    source_bid = ghost.source_bucket_id
                    source_bucket = self._buckets.get(source_bid)
                    if source_bucket is None:
                        to_remove.append(gid)
                        continue
                    tetra = source_bucket.get_tetrahedron(gid)
                    if tetra is None:
                        to_remove.append(gid)
                        stats["removed"] += 1
                        continue
                    current_version = tetra.integration_count
                    is_consistent = ghost.verify(current_version, tetra.weight)
                    if not is_consistent:
                        ghost.weight = tetra.weight
                        ghost.labels = list(tetra.labels)
                        stats["refreshed"] += 1
                    if ghost.is_stale:
                        stats["stale"] += 1
                for gid in to_remove:
                    del bucket.ghost_cells[gid]
                    stats["removed"] += 1
        return stats

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
