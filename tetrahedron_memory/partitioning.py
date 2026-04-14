import hashlib
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class BoundingBox:
    min_bounds: np.ndarray
    max_bounds: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.min_bounds + self.max_bounds) / 2

    @property
    def size(self) -> np.ndarray:
        return self.max_bounds - self.min_bounds

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.all(point >= self.min_bounds) and np.all(point <= self.max_bounds))

    def intersects(self, other: "BoundingBox") -> bool:
        return bool(
            np.all(self.min_bounds <= other.max_bounds)
            and np.all(self.max_bounds >= other.min_bounds)
        )


@dataclass
class GhostCell:
    """Lightweight shadow copy of a boundary node replicated into a neighbor bucket.

    Ghost cells enable fast cross-bucket association without querying remote actors.
    They carry enough metadata to score association relevance and expire when stale.

    Version tracking (v2): Each ghost carries a version number that matches the
    source tetrahedron's integration_count. When the source mutates (integration,
    weight change, label change), the ghost becomes stale until re-verified.
    """

    node_id: str
    source_bucket_id: str
    geometry: np.ndarray
    weight: float = 1.0
    content_hash: str = ""
    labels: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    ttl: float = 3600.0
    access_count: int = 0
    version: int = 0
    source_version: int = 0
    last_verified_at: float = field(default_factory=time.time)
    verify_interval: float = 60.0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl

    @property
    def is_stale(self) -> bool:
        return self.version != self.source_version

    @property
    def needs_verification(self) -> bool:
        return (time.time() - self.last_verified_at) > self.verify_interval

    def touch(self) -> None:
        self.access_count += 1

    def verify(self, current_version: int, current_weight: float) -> bool:
        self.last_verified_at = time.time()
        if self.version != current_version:
            self.version = current_version
            self.source_version = current_version
            self.weight = current_weight
            return False
        return True


@dataclass
class OctreeNode:
    bounds: BoundingBox
    points: List[Tuple[np.ndarray, str]] = field(default_factory=list)
    children: Optional[List["OctreeNode"]] = None
    depth: int = 0
    max_points: int = 8
    max_depth: int = 10

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def is_empty(self) -> bool:
        return len(self.points) == 0 and self.is_leaf

    def subdivide(self) -> None:
        if not self.is_leaf:
            return

        center = self.bounds.center
        min_b = self.bounds.min_bounds
        max_b = self.bounds.max_bounds

        self.children = []
        for i in range(8):
            child_min = np.array(
                [
                    min_b[0] if (i & 1) == 0 else center[0],
                    min_b[1] if (i & 2) == 0 else center[1],
                    min_b[2] if (i & 4) == 0 else center[2],
                ]
            )
            child_max = np.array(
                [
                    center[0] if (i & 1) == 0 else max_b[0],
                    center[1] if (i & 2) == 0 else max_b[1],
                    center[2] if (i & 4) == 0 else max_b[2],
                ]
            )
            child_bounds = BoundingBox(child_min, child_max)
            self.children.append(
                OctreeNode(
                    bounds=child_bounds,
                    depth=self.depth + 1,
                    max_points=self.max_points,
                    max_depth=self.max_depth,
                )
            )

        points_to_reinsert = self.points.copy()
        self.points.clear()

        for point, node_id in points_to_reinsert:
            for child in self.children:
                if child.bounds.contains(point):
                    child.insert(point, node_id)
                    break

    def insert(self, point: np.ndarray, node_id: str) -> None:
        if not self.bounds.contains(point):
            return

        if self.is_leaf:
            if len(self.points) < self.max_points or self.depth >= self.max_depth:
                self.points.append((point, node_id))
                return
            else:
                self.subdivide()

        if self.children is not None:
            for child in self.children:
                if child.bounds.contains(point):
                    child.insert(point, node_id)
                    break


class Octree:
    """
    i-Octree spatial partitioning for memory organization.
    """

    def __init__(
        self,
        bounds: BoundingBox,
        max_points: int = 8,
        max_depth: int = 10,
    ):
        self.root = OctreeNode(
            bounds=bounds,
            max_points=max_points,
            max_depth=max_depth,
        )
        self._node_map: Dict[str, OctreeNode] = {}

    def insert(self, point: np.ndarray, node_id: str) -> None:
        self._insert_recursive(self.root, point, node_id)

    def _insert_recursive(self, node: OctreeNode, point: np.ndarray, node_id: str) -> None:
        if not node.bounds.contains(point):
            return

        if node.is_leaf:
            if len(node.points) < node.max_points or node.depth >= node.max_depth:
                node.points.append((point, node_id))
                self._node_map[node_id] = node
                return
            else:
                node.subdivide()

        if node.children is not None:
            for child in node.children:
                if child.bounds.contains(point):
                    self._insert_recursive(child, point, node_id)
                    break

    def query_range(self, query_bounds: BoundingBox) -> List[str]:
        results = []
        self._query_recursive(self.root, query_bounds, results)
        return results

    def _query_recursive(
        self, node: OctreeNode, query_bounds: BoundingBox, results: List[str]
    ) -> None:
        if not node.bounds.intersects(query_bounds):
            return

        if node.is_leaf:
            for point, node_id in node.points:
                if query_bounds.contains(point):
                    results.append(node_id)
        else:
            assert node.children is not None
            for child in node.children:
                self._query_recursive(child, query_bounds, results)

    def query_nearest(self, point: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        candidates = []
        self._collect_all(self.root, point, candidates)

        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

    def _collect_all(
        self, node: OctreeNode, query_point: np.ndarray, results: List[Tuple[str, float]]
    ) -> None:
        if node.is_leaf:
            for point, node_id in node.points:
                dist = float(np.linalg.norm(point - query_point))
                results.append((node_id, dist))
        else:
            assert node.children is not None
            for child in node.children:
                self._collect_all(child, query_point, results)

    def remove(self, node_id: str) -> bool:
        if node_id not in self._node_map:
            return False

        node = self._node_map[node_id]
        node.points = [(p, nid) for p, nid in node.points if nid != node_id]
        del self._node_map[node_id]
        return True

    def get_statistics(self) -> Dict[str, Any]:
        total_nodes = 0
        leaf_nodes = 0
        max_depth = 0
        total_points = 0

        def traverse(node: OctreeNode) -> None:
            nonlocal total_nodes, leaf_nodes, max_depth, total_points
            total_nodes += 1
            max_depth = max(max_depth, node.depth)

            if node.is_leaf:
                leaf_nodes += 1
                total_points += len(node.points)
            else:
                assert node.children is not None
                for child in node.children:
                    traverse(child)

        traverse(self.root)

        return {
            "total_nodes": total_nodes,
            "leaf_nodes": leaf_nodes,
            "max_depth": max_depth,
            "total_points": total_points,
        }


class M3NOPartitioner:
    """
    M3NO (Memory Mesh Network Organization) partitioner.

    Combines octree spatial partitioning with graph-based
    connectivity for efficient memory organization.
    """

    def __init__(
        self,
        dimension: int = 3,
        max_points_per_cell: int = 8,
        max_depth: int = 10,
        connection_radius: float = 1.0,
    ):
        self.dimension = dimension
        self.max_points_per_cell = max_points_per_cell
        self.max_depth = max_depth
        self.connection_radius = connection_radius
        self._octree: Optional[Octree] = None
        self._initialized = False
        self._graph: Dict[str, Set[str]] = defaultdict(set)
        self._point_map: Dict[str, np.ndarray] = {}

    def initialize(self, points: np.ndarray) -> None:
        if len(points) == 0:
            return

        min_bounds = np.min(points, axis=0) - 0.1
        max_bounds = np.max(points, axis=0) + 0.1

        bounds = BoundingBox(min_bounds, max_bounds)
        self._octree = Octree(
            bounds=bounds,
            max_points=self.max_points_per_cell,
            max_depth=self.max_depth,
        )
        self._initialized = True

    def add_point(self, point: np.ndarray, node_id: str) -> None:
        if not self._initialized or self._octree is None:
            self.initialize(np.array([point]))
            assert self._octree is not None
            self._octree.insert(point, node_id)
        else:
            self._octree.insert(point, node_id)

        self._point_map[node_id] = point
        self._update_connections(node_id, point)

    def _update_connections(self, node_id: str, point: np.ndarray) -> None:
        if self._octree is None:
            return
        neighbors = self._octree.query_nearest(point, k=10)
        for neighbor_id, distance in neighbors:
            if neighbor_id != node_id and distance <= self.connection_radius:
                self._graph[node_id].add(neighbor_id)
                self._graph[neighbor_id].add(node_id)

    def query_range(self, center: np.ndarray, radius: float) -> List[str]:
        if not self._initialized or self._octree is None:
            return []

        query_bounds = BoundingBox(
            center - radius,
            center + radius,
        )
        return self._octree.query_range(query_bounds)

    def query_nearest(self, point: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if not self._initialized or self._octree is None:
            return []

        return self._octree.query_nearest(point, k)

    def query_connected(self, node_id: str, max_depth: int = 2) -> List[str]:
        if node_id not in self._graph:
            return []

        visited = {node_id}
        current_level = {node_id}
        connected = []

        for depth in range(max_depth):
            next_level = set()
            for current_id in current_level:
                for neighbor_id in self._graph[current_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        connected.append(neighbor_id)
                        next_level.add(neighbor_id)
            current_level = next_level

        return connected

    def remove_point(self, node_id: str) -> bool:
        if not self._initialized or self._octree is None:
            return False

        success = self._octree.remove(node_id)
        if success:
            if node_id in self._point_map:
                del self._point_map[node_id]
            if node_id in self._graph:
                for neighbor_id in self._graph[node_id]:
                    self._graph[neighbor_id].discard(node_id)
                del self._graph[node_id]

        return success

    def is_initialized(self) -> bool:
        return self._initialized

    def get_statistics(self) -> Dict[str, Any]:
        if not self._initialized or self._octree is None:
            return {"initialized": False}

        stats = self._octree.get_statistics()
        stats["initialized"] = True
        stats["graph_nodes"] = len(self._graph)
        stats["graph_edges"] = sum(len(neighbors) for neighbors in self._graph.values()) // 2
        return stats


class BucketActor:
    def __init__(self, bucket_id: str, dimension: int = 3, persistence=None, consistency=None):
        self.bucket_id = bucket_id
        self._dimension = dimension
        self._body: Optional[Any] = None
        self._lock = threading.RLock()
        self._persistence = persistence
        self._consistency = consistency

    def _ensure_body(self) -> Any:
        if self._body is None:
            from tetrahedron_memory.core import GeoMemoryBody

            self._body = GeoMemoryBody(
                dimension=self._dimension,
                bucket_id=self.bucket_id,
                persistence=self._persistence,
                consistency=self._consistency,
            )
        return self._body

    def store(
        self,
        content: str,
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ) -> str:
        with self._lock:
            return self._ensure_body().store(
                content, labels=labels, metadata=metadata, weight=weight
            )

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            results = self._ensure_body().query(query_text, k=k)
            return [
                {
                    "node_id": r.node.id,
                    "content": r.node.content,
                    "distance": r.distance,
                    "persistence_score": r.persistence_score,
                }
                for r in results
            ]

    def associate(self, memory_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        with self._lock:
            results = self._ensure_body().associate(memory_id, max_depth=max_depth)
            return [
                {
                    "node_id": node.id,
                    "content": node.content,
                    "score": score,
                    "assoc_type": assoc_type,
                }
                for node, score, assoc_type in results
            ]

    def self_organize(self) -> Dict[str, Any]:
        with self._lock:
            return self._ensure_body().self_organize()

    def get_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            body = self._ensure_body()
            return {
                nid: {
                    "content": n.content,
                    "geometry": n.geometry.tolist(),
                    "weight": n.weight,
                    "labels": n.labels,
                }
                for nid, n in body._nodes.items()
            }

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return self._ensure_body().get_statistics()


class SpatialBucketRouter:
    def __init__(self, max_points_per_bucket: int = 1000, ghost_ttl: float = 3600.0):
        self._bucket_map: Dict[str, str] = {}
        self._bucket_bounds: Dict[str, BoundingBox] = {}
        self._bucket_sizes: Dict[str, int] = {}
        self._actors: Dict[str, BucketActor] = {}
        self._lock = threading.RLock()
        self._max_points_per_bucket = max_points_per_bucket
        self._next_bucket_idx = 0
        self._default_bounds = BoundingBox(np.array([-1.1, -1.1, -1.1]), np.array([1.1, 1.1, 1.1]))
        self._ghost_cells: Dict[str, Dict[str, GhostCell]] = defaultdict(dict)
        self._ghost_ttl = ghost_ttl
        self._boundary_width_ratio = 0.15

    def initialize(self, initial_bounds: Optional[BoundingBox] = None) -> None:
        bounds = initial_bounds or self._default_bounds
        self._auto_create_bucket(bounds)
        self._default_bounds = bounds

    def route_store(self, geometry: np.ndarray, content: str, **kwargs: Any) -> Tuple[str, str]:
        with self._lock:
            best_bid = self._find_best_bucket(geometry)
            if best_bid is None:
                best_bid = self._auto_create_bucket(self._default_bounds)

            if self._bucket_sizes.get(best_bid, 0) >= self._max_points_per_bucket:
                new_ids = self._split_bucket(best_bid)
                best_bid = self._find_best_bucket(geometry) or new_ids[0]

            actor = self._actors[best_bid]
            node_id = actor.store(content=content, **kwargs)
            self._bucket_map[node_id] = best_bid
            self._bucket_sizes[best_bid] = self._bucket_sizes.get(best_bid, 0) + 1
            return best_bid, node_id

    def route_query(self, geometry: np.ndarray, k: int = 5) -> List[str]:
        with self._lock:
            scored: List[Tuple[float, str]] = []
            for bid, bounds in self._bucket_bounds.items():
                center = bounds.center
                dist = float(np.linalg.norm(geometry - center))
                scored.append((dist, bid))
            scored.sort()
            return [bid for _, bid in scored[: max(3, (k // 5) + 1)]]

    def route_associate(self, geometry: np.ndarray, radius: float) -> List[str]:
        with self._lock:
            result = []
            for bid, bounds in self._bucket_bounds.items():
                center = bounds.center
                half_diag = float(np.linalg.norm(bounds.size / 2))
                dist = float(np.linalg.norm(geometry - center))
                if dist - half_diag <= radius:
                    result.append(bid)
            return result

    def get_bucket_for_node(self, node_id: str) -> Optional[str]:
        with self._lock:
            return self._bucket_map.get(node_id)

    def get_all_bucket_ids(self) -> List[str]:
        with self._lock:
            return list(self._actors.keys())

    def get_actor(self, bucket_id: str) -> Optional[BucketActor]:
        with self._lock:
            return self._actors.get(bucket_id)

    def cross_bucket_query(
        self, geometry: np.ndarray, query_text: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        candidate_bids = self.route_query(geometry, k)
        all_results: List[Dict[str, Any]] = []
        with self._lock:
            for bid in candidate_bids:
                actor = self._actors.get(bid)
                if actor is not None:
                    all_results.extend(actor.query(query_text, k=k))
        all_results.sort(key=lambda r: r.get("distance", float("inf")))
        return all_results[:k]

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            total_ghosts = sum(len(cells) for cells in self._ghost_cells.values())
            return {
                "total_buckets": len(self._actors),
                "total_nodes": sum(self._bucket_sizes.values()),
                "bucket_ids": list(self._actors.keys()),
                "bucket_sizes": dict(self._bucket_sizes),
                "ghost_cells": total_ghosts,
            }

    def _compute_boundary_zone(self, bucket_id: str) -> Optional[Tuple[BoundingBox, BoundingBox]]:
        bounds = self._bucket_bounds.get(bucket_id)
        if bounds is None:
            return None
        width = bounds.size * self._boundary_width_ratio
        inner = BoundingBox(bounds.min_bounds + width, bounds.max_bounds - width)
        outer = BoundingBox(bounds.min_bounds - width, bounds.max_bounds + width)
        return inner, outer

    def _populate_ghost_cells(self, bucket_id: str) -> int:
        result = self._compute_boundary_zone(bucket_id)
        if result is None:
            return 0
        _, outer_bounds = result
        source_bounds = self._bucket_bounds.get(bucket_id)
        if source_bounds is None:
            return 0

        created = 0
        for other_bid, other_bounds in self._bucket_bounds.items():
            if other_bid == bucket_id:
                continue
            if not other_bounds.intersects(outer_bounds):
                continue

            actor = self._actors.get(other_bid)
            if actor is None:
                continue
            snapshot = actor.get_snapshot()
            inner, _ = result
            for nid, data in snapshot.items():
                geom = np.array(data["geometry"])
                if source_bounds.contains(geom):
                    continue
                if inner.contains(geom):
                    continue
                if not outer_bounds.contains(geom):
                    continue

                content = data.get("content", "")
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
                ghost = GhostCell(
                    node_id=nid,
                    source_bucket_id=other_bid,
                    geometry=geom,
                    weight=data.get("weight", 1.0),
                    content_hash=content_hash,
                    labels=data.get("labels", []),
                    ttl=self._ghost_ttl,
                )
                if nid not in self._ghost_cells[bucket_id]:
                    created += 1
                self._ghost_cells[bucket_id][nid] = ghost

        return created

    def cross_bucket_associate(
        self, node_id: str, max_depth: int = 2, radius: float = 1.0
    ) -> List[Dict[str, Any]]:
        with self._lock:
            source_bid = self._bucket_map.get(node_id)
            if source_bid is None:
                return []

            source_actor = self._actors.get(source_bid)
            if source_actor is None:
                return []

            source_geom: Optional[np.ndarray] = None
            snapshot = source_actor.get_snapshot()
            if node_id in snapshot:
                source_geom = np.array(snapshot[node_id]["geometry"])

            results: Dict[str, Dict[str, Any]] = {}

            for bid, ghost_map in self._ghost_cells.items():
                if bid != source_bid:
                    continue
                for gid, ghost in ghost_map.items():
                    if ghost.is_expired:
                        continue
                    if source_geom is not None:
                        dist = float(np.linalg.norm(source_geom - ghost.geometry))
                        if dist > radius:
                            continue
                        score = 1.0 / (1.0 + dist)
                    else:
                        score = ghost.weight
                    results[gid] = {
                        "node_id": gid,
                        "source_bucket_id": ghost.source_bucket_id,
                        "geometry": ghost.geometry.tolist(),
                        "weight": ghost.weight,
                        "score": score,
                        "labels": ghost.labels,
                        "via": "ghost_cell",
                    }
                    ghost.touch()

            target_bids = self.route_associate(
                source_geom if source_geom is not None else np.zeros(3), radius
            )
            for bid in target_bids:
                if bid == source_bid:
                    continue
                actor = self._actors.get(bid)
                if actor is None:
                    continue
                assoc_results = actor.associate(node_id, max_depth=max_depth)
                for item in assoc_results:
                    rid = item["node_id"]
                    if rid not in results:
                        results[rid] = {
                            **item,
                            "source_bucket_id": bid,
                            "via": "actor_query",
                        }

            return sorted(results.values(), key=lambda x: x.get("score", 0.0), reverse=True)

    def update_ghost_cells(self, bucket_id: Optional[str] = None) -> int:
        with self._lock:
            total = 0
            targets = [bucket_id] if bucket_id else list(self._bucket_bounds.keys())
            for bid in targets:
                self._ghost_cells.pop(bid, None)
                total += self._populate_ghost_cells(bid)
            return total

    def prune_expired_ghosts(self) -> int:
        with self._lock:
            removed = 0
            for bid in list(self._ghost_cells.keys()):
                cells = self._ghost_cells[bid]
                expired = [nid for nid, gc in cells.items() if gc.is_expired]
                for nid in expired:
                    del cells[nid]
                    removed += 1
            return removed

    def get_ghost_cell_stats(self) -> Dict[str, Any]:
        with self._lock:
            per_bucket = {bid: len(cells) for bid, cells in self._ghost_cells.items()}
            expired = sum(
                1 for cells in self._ghost_cells.values() for gc in cells.values() if gc.is_expired
            )
            return {
                "total_ghost_cells": sum(per_bucket.values()),
                "per_bucket": per_bucket,
                "expired": expired,
            }

    def _find_best_bucket(self, geometry: np.ndarray) -> Optional[str]:
        best_bid: Optional[str] = None
        best_dist = float("inf")
        for bid, bounds in self._bucket_bounds.items():
            if bounds.contains(geometry):
                center = bounds.center
                dist = float(np.linalg.norm(geometry - center))
                if dist < best_dist:
                    best_dist = dist
                    best_bid = bid
        if best_bid is None:
            best_dist = float("inf")
            for bid, bounds in self._bucket_bounds.items():
                dist = float(np.linalg.norm(geometry - bounds.center))
                if dist < best_dist:
                    best_dist = dist
                    best_bid = bid
        return best_bid

    def _split_bucket(self, bucket_id: str) -> List[str]:
        bounds = self._bucket_bounds.get(bucket_id)
        actor = self._actors.get(bucket_id)
        if bounds is None or actor is None:
            return [bucket_id]

        snapshot = actor.get_snapshot()
        old_size = self._bucket_sizes.get(bucket_id, 0)
        if old_size == 0:
            return [bucket_id]

        geometries = []
        for nid, data in snapshot.items():
            geom = np.array(data["geometry"])
            geometries.append((nid, data, geom))

        axis_sizes = bounds.size
        split_axis = int(np.argmax(axis_sizes))
        coords = [g[2][split_axis] for g in geometries]
        coords.sort()
        median = coords[len(coords) // 2]

        min1 = bounds.min_bounds.copy()
        max1 = bounds.max_bounds.copy()
        max1[split_axis] = median
        bounds1 = BoundingBox(min1, max1)

        min2 = bounds.min_bounds.copy()
        min2[split_axis] = median
        max2 = bounds.max_bounds.copy()
        bounds2 = BoundingBox(min2, max2)

        bid1 = self._auto_create_bucket(bounds1)
        bid2 = self._auto_create_bucket(bounds2)

        actor1 = self._actors[bid1]
        actor2 = self._actors[bid2]
        size1 = 0
        size2 = 0

        for nid, data, geom in geometries:
            if geom[split_axis] <= median:
                actor1.store(
                    content=data["content"],
                    labels=data.get("labels"),
                    weight=data.get("weight", 1.0),
                )
                self._bucket_map[nid] = bid1
                size1 += 1
            else:
                actor2.store(
                    content=data["content"],
                    labels=data.get("labels"),
                    weight=data.get("weight", 1.0),
                )
                self._bucket_map[nid] = bid2
                size2 += 1

        self._bucket_sizes[bid1] = size1
        self._bucket_sizes[bid2] = size2

        del self._actors[bucket_id]
        del self._bucket_bounds[bucket_id]
        self._bucket_sizes.pop(bucket_id, None)
        unregister_bucket(bucket_id)

        return [bid1, bid2]

    def _auto_create_bucket(self, bounds: BoundingBox) -> str:
        bid = f"spatial_bucket_{self._next_bucket_idx}"
        self._next_bucket_idx += 1
        actor = BucketActor(bid)
        self._actors[bid] = actor
        self._bucket_bounds[bid] = bounds
        self._bucket_sizes[bid] = 0
        register_bucket(bid, actor)
        return bid


_global_bucket_registry: Dict[str, Any] = {}
_registry_lock = threading.RLock()


def register_bucket(bucket_id: str, actor: Any) -> None:
    with _registry_lock:
        _global_bucket_registry[bucket_id] = actor


def unregister_bucket(bucket_id: str) -> None:
    with _registry_lock:
        _global_bucket_registry.pop(bucket_id, None)


def get_all_buckets() -> Dict[str, Any]:
    with _registry_lock:
        return dict(_global_bucket_registry)


def global_coarse_grid_sync() -> Optional[Any]:
    from tetrahedron_memory.core import GeoMemoryBody

    with _registry_lock:
        buckets = dict(_global_bucket_registry)

    if not buckets:
        return None

    temp_body = GeoMemoryBody(dimension=3)
    for bucket_id, actor in buckets.items():
        snapshot = actor.get_snapshot()
        for nid, data in snapshot.items():
            temp_body.store(
                content=data["content"],
                labels=data.get("labels", []),
                weight=data.get("weight", 1.0),
            )

    return temp_body.get_persistence_diagram()


class TetraMemRayController:
    def __init__(self, num_buckets: int = 4, use_spatial_routing: bool = False):
        self.num_buckets = num_buckets
        self._actors: Dict[str, BucketActor] = {}
        self._ray_mode = False
        self._initialized = False
        self._lock = threading.RLock()
        self._use_spatial_routing = use_spatial_routing
        self._router: Optional[SpatialBucketRouter] = None

    def initialize(self) -> None:
        try:
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            self._ray_mode = True
        except ImportError:
            self._ray_mode = False

        with self._lock:
            if self._use_spatial_routing:
                self._router = SpatialBucketRouter()
                self._router.initialize()
                self._actors = self._router._actors
            else:
                for i in range(self.num_buckets):
                    bid = f"bucket_{i}"
                    if self._ray_mode:
                        try:
                            import ray

                            RayBucketActor = ray.remote(BucketActor)
                            actor = RayBucketActor.remote(bid)
                        except Exception:
                            actor = BucketActor(bid)
                    else:
                        actor = BucketActor(bid)
                    self._actors[bid] = actor
                    register_bucket(bid, actor)

        self._initialized = True

    def store_routed(self, content: str, **kwargs: Any) -> Tuple[str, str]:
        if not self._initialized:
            return "", ""
        if self._router is not None:
            from tetrahedron_memory.core import GeoMemoryBody

            tmp_body = GeoMemoryBody()
            geometry = tmp_body._text_to_geometry(content)
            return self._router.route_store(geometry, content, **kwargs)
        bid = list(self._actors.keys())[0] if self._actors else ""
        node_id = self.store(bid, content, **kwargs)
        return bid, node_id

    def query_routed(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self._initialized:
            return []
        if self._router is not None:
            from tetrahedron_memory.core import GeoMemoryBody

            tmp_body = GeoMemoryBody()
            geometry = tmp_body._text_to_geometry(query_text)
            return self._router.cross_bucket_query(geometry, query_text, k=k)
        all_results: List[Dict[str, Any]] = []
        for actor in self._actors.values():
            all_results.extend(actor.query(query_text, k=k))
        all_results.sort(key=lambda r: r.get("distance", float("inf")))
        return all_results[:k]

    def auto_balance(self) -> Dict[str, Any]:
        if self._router is None:
            return {"balanced": False, "reason": "no_spatial_routing"}
        stats = self._router.get_statistics()
        return {"balanced": True, "stats": stats}

    def store(self, bucket_id: str, content: str, **kwargs) -> str:
        actor = self._actors.get(bucket_id)
        if actor is None:
            return ""
        if self._ray_mode:
            try:
                import ray

                return ray.get(actor.store.remote(content, **kwargs))
            except Exception as e:
                import logging
                logging.getLogger("tetramem.ray").warning("Ray store fallback: %s", e)
        return actor.store(content, **kwargs)

    def query(self, bucket_id: str, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        actor = self._actors.get(bucket_id)
        if actor is None:
            return []
        if self._ray_mode:
            try:
                import ray

                return ray.get(actor.query.remote(query_text, k))
            except Exception as e:
                import logging
                logging.getLogger("tetramem.ray").warning("Ray query fallback: %s", e)
        return actor.query(query_text, k)

    def associate(self, bucket_id: str, memory_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        actor = self._actors.get(bucket_id)
        if actor is None:
            return []
        if self._ray_mode:
            try:
                import ray

                return ray.get(actor.associate.remote(memory_id, max_depth))
            except Exception as e:
                import logging
                logging.getLogger("tetramem.ray").warning("Ray associate fallback: %s", e)
        return actor.associate(memory_id, max_depth)

    def self_organize(self, bucket_id: str) -> Dict[str, Any]:
        actor = self._actors.get(bucket_id)
        if actor is None:
            return {"actions": 0, "reason": "unknown_bucket"}
        if self._ray_mode:
            try:
                import ray

                return ray.get(actor.self_organize.remote())
            except Exception as e:
                import logging
                logging.getLogger("tetramem.ray").warning("Ray self_organize fallback: %s", e)
        return actor.self_organize()

    def sync_all(self) -> Optional[Any]:
        return global_coarse_grid_sync()

    def shutdown(self) -> None:
        with self._lock:
            for bid in list(self._actors.keys()):
                unregister_bucket(bid)
            self._actors.clear()

        if self._ray_mode:
            try:
                import ray

                if ray.is_initialized():
                    ray.shutdown()
            except Exception as e:
                import logging
                logging.getLogger("tetramem.ray").warning("Ray shutdown error: %s", e)

        self._initialized = False
        self._ray_mode = False

    def is_initialized(self) -> bool:
        return self._initialized

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "ray_mode": self._ray_mode,
            "num_buckets": self.num_buckets,
            "active_buckets": list(self._actors.keys()),
        }
