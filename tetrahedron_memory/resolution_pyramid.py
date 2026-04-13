"""
Resolution Pyramid — Multi-scale memory representation for TetraMem-XL.

Implements a hierarchical pyramid where:
  - Level 0: finest resolution (individual tetrahedra)
  - Level 1..N: coarser clusters (spatial + topological grouping)

Design principles:
  - Pyramid auto-builds from TetraMesh using spatial clustering
  - Queries route through pyramid: coarse match first, then drill down
  - Each pyramid node has a representative content (highest weight member)
  - Levels auto-update when mesh changes significantly
  - Topological connectivity is preserved across levels

Performance guarantees:
  - Level 0 query: O(n) — full scan
  - Pyramid query: O(n/L * k) where L = number of levels
  - Auto-route selects level based on k and density

Integration:
  - GeoMemoryBody.query() can use pyramid for speedup
  - Dream cycle uses coarse level for walk initialization
  - Emergence pressure uses coarse stats for anomaly detection
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("tetramem.pyramid")


@dataclass
class PyramidNode:
    node_id: str
    level: int
    centroid: np.ndarray
    member_tetra_ids: List[str] = field(default_factory=list)
    child_node_ids: List[str] = field(default_factory=list)
    parent_node_id: Optional[str] = None
    representative_content: str = ""
    representative_id: str = ""
    avg_weight: float = 1.0
    max_weight: float = 1.0
    total_weight: float = 1.0
    labels: List[str] = field(default_factory=list)
    label_distribution: Dict[str, int] = field(default_factory=dict)
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None
    tetra_count: int = 0
    vertex_count: int = 0
    last_updated: float = 0.0

    def contains_point(self, point: np.ndarray, margin: float = 0.1) -> bool:
        if self.bbox_min is None or self.bbox_max is None:
            return True
        expanded_min = self.bbox_min - margin
        expanded_max = self.bbox_max + margin
        return bool(np.all(point >= expanded_min) and np.all(point <= expanded_max))

    def similarity_score(self, query_point: np.ndarray) -> float:
        dist = float(np.linalg.norm(query_point - self.centroid))
        size_factor = max(1.0, np.log2(max(self.tetra_count, 1)))
        return self.max_weight * size_factor / (1.0 + dist)


class ResolutionPyramid:
    """
    Multi-scale pyramid over a TetraMesh.

    Builds hierarchical clusters using iterative spatial partitioning.
    Each level coarsens by grouping nearby tetrahedra.
    """

    def __init__(
        self,
        max_levels: int = 4,
        min_cluster_size: int = 3,
        max_cluster_size: int = 50,
        rebuild_interval: int = 100,
        coarsening_ratio: float = 0.4,
    ):
        self._max_levels = max_levels
        self._min_cluster = min_cluster_size
        self._max_cluster = max_cluster_size
        self._rebuild_interval = rebuild_interval
        self._coarsening_ratio = coarsening_ratio
        self._levels: Dict[int, Dict[str, PyramidNode]] = {}
        self._dirty = True
        self._last_tetra_count = 0
        self._total_rebuilds = 0
        self._last_rebuild_time: float = 0.0
        self._lock = threading.RLock()
        self._feedback_history: List[Dict[str, Any]] = []
        self._max_feedback = 20
        self._adaptive_max_levels = max_levels
        self._adaptive_coarsening = coarsening_ratio
        self._query_hit_rates: Dict[int, int] = defaultdict(int)
        self._query_totals: Dict[int, int] = defaultdict(int)
        self._last_query_level: int = 0

    @property
    def num_levels(self) -> int:
        return len(self._levels)

    def mark_dirty(self) -> None:
        with self._lock:
            self._dirty = True

    def build(self, mesh: Any) -> Dict[str, Any]:
        with self._lock:
            return self._build_locked(mesh)

    def _build_locked(self, mesh: Any) -> Dict[str, Any]:
        tetrahedra = mesh.tetrahedra
        n_tetra = len(tetrahedra)
        if n_tetra < self._min_cluster:
            self._levels = {}
            self._dirty = False
            return {"levels": 0, "nodes_per_level": {}, "reason": "too_few_tetra"}

        start = time.time()
        effective_max_levels = self._adaptive_max_levels
        effective_coarsening = self._adaptive_coarsening

        self._levels = {}
        self._levels[0] = self._build_level_0(mesh)

        for level in range(1, effective_max_levels):
            prev_nodes = self._levels[level - 1]
            if len(prev_nodes) < self._min_cluster * 2:
                break
            self._levels[level] = self._coarsen_level_adaptive(
                prev_nodes, level, effective_coarsening
            )

        self._dirty = False
        self._last_tetra_count = n_tetra
        self._total_rebuilds += 1
        self._last_rebuild_time = time.time()

        elapsed = time.time() - start
        nodes_per_level = {str(l): len(nodes) for l, nodes in self._levels.items()}
        logger.info(
            "Pyramid built: %d levels, %s nodes in %.3fs",
            len(self._levels),
            nodes_per_level,
            elapsed,
        )
        return {
            "levels": len(self._levels),
            "nodes_per_level": nodes_per_level,
            "build_time": elapsed,
        }

    def ensure_built(self, mesh: Any) -> None:
        with self._lock:
            n_tetra = len(mesh.tetrahedra)
            if self._dirty or abs(n_tetra - self._last_tetra_count) >= self._rebuild_interval:
                self._build_locked(mesh)

    def query(
        self,
        query_point: np.ndarray,
        k: int = 5,
        level: int = -1,
    ) -> List[Tuple[str, float]]:
        with self._lock:
            if not self._levels:
                return []

            target_level = level if level >= 0 else self._auto_select_level(k)
            if target_level not in self._levels:
                target_level = max(self._levels.keys())

            self._last_query_level = target_level

            if target_level == 0:
                return self._query_level_0(query_point, k)

            return self._query_coarse(query_point, k, target_level)

    def auto_route(
        self,
        query_point: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        with self._lock:
            if not self._levels:
                return []

            coarse_level = max(self._levels.keys())
            coarse_results = self._query_coarse(query_point, max(k, 10), coarse_level)

            if not coarse_results:
                return self._query_level_0(query_point, k)

            candidate_ids: Set[str] = set()
            for node_id, _ in coarse_results:
                node = self._levels[coarse_level].get(node_id)
                if node:
                    candidate_ids.update(node.member_tetra_ids)

            if not candidate_ids:
                return self._query_level_0(query_point, k)

            scored = []
            for tid in candidate_ids:
                tetra = None
                for lev in range(len(self._levels)):
                    n0 = self._levels.get(0, {}).get(tid)
                    if n0:
                        tetra_node = n0
                        break
                else:
                    for nid, n in self._levels.get(0, {}).items():
                        if tid in n.member_tetra_ids:
                            tetra_node = n
                            break
                    else:
                        continue

                dist = float(np.linalg.norm(query_point - tetra_node.centroid))
                scored.append((tid, dist))

            scored.sort(key=lambda x: x[1])
            return scored[:k]

    def get_level_stats(self, level: int) -> Dict[str, Any]:
        with self._lock:
            nodes = self._levels.get(level, {})
            if not nodes:
                return {"level": level, "nodes": 0}

            weights = [n.avg_weight for n in nodes.values()]
            sizes = [n.tetra_count for n in nodes.values()]

            return {
                "level": level,
                "nodes": len(nodes),
                "avg_weight": float(np.mean(weights)),
                "avg_cluster_size": float(np.mean(sizes)),
                "max_cluster_size": max(sizes),
                "min_cluster_size": min(sizes),
            }

    def get_all_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "num_levels": len(self._levels),
                "total_rebuilds": self._total_rebuilds,
                "last_rebuild_time": self._last_rebuild_time,
                "levels": {str(l): self.get_level_stats(l) for l in self._levels},
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "num_levels": len(self._levels),
                "total_rebuilds": self._total_rebuilds,
                "dirty": self._dirty,
                "last_tetra_count": self._last_tetra_count,
            }

    def _build_level_0(self, mesh: Any) -> Dict[str, PyramidNode]:
        nodes = {}
        tetrahedra = mesh.tetrahedra

        for tid, tetra in tetrahedra.items():
            label_dist: Dict[str, int] = {}
            for lbl in tetra.labels:
                label_dist[lbl] = label_dist.get(lbl, 0) + 1

            node = PyramidNode(
                node_id=tid,
                level=0,
                centroid=tetra.centroid.copy(),
                member_tetra_ids=[tid],
                representative_content=tetra.content,
                representative_id=tid,
                avg_weight=tetra.weight,
                max_weight=tetra.weight,
                total_weight=tetra.weight,
                labels=list(tetra.labels),
                label_distribution=label_dist,
                bbox_min=tetra.centroid.copy(),
                bbox_max=tetra.centroid.copy(),
                tetra_count=1,
                vertex_count=len(tetra.vertex_indices),
                last_updated=time.time(),
            )
            nodes[tid] = node

        return nodes

    def _coarsen_level_adaptive(
        self, prev_nodes: Dict[str, PyramidNode], level: int, coarsening_ratio: float
    ) -> Dict[str, PyramidNode]:
        if not prev_nodes:
            return {}

        node_list = list(prev_nodes.values())
        n = len(node_list)
        target_clusters = max(self._min_cluster, int(n * coarsening_ratio))

        if target_clusters >= n:
            return {}

        centroids = np.array([nd.centroid for nd in node_list])

        assignments = self._spatial_cluster(centroids, target_clusters)

        clusters: Dict[int, List[int]] = {}
        for idx, cluster_id in enumerate(assignments):
            clusters.setdefault(int(cluster_id), []).append(idx)

        coarsened: Dict[str, PyramidNode] = {}
        for cluster_id, member_indices in clusters.items():
            if not member_indices:
                continue

            member_nodes = [node_list[i] for i in member_indices]
            coarsened_node = self._merge_nodes(member_nodes, level, cluster_id)
            coarsened[coarsened_node.node_id] = coarsened_node

        return coarsened

    def _spatial_cluster(self, centroids: np.ndarray, k: int) -> np.ndarray:
        n = len(centroids)
        if n <= k:
            return np.arange(n)

        indices = np.random.choice(n, size=k, replace=False)
        centers = centroids[indices].copy()

        for _ in range(10):
            dists = np.linalg.norm(centroids[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for c in range(k):
                mask = assignments == c
                if np.any(mask):
                    new_centers[c] = centroids[mask].mean(axis=0)
                else:
                    new_centers[c] = centers[c]

            if np.allclose(centers, new_centers, atol=1e-8):
                break
            centers = new_centers

        return assignments

    def _merge_nodes(self, nodes: List[PyramidNode], level: int, cluster_id: int) -> PyramidNode:
        all_member_ids: List[str] = []
        all_child_ids: List[str] = []
        total_w = 0.0
        max_w = 0.0
        best_content = ""
        best_id = ""
        best_w = -1.0
        all_labels: Dict[str, int] = {}
        centroid_sum = np.zeros(3)
        bbox_min = np.full(3, np.inf)
        bbox_max = np.full(3, -np.inf)
        total_tetra = 0
        total_vertex = 0

        for nd in nodes:
            all_member_ids.extend(nd.member_tetra_ids)
            all_child_ids.append(nd.node_id)
            total_w += nd.total_weight
            if nd.max_weight > max_w:
                max_w = nd.max_weight
            if nd.max_weight > best_w:
                best_w = nd.max_weight
                best_content = nd.representative_content
                best_id = nd.representative_id
            for lbl, cnt in nd.label_distribution.items():
                all_labels[lbl] = all_labels.get(lbl, 0) + cnt
            centroid_sum += nd.centroid * nd.tetra_count
            if nd.bbox_min is not None:
                bbox_min = np.minimum(bbox_min, nd.bbox_min)
            if nd.bbox_max is not None:
                bbox_max = np.maximum(bbox_max, nd.bbox_max)
            total_tetra += nd.tetra_count
            total_vertex += nd.vertex_count

        centroid = centroid_sum / max(total_tetra, 1)
        avg_w = total_w / max(len(all_member_ids), 1)

        top_labels = sorted(all_labels.items(), key=lambda x: x[1], reverse=True)[:8]
        top_label_names = [lbl for lbl, _ in top_labels]

        import hashlib

        nid = hashlib.sha256(
            ("pyramid_L" + str(level) + "_C" + str(cluster_id) + str(time.time())).encode()
        ).hexdigest()[:12]

        return PyramidNode(
            node_id=nid,
            level=level,
            centroid=centroid,
            member_tetra_ids=all_member_ids,
            child_node_ids=all_child_ids,
            representative_content=best_content,
            representative_id=best_id,
            avg_weight=avg_w,
            max_weight=max_w,
            total_weight=total_w,
            labels=top_label_names,
            label_distribution=all_labels,
            bbox_min=bbox_min if np.all(np.isfinite(bbox_min)) else None,
            bbox_max=bbox_max if np.all(np.isfinite(bbox_max)) else None,
            tetra_count=total_tetra,
            vertex_count=total_vertex,
            last_updated=time.time(),
        )

    def _query_level_0(self, query_point: np.ndarray, k: int) -> List[Tuple[str, float]]:
        nodes = self._levels.get(0, {})
        if not nodes:
            return []

        scored = []
        for nid, node in nodes.items():
            dist = float(np.linalg.norm(query_point - node.centroid))
            scored.append((nid, dist))

        scored.sort(key=lambda x: x[1])
        return scored[:k]

    def _query_coarse(self, query_point: np.ndarray, k: int, level: int) -> List[Tuple[str, float]]:
        nodes = self._levels.get(level, {})
        if not nodes:
            return []

        candidates = []
        for nid, node in nodes.items():
            if node.bbox_min is not None and not node.contains_point(query_point, margin=0.5):
                score = node.similarity_score(query_point) * 0.3
                candidates.append((nid, 1.0 / (score + 1e-12)))
                continue
            score = node.similarity_score(query_point)
            candidates.append((nid, 1.0 / (score + 1e-12)))

        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

    def _auto_select_level(self, k: int) -> int:
        if not self._levels:
            return 0

        max_level = max(self._levels.keys())
        if max_level == 0:
            return 0

        for level in range(max_level, 0, -1):
            n_nodes = len(self._levels.get(level, {}))
            if n_nodes >= k * 2:
                return level

        return 1 if 1 in self._levels else 0

    def record_dream_feedback(
        self, entropy_delta: float, dreams_created: int, dreams_reintegrated: int
    ) -> None:
        """
        Record feedback from a dream cycle to enable adaptive level adjustment.
        This is the closed-loop feedback: dream effectiveness -> pyramid adaptation.
        """
        with self._lock:
            feedback = {
                "timestamp": time.time(),
                "entropy_delta": entropy_delta,
                "dreams_created": dreams_created,
                "dreams_reintegrated": dreams_reintegrated,
                "num_levels": len(self._levels),
                "coarsening": self._adaptive_coarsening,
            }
            self._feedback_history.append(feedback)
            if len(self._feedback_history) > self._max_feedback:
                self._feedback_history = self._feedback_history[-self._max_feedback :]

            if len(self._feedback_history) >= 3:
                self._adapt_parameters()

    def record_query_feedback(self, level: int, hit: bool) -> None:
        with self._lock:
            self._query_totals[level] = self._query_totals.get(level, 0) + 1
            if hit:
                self._query_hit_rates[level] = self._query_hit_rates.get(level, 0) + 1

    def _adapt_parameters(self) -> None:
        """
        Adapt pyramid parameters based on accumulated feedback from dream cycles and queries.

        Rules:
          1. If recent dream entropy_delta is consistently positive (>0.1 avg), the pyramid
             is well-structured — maintain or increase granularity.
          2. If recent dream entropy_delta is consistently negative or near-zero, the pyramid
             may be too coarse — increase max_levels or decrease coarsening_ratio.
          3. If query hit rate at coarse levels is low (<30%), increase levels for finer routing.
          4. If query hit rate at coarse levels is high (>70%), can afford coarser pyramid.
        """
        recent = self._feedback_history[-5:]
        nonzero = [f["entropy_delta"] for f in recent if f["entropy_delta"] != 0]
        avg_delta = float(np.mean(nonzero)) if nonzero else 0.0

        if avg_delta > 0.1 and self._adaptive_max_levels < 6:
            self._adaptive_max_levels = min(6, self._adaptive_max_levels + 1)
            self._adaptive_coarsening = max(0.25, self._adaptive_coarsening - 0.02)
        elif avg_delta <= 0.02 and self._adaptive_coarsening < 0.5:
            self._adaptive_coarsening = min(0.5, self._adaptive_coarsening + 0.03)

        worst_hit_rate = 1.0
        best_hit_rate = 0.0
        for level in range(1, max(self._levels.keys()) + 1) if self._levels else []:
            total = self._query_totals.get(level, 0)
            hits = self._query_hit_rates.get(level, 0)
            if total > 10:
                hit_rate = hits / total
                worst_hit_rate = min(worst_hit_rate, hit_rate)
                best_hit_rate = max(best_hit_rate, hit_rate)

        if worst_hit_rate < 0.3 and self._adaptive_max_levels < 6:
            self._adaptive_max_levels = min(6, self._adaptive_max_levels + 1)
        elif best_hit_rate > 0.7 and self._adaptive_max_levels > 2:
            self._adaptive_max_levels = max(2, self._adaptive_max_levels - 1)

    def get_adaptive_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "adaptive_max_levels": self._adaptive_max_levels,
                "adaptive_coarsening": self._adaptive_coarsening,
                "feedback_count": len(self._feedback_history),
                "query_hit_rates": {
                    str(l): self._query_hit_rates.get(l, 0) / max(self._query_totals.get(l, 1), 1)
                    for l in self._query_totals
                },
                "recent_feedback": self._feedback_history[-3:] if self._feedback_history else [],
            }
