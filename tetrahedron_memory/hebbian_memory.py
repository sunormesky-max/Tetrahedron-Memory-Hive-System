from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


class HebbianPathMemory:
    """
    Tracks frequently-used pulse propagation paths and reinforces them.

    Hebbian principle: "neurons that fire together, wire together"
    When a pulse path leads to a successful bridge or hits a high-weight node,
    the path segments are reinforced. Future pulses are biased toward these paths.

    This creates an emergent "fast pathway" network on top of the BCC lattice.
    """

    def __init__(
        self,
        max_paths: int = 500,
        decay: float = 0.98,
        reinforce_factor: float = 1.15,
        min_weight: float = 0.01,
    ):
        self._edges: Dict[Tuple[str, str], float] = defaultdict(float)
        self._traversal_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self._max_paths = max_paths
        self._decay = decay
        self._reinforce = reinforce_factor
        self._min_weight = min_weight
        self._success_count = 0
        self._total_decay_count = 0
        self._default_segment_weight = 0.5
        self._golden_threshold = 10
        self._golden_multiplier = 1.5

    def record_path(self, path: List[str], success: bool, strength: float):
        if len(path) < 2:
            return

        factor = self._reinforce if success else 1.0
        edge_strength = min(strength * factor / max(len(path) - 1, 1), 2.0)
        if edge_strength < self._default_segment_weight and success:
            edge_strength = min(self._default_segment_weight, 2.0)

        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            rev_key = (path[i + 1], path[i])
            if key in self._edges:
                self._edges[key] = min(self._edges[key] + edge_strength, 10.0)
                self._traversal_count[key] = self._traversal_count.get(key, 0) + 1
                if self._traversal_count[key] >= self._golden_threshold:
                    self._edges[key] = min(self._edges[key] * self._golden_multiplier, 10.0)
                    self._traversal_count[key] = 0
            elif rev_key in self._edges:
                self._edges[rev_key] = min(self._edges[rev_key] + edge_strength, 10.0)
                self._traversal_count[rev_key] = self._traversal_count.get(rev_key, 0) + 1
                if self._traversal_count[rev_key] >= self._golden_threshold:
                    self._edges[rev_key] = min(self._edges[rev_key] * self._golden_multiplier, 10.0)
                    self._traversal_count[rev_key] = 0
            else:
                self._edges[key] = min(edge_strength, 10.0)
                self._traversal_count[key] = 1

        if success:
            self._success_count += 1

        if len(self._edges) > self._max_paths:
            self._prune()

    def get_path_bias(self, from_id: str, to_id: str) -> float:
        w = self._edges.get((from_id, to_id), 0.0)
        if w == 0.0:
            w = self._edges.get((to_id, from_id), 0.0)
        return w

    def decay_all(self):
        to_remove = []
        for key in self._edges:
            self._edges[key] *= self._decay
            if self._edges[key] < self._min_weight:
                to_remove.append(key)
        for key in to_remove:
            del self._edges[key]
        self._total_decay_count += 1

    def _prune(self):
        sorted_edges = sorted(self._edges.items(), key=lambda x: x[1])
        to_remove = len(self._edges) - self._max_paths
        for i in range(min(to_remove, len(sorted_edges))):
            key = sorted_edges[i][0]
            del self._edges[key]
            self._traversal_count.pop(key, None)

    def path_quality_score(self, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0
        total_weight = 0.0
        for i in range(len(path) - 1):
            w = self.get_path_bias(path[i], path[i + 1])
            total_weight += w
        avg_weight = total_weight / max(len(path) - 1, 1)
        avg_traversal = 0
        count = 0
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            rev_key = (path[i + 1], path[i])
            tc = self._traversal_count.get(key, self._traversal_count.get(rev_key, 0))
            avg_traversal += tc
            count += 1
        if count > 0:
            avg_traversal /= count
        quality = avg_weight * (1.0 + min(avg_traversal, 20) * 0.05)
        return round(min(quality, 10.0), 4)

    def get_golden_paths(self, n: int = 10) -> List[Tuple[str, str, float, int]]:
        golden = []
        for (a, b), w in self._edges.items():
            tc = self._traversal_count.get((a, b), 0)
            if tc >= self._golden_threshold // 2 or w >= 3.0:
                golden.append((a[:8], b[:8], round(w, 4), tc))
        golden.sort(key=lambda x: -x[2])
        return golden[:n]

    def get_top_paths(self, n: int = 20) -> List[Tuple[str, str, float]]:
        sorted_edges = sorted(self._edges.items(), key=lambda x: -x[1])
        return [(k[0][:8], k[1][:8], round(v, 4)) for k, v in sorted_edges[:n]]

    def stats(self) -> Dict[str, Any]:
        weights = list(self._edges.values())
        golden_count = sum(1 for w in weights if w >= 3.0)
        total_traversals = sum(self._traversal_count.values())
        return {
            "total_path_segments": len(self._edges),
            "success_count": self._success_count,
            "decay_cycles": self._total_decay_count,
            "avg_path_weight": float(np.mean(weights)) if weights else 0.0,
            "max_path_weight": float(max(weights)) if weights else 0.0,
            "golden_paths": golden_count,
            "total_traversals": total_traversals,
            "avg_quality_score": round(float(np.mean(weights)) * (1.0 + min(total_traversals / max(len(weights), 1), 20) * 0.05), 4) if weights else 0.0,
        }
