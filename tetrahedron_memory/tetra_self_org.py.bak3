"""
TetraSelfOrganizer — Self-organization loop for TetraMesh.

Per the Grok production specification and TetraMem-XL core principles:
  - NO forgetting/deletion — all memories are eternal
  - Low persistence + low heat → integration catalyst (weight boost)
  - Long-lived H2 → insert repulsion point (cave growth)
  - High overlap low conflict → edge contraction (merge)
  - Persistent entropy drives integration decisions

Converges to steady state by tracking action count per cycle
and stopping when below a threshold.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .persistent_entropy import EntropyTracker, compute_persistent_entropy

logger = logging.getLogger("tetramem.selforg")


class TetraSelfOrganizer:
    def __init__(
        self,
        mesh: Any,
        h2_threshold: float = 1.0,
        merge_overlap_threshold: float = 0.3,
        integrate_persistence_threshold: float = 0.1,
        integrate_heat_threshold: float = 0.2,
        integration_strength: float = 1.0,
        entropy_trigger_ratio: float = 1.3,
        max_iterations: int = 10,
        convergence_threshold: int = 2,
    ):
        self.mesh = mesh
        self.h2_threshold = h2_threshold
        self.merge_overlap_threshold = merge_overlap_threshold
        self.integrate_persistence_threshold = integrate_persistence_threshold
        self.integrate_heat_threshold = integrate_heat_threshold
        self.integration_strength = integration_strength
        self.entropy_trigger_ratio = entropy_trigger_ratio
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self._total_actions = 0
        self._entropy_tracker = EntropyTracker()
        self._entropy_convergence_window = 3
        self._entropy_convergence_tolerance = 0.05

    def run(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "iterations": 0,
            "total_actions": 0,
            "cave_growths": 0,
            "merges": 0,
            "integrations": 0,
            "converged": False,
            "entropy_before": 0.0,
            "entropy_after": 0.0,
            "entropy_delta": 0.0,
        }

        st = self.mesh.compute_ph()
        if st is not None:
            entropy_before = compute_persistent_entropy(st)
            self._entropy_tracker.record(entropy_before)
            stats["entropy_before"] = entropy_before

        for iteration in range(self.max_iterations):
            cycle_actions = self._run_one_cycle(stats)
            stats["iterations"] = iteration + 1
            stats["total_actions"] += cycle_actions

            if cycle_actions < self.convergence_threshold:
                stats["converged"] = True
                stats["convergence_reason"] = "low_actions"
                break

            if self._check_entropy_convergence():
                stats["converged"] = True
                stats["convergence_reason"] = "entropy_stable"
                break

        st = self.mesh.compute_ph()
        if st is not None:
            entropy_after = compute_persistent_entropy(st)
            self._entropy_tracker.record(entropy_after)
            stats["entropy_after"] = entropy_after
            if stats["entropy_before"] > 0:
                stats["entropy_delta"] = (
                    stats["entropy_before"] - entropy_after
                ) / stats["entropy_before"]

        self._total_actions += stats["total_actions"]
        return stats

    def _run_one_cycle(self, stats: Dict[str, Any]) -> int:
        actions = 0

        st = self.mesh.compute_ph()
        if st is None:
            return 0

        h2_intervals = st.persistence_intervals_in_dimension(2)
        h0_intervals = st.persistence_intervals_in_dimension(0)

        for birth, death in h2_intervals:
            if death - birth > self.h2_threshold:
                n = self._cave_growth(birth, death)
                stats["cave_growths"] += n
                actions += n

        merge_count = self._detect_and_merge(h0_intervals)
        stats["merges"] += merge_count
        actions += merge_count

        integrate_count = self._detect_and_integrate(h0_intervals)
        stats["integrations"] += integrate_count
        actions += integrate_count

        return actions

    def _cave_growth(self, birth: float, death: float) -> int:
        import hashlib

        tetrahedra = self.mesh.tetrahedra
        if len(tetrahedra) < 4:
            return 0

        centroid = np.zeros(3)
        count = 0
        for t in tetrahedra.values():
            centroid += t.centroid
            count += 1
        centroid /= count

        direction = centroid / (np.linalg.norm(centroid) + 1e-12)
        repulsion_point = centroid + direction * 0.5

        self._repel_nearby_vertices(repulsion_point, radius=0.3, strength=0.05)

        self.mesh.store(
            content="__cave_repulsion__",
            seed_point=repulsion_point,
            labels=["__system__", "__cave__"],
            metadata={"type": "cave_repulsion", "birth": float(birth), "death": float(death)},
            weight=8.0,
        )
        return 1

    def _repel_nearby_vertices(self, repulsion_point: np.ndarray, radius: float = 0.3, strength: float = 0.05) -> int:
        repelled = 0
        with self.mesh._lock:
            vertices = self.mesh.vertices
            for i, v in enumerate(vertices):
                diff = v - repulsion_point
                dist = float(np.linalg.norm(diff))
                if 1e-12 < dist < radius:
                    push = diff / dist * strength * (1.0 - dist / radius)
                    vertices[i] = v + push
                    repelled += 1
        return repelled

    def _detect_and_merge(self, h0_intervals: np.ndarray) -> int:
        if len(h0_intervals) == 0:
            return 0

        short_intervals = [
            (b, d) for b, d in h0_intervals if (d - b) < self.merge_overlap_threshold
        ]
        if not short_intervals:
            return 0

        tetrahedra = list(self.mesh.tetrahedra.values())
        if len(tetrahedra) < 2:
            return 0

        merges = 0
        face_pairs = self._find_face_connected_pairs()
        for t1_id, t2_id in face_pairs[:3]:
            t1 = self.mesh.get_tetrahedron(t1_id)
            t2 = self.mesh.get_tetrahedron(t2_id)
            if t1 is None or t2 is None:
                continue
            if "__system__" in t1.labels or "__system__" in t2.labels:
                continue

            v1 = set(t1.vertex_indices)
            v2 = set(t2.vertex_indices)
            overlap = len(v1 & v2) / len(v1 | v2)

            if overlap >= self.merge_overlap_threshold:
                merged = self.mesh.edge_contraction(t1_id, t2_id)
                if merged:
                    merges += 1

        return merges

    def _detect_and_integrate(self, h0_intervals: np.ndarray) -> int:
        if len(h0_intervals) == 0:
            return 0

        to_integrate = []
        with self.mesh._lock:
            for tid, tetra in self.mesh.tetrahedra.items():
                if "__system__" in tetra.labels:
                    continue
                heat = tetra.weight / (tetra.init_weight + 1e-6)
                if heat < self.integrate_heat_threshold:
                    to_integrate.append(tid)

            for tid in to_integrate[:5]:
                tetra = self.mesh.get_tetrahedron(tid)
                if tetra is not None:
                    tetra.catalyze_integration(self.integration_strength)

        return len(to_integrate[:5])

    def _find_face_connected_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        seen = set()
        with self.mesh._lock:
            for tid in list(self.mesh.tetrahedra.keys()):
                neighbors = self.mesh._face_neighbors(tid)
                for nid in neighbors:
                    key = tuple(sorted([tid, nid]))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((tid, nid))
        return pairs

    def _check_entropy_convergence(self) -> bool:
        history = self._entropy_tracker._history
        if len(history) < self._entropy_convergence_window:
            return False
        recent = history[-self._entropy_convergence_window:]
        if recent[0] <= 0:
            return False
        variation = float(np.std(recent)) / (float(np.mean(recent)) + 1e-12)
        return variation < self._entropy_convergence_tolerance

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_actions": self._total_actions,
            "h2_threshold": self.h2_threshold,
            "merge_threshold": self.merge_overlap_threshold,
            "integrate_heat_threshold": self.integrate_heat_threshold,
            "entropy": self._entropy_tracker.get_summary(),
        }
