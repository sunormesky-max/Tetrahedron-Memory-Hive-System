"""
GlobalPhaseTransition — system-wide topological restructuring.

When accumulated tension across the mesh exceeds a critical threshold,
triggers a global restructure (insight event):
  1. Identify high-tension clusters
  2. Merge clusters via edge contraction / abstraction
  3. Create bridging memories at tension boundaries
  4. Reset local tension scores

This is not a local dream walk — it is a system-wide topology collapse,
analogous to a phase transition or sudden insight.
"""

import hashlib
import logging
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("tetramem.phase_transition")


class PhaseTransitionDetector:
    __slots__ = (
        "tension_threshold",
        "min_cluster_size",
        "max_clusters_per_transition",
        "cooldown_seconds",
        "_last_transition_time",
        "_transition_count",
        "_tension_history",
        "_max_history",
    )

    def __init__(
        self,
        tension_threshold: float = 8.0,
        min_cluster_size: int = 3,
        max_clusters_per_transition: int = 5,
        cooldown_seconds: float = 600.0,
    ):
        self.tension_threshold = tension_threshold
        self.min_cluster_size = min_cluster_size
        self.max_clusters_per_transition = max_clusters_per_transition
        self.cooldown_seconds = cooldown_seconds
        self._last_transition_time = 0.0
        self._transition_count = 0
        self._tension_history = []
        self._max_history = 100

    def compute_global_tension(self, mesh) -> Tuple[float, Dict[str, float]]:
        tensions = {}
        tetrahedra = mesh.tetrahedra
        time_lambda = getattr(mesh, '_time_lambda', 0.001)

        for tid, tetra in tetrahedra.items():
            if "__system__" in tetra.labels:
                continue
            t_score = 0.0

            face_nbs = mesh._face_neighbors(tid)
            edge_nbs = mesh._edge_neighbors(tid)
            vertex_nbs = mesh._vertex_neighbors(tid) if hasattr(mesh, '_vertex_neighbors') else []
            total_nbs = len(face_nbs) + len(edge_nbs) + len(vertex_nbs)

            if total_nbs == 0:
                continue

            face_ratio = len(face_nbs) / max(total_nbs, 1)

            nb_weights = []
            nb_centroids = []
            for nid in set(face_nbs + edge_nbs):
                nt = mesh.get_tetrahedron(nid)
                if nt is not None:
                    nb_weights.append(nt.weight)
                    nb_centroids.append(nt.centroid)

            if nb_weights:
                avg_nw = sum(nb_weights) / len(nb_weights)
                if avg_nw > 1.0 and tetra.weight < avg_nw * 0.5:
                    t_score += 2.0 * (avg_nw - tetra.weight)
                weight_var = sum((w - avg_nw) ** 2 for w in nb_weights) / len(nb_weights)
                t_score += weight_var * 0.5
                density_imbalance = abs(len(nb_weights) - 6.0) / 6.0
                t_score += density_imbalance * 1.5

            if face_ratio < 0.3:
                t_score += (0.3 - face_ratio) * 3.0

            own_fil = tetra.filtration(time_lambda)
            if nb_centroids:
                own_c = tetra.centroid
                dists = [np.linalg.norm(own_c - nc) for nc in nb_centroids]
                avg_dist = sum(dists) / len(dists)
                dist_var = sum((d - avg_dist) ** 2 for d in dists) / len(dists)
                t_score += dist_var * 2.0
                spatial_isolation = avg_dist / max(dists) if dists else 0
                t_score += (1.0 - spatial_isolation) * 0.5

            nb_fils = []
            for nid in set(face_nbs + edge_nbs):
                nt = mesh.get_tetrahedron(nid)
                if nt is not None:
                    nb_fils.append(nt.filtration(time_lambda))
            if nb_fils:
                fil_gradient = abs(own_fil - sum(nb_fils) / len(nb_fils))
                t_score += fil_gradient * 0.3

            tensions[tid] = t_score

        global_tension = sum(tensions.values()) if tensions else 0.0
        self._tension_history.append((time.time(), global_tension))
        if len(self._tension_history) > self._max_history:
            self._tension_history = self._tension_history[-self._max_history // 2:]

        return global_tension, tensions

    def should_trigger(self, global_tension: float) -> bool:
        if time.time() - self._last_transition_time < self.cooldown_seconds:
            return False
        if global_tension < self.tension_threshold:
            return False
        if len(self._tension_history) >= 3:
            recent = [t for _, t in self._tension_history[-3:]]
            if recent[-1] < recent[0]:
                return False
        return True

    def identify_tension_clusters(
        self, tensions: Dict[str, float], mesh
    ) -> List[List[str]]:
        sorted_t = sorted(tensions.items(), key=lambda x: x[1], reverse=True)
        high_tension = {tid for tid, t in sorted_t if t > 1.0}
        if not high_tension:
            return []

        visited = set()
        clusters = []

        for seed_id, _ in sorted_t:
            if seed_id in visited or seed_id not in high_tension:
                continue
            cluster = []
            stack = [seed_id]
            while stack:
                tid = stack.pop()
                if tid in visited or tid not in high_tension:
                    continue
                visited.add(tid)
                cluster.append(tid)
                for method in (mesh._face_neighbors, mesh._edge_neighbors):
                    for nid in method(tid):
                        if nid not in visited and nid in high_tension:
                            stack.append(nid)
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        clusters.sort(key=len, reverse=True)
        return clusters[:self.max_clusters_per_transition]

    def execute_transition(
        self, mesh, tensions: Dict[str, float], clusters: List[List[str]]
    ) -> Dict[str, Any]:
        result = {
            "phase": "global_transition",
            "global_tension": sum(tensions.values()),
            "clusters_processed": 0,
            "merges": 0,
            "bridges_created": 0,
            "abstractions": 0,
            "timestamp": time.time(),
        }

        for cluster in clusters:
            result["clusters_processed"] += 1
            members = []
            for tid in cluster:
                t = mesh.get_tetrahedron(tid)
                if t is not None:
                    members.append(t)

            if len(members) < self.min_cluster_size:
                continue

            merge_pairs = self._find_merge_candidates(cluster, tensions, mesh)
            for tid_a, tid_b in merge_pairs[:3]:
                try:
                    mesh.edge_contraction(tid_a, tid_b)
                    result["merges"] += 1
                except Exception:
                    pass

            if len(cluster) >= 4:
                bridge_id = self._create_cluster_bridge(members, mesh)
                if bridge_id:
                    result["bridges_created"] += 1

            if len(members) >= 3:
                abs_id = self._create_abstraction(members, mesh)
                if abs_id:
                    result["abstractions"] += 1

        self._last_transition_time = time.time()
        self._transition_count += 1
        result["total_transitions"] = self._transition_count
        logger.info(
            "Phase transition #%d: tension=%.2f clusters=%d merges=%d bridges=%d",
            self._transition_count, result["global_tension"],
            result["clusters_processed"], result["merges"],
            result["bridges_created"],
        )
        return result

    def _find_merge_candidates(
        self, cluster: List[str], tensions: Dict[str, float], mesh
    ) -> List[Tuple[str, str]]:
        pairs = []
        seen = set()
        for tid in cluster:
            for nid in mesh._face_neighbors(tid):
                if nid in cluster:
                    key = tuple(sorted([tid, nid]))
                    if key not in seen:
                        seen.add(key)
                        combined_t = tensions.get(tid, 0) + tensions.get(nid, 0)
                        pairs.append((combined_t, tid, nid))
        pairs.sort(reverse=True)
        return [(a, b) for _, a, b in pairs]

    def _create_cluster_bridge(self, members: list, mesh) -> Optional[str]:
        contents = []
        labels_union = set()
        centroids = []
        total_weight = 0.0
        for t in members[:6]:
            if t.content:
                contents.append(t.content[:60])
            labels_union.update(t.labels)
            centroids.append(t.centroid)
            total_weight += t.weight

        if not contents:
            return None

        labels_union.discard("__dream__")
        labels_union.discard("__system__")
        bridge_content = "[insight:bridge:" + ",".join(list(labels_union)[:4]) + "] " + " | ".join(contents[:4])

        centroid = np.mean(centroids, axis=0)
        bridge_labels = list(labels_union)[:6] + ["__dream__", "__insight__"]

        seed = centroid + np.random.normal(0, 0.05, 3)
        tid = mesh.store(
            content=bridge_content,
            seed_point=seed,
            labels=bridge_labels,
            weight=min(total_weight / len(members), 5.0),
        )
        return tid

    def _create_abstraction(self, members: list, mesh) -> Optional[str]:
        labels_by_freq = defaultdict(int)
        total_weight = 0.0
        centroids = []
        for t in members:
            for l in t.labels:
                if l not in ("__dream__", "__system__", "__insight__"):
                    labels_by_freq[l] += 1
            total_weight += t.weight
            centroids.append(t.centroid)

        if not labels_by_freq:
            return None

        shared_labels = [l for l, c in sorted(labels_by_freq.items(), key=lambda x: -x[1]) if c >= 2][:4]
        content_parts = []
        for t in members[:4]:
            if t.content:
                content_parts.append(t.content[:50])

        abs_content = "[insight:abstract:" + ",".join(shared_labels) + "] Synthesis from " + str(len(members)) + " memories"
        if content_parts:
            abs_content += ": " + " // ".join(content_parts)

        centroid = np.mean(centroids, axis=0)
        abs_labels = shared_labels + ["__dream__", "__insight__"]

        seed = centroid + np.random.normal(0, 0.08, 3)
        tid = mesh.store(
            content=abs_content,
            seed_point=seed,
            labels=abs_labels,
            weight=min(total_weight / len(members) * 1.5, 6.0),
        )
        return tid

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_transitions": self._transition_count,
            "last_transition": self._last_transition_time,
            "tension_threshold": self.tension_threshold,
            "cooldown_active": time.time() - self._last_transition_time < self.cooldown_seconds,
            "history_size": len(self._tension_history),
        }

    def get_tension_trend(self) -> str:
        if len(self._tension_history) < 3:
            return "insufficient_data"
        recent = [t for _, t in self._tension_history[-5:]]
        if recent[-1] > recent[0] * 1.2:
            return "rising"
        elif recent[-1] < recent[0] * 0.8:
            return "falling"
        return "stable"
