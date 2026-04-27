"""
HoneycombPhaseTransition — phase transitions for BCC Honeycomb neural field.

Adapted from phase_transition.py to work with HoneycombNeuralField instead
of TetraMesh. Uses honeycomb node properties (activation, weight, position,
face_neighbors, edge_neighbors) instead of tetrahedra properties.

When accumulated tension across the honeycomb exceeds a critical threshold,
triggers a system-wide restructure (insight event):
  1. Identify high-tension clusters
  2. Create bridging memories at tension boundaries
  3. Create abstraction memories for related clusters
  4. Fire tension-sensing pulses to integrate discovered regions
"""

import hashlib
import logging
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("tetramem.phase_transition.honeycomb")


class HoneycombPhaseTransition:
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

    def compute_global_tension(self, honeycomb) -> Tuple[float, Dict[str, float]]:
        """
        Compute tension for each occupied node in the honeycomb.

        Tension factors (geometric + topological):
          1. Weight variance with neighbors (imbalanced integration priority)
          2. Connectivity imbalance (low face-ratio = topological isolation)
          3. Spatial isolation (centroid distance to neighbors)
          4. Activation gradient (sudden activation changes)
          5. Internal activity vs threshold ratio (PCNN firing pressure)
        """
        tensions = {}

        for nid, node in honeycomb._nodes.items():
            if not node.is_occupied or "__system__" in node.labels:
                continue

            t_score = 0.0

            all_nbs = list(node.face_neighbors) + list(node.edge_neighbors)
            if not all_nbs:
                continue

            face_ratio = len(node.face_neighbors) / max(len(all_nbs), 1)

            nb_weights = []
            nb_positions = []
            nb_activations = []
            for nid2 in all_nbs:
                nb = honeycomb._nodes.get(nid2)
                if nb and nb.is_occupied:
                    nb_weights.append(nb.weight)
                    nb_positions.append(nb.position)
                    nb_activations.append(nb.activation)

            if nb_weights:
                avg_nw = sum(nb_weights) / len(nb_weights)
                if avg_nw > 1.0 and node.weight < avg_nw * 0.5:
                    t_score += 2.0 * (avg_nw - node.weight)

                w_var = sum((w - avg_nw) ** 2 for w in nb_weights) / len(nb_weights)
                t_score += w_var * 0.5

                density_imbalance = abs(len(nb_weights) - 8.0) / 8.0
                t_score += density_imbalance * 1.5

                avg_act = sum(nb_activations) / len(nb_activations)
                act_gradient = abs(node.activation - avg_act)
                t_score += act_gradient * 1.0

            if face_ratio < 0.3:
                t_score += (0.3 - face_ratio) * 3.0

            if nb_positions:
                dists = [float(np.linalg.norm(node.position - p)) for p in nb_positions]
                avg_dist = sum(dists) / len(dists)
                dist_var = sum((d - avg_dist) ** 2 for d in dists) / len(dists)
                t_score += dist_var * 2.0
                spatial_isolation = avg_dist / max(dists) if dists else 0
                t_score += (1.0 - spatial_isolation) * 0.5

            if node.threshold > 0:
                firing_pressure = node.internal_activity / max(node.threshold, 0.01)
                if firing_pressure > 0.8:
                    t_score += (firing_pressure - 0.8) * 3.0

            tensions[nid] = t_score

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
            if len(recent) >= 3:
                vel = recent[-1] - recent[-2]
                prev_vel = recent[-2] - recent[-3]
                accel = vel - prev_vel
                if accel > self.tension_threshold * 0.3:
                    return True
        return True

    def identify_tension_clusters(
        self, tensions: Dict[str, float], honeycomb
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
                node = honeycomb._nodes.get(tid)
                if node:
                    for nid in node.face_neighbors + node.edge_neighbors:
                        if nid not in visited and nid in high_tension:
                            stack.append(nid)
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        clusters.sort(key=len, reverse=True)
        return clusters[:self.max_clusters_per_transition]

    def execute_transition(
        self, honeycomb, tensions: Dict[str, float], clusters: List[List[str]]
    ) -> Dict[str, Any]:
        result = {
            "phase": "global_transition",
            "global_tension": sum(tensions.values()),
            "clusters_processed": 0,
            "bridges_created": 0,
            "abstractions": 0,
            "tension_pulses_fired": 0,
            "timestamp": time.time(),
        }

        for cluster in clusters:
            result["clusters_processed"] += 1
            members = []
            for tid in cluster:
                node = honeycomb._nodes.get(tid)
                if node and node.is_occupied:
                    members.append(node)

            if len(members) < self.min_cluster_size:
                continue

            if len(cluster) >= 4:
                bridge_id = self._create_cluster_bridge(members, honeycomb)
                if bridge_id:
                    result["bridges_created"] += 1

            if len(members) >= 3:
                abs_id = self._create_abstraction(members, honeycomb)
                if abs_id:
                    result["abstractions"] += 1

            pulse_count = self._fire_tension_pulses(cluster, honeycomb)
            result["tension_pulses_fired"] += pulse_count

        self._last_transition_time = time.time()
        self._transition_count += 1
        result["total_transitions"] = self._transition_count
        logger.info(
            "Honeycomb phase transition #%d: tension=%.2f clusters=%d bridges=%d abs=%d",
            self._transition_count, result["global_tension"],
            result["clusters_processed"], result["bridges_created"],
            result["abstractions"],
        )
        return result

    def _create_cluster_bridge(self, members: list, honeycomb) -> Optional[str]:
        contents = []
        labels_union = set()
        positions = []
        total_weight = 0.0
        for m in members[:6]:
            if m.content:
                contents.append(m.content[:60])
            labels_union.update(m.labels)
            positions.append(m.position)
            total_weight += m.weight

        if not contents:
            return None

        labels_union.discard("__dream__")
        labels_union.discard("__system__")
        bridge_content = "[insight:bridge:" + ",".join(list(labels_union)[:4]) + "] " + " | ".join(contents[:4])

        centroid = np.mean(positions, axis=0)
        bridge_labels = list(labels_union)[:6] + ["__dream__", "__insight__"]

        tid = honeycomb.store(
            content=bridge_content,
            labels=bridge_labels,
            weight=min(total_weight / len(members), 5.0),
            metadata={"type": "phase_transition_bridge", "source_count": len(members)},
        )
        return tid

    def _create_abstraction(self, members: list, honeycomb) -> Optional[str]:
        labels_by_freq = defaultdict(int)
        total_weight = 0.0
        positions = []
        for m in members:
            for l in m.labels:
                if l not in ("__dream__", "__system__", "__insight__"):
                    labels_by_freq[l] += 1
            total_weight += m.weight
            positions.append(m.position)

        if not labels_by_freq:
            return None

        shared_labels = [l for l, c in sorted(labels_by_freq.items(), key=lambda x: -x[1]) if c >= 2][:4]
        content_parts = []
        for m in members[:4]:
            if m.content:
                content_parts.append(m.content[:50])

        abs_content = "[insight:abstract:" + ",".join(shared_labels) + "] Synthesis from " + str(len(members)) + " memories"
        if content_parts:
            abs_content += ": " + " // ".join(content_parts)

        abs_labels = shared_labels + ["__dream__", "__insight__"]

        tid = honeycomb.store(
            content=abs_content,
            labels=abs_labels,
            weight=min(total_weight / len(members) * 1.5, 6.0),
            metadata={"type": "phase_transition_abstraction", "source_count": len(members)},
        )
        return tid

    def _fire_tension_pulses(self, cluster: List[str], honeycomb) -> int:
        from .honeycomb_neural_field import PulseType
        count = 0
        for tid in cluster[:3]:
            node = honeycomb._nodes.get(tid)
            if node and node.is_occupied:
                honeycomb._emit_pulse(tid, strength=0.6, pulse_type=PulseType.TENSION_SENSING)
                count += 1
        return count

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

    @property
    def transition_count(self) -> int:
        return self._transition_count
