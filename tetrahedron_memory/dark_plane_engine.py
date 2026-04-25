"""
Dark Plane Engine — Energy Landscape Model for Memory Depth

Mathematical Foundation
======================
Each memory node occupies a position in an energy landscape.
The "depth" of a node determines which dark plane it belongs to.

Energy Landscape:
    Internal Energy:  U(n) = weight(n) * activation(n)
    Neighborhood Entropy:  S(n) = -sum(p_i * ln(p_i)) over neighbor label distribution
    Free Energy:  F(n) = U(n) - T * S(n)    (T = system temperature)
    Well Depth:  D(n) = -F(n)               (higher = deeper in landscape)

Plane Assignment:
    Surface:  D < tau_1          (shallow well, easy to excite)
    Shallow:  tau_1 <= D < tau_2 (medium well)
    Deep:     tau_2 <= D < tau_3 (deep well, needs energy to awaken)
    Abyss:    D >= tau_3         (very deep, near-permanent storage)

Transitions:
    Descent (surface -> abyss):  spontaneous, governed by decay dynamics
    Ascent  (abyss -> surface):  requires energy input, modeled as quantum tunneling

Tunneling Probability:
    P_awaken(n) = exp(-2 * kappa * (D_target - D_current))
    where kappa is the tunneling barrier coefficient

Temperature:
    T = T_base * (1 + alpha * stress_level) * circadian_factor
    - Higher temperature = more entropy = nodes stay shallower
    - Lower temperature = less entropy = nodes sink deeper

Boltzmann Redistribution (every flow cycle):
    P(plane_i | node) = exp(-D(n) / T) / Z
    where Z is the partition function over all planes
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField

PLANE_ORDER = ("surface", "shallow", "deep", "abyss")
PLANE_INDEX = {p: i for i, p in enumerate(PLANE_ORDER)}


class DarkPlaneEngine:
    __slots__ = (
        "_field",
        "_temperature",
        "_base_temperature",
        "_tunneling_kappa",
        "_thresholds",
        "_plane_assignment",
        "_plane_cache_time",
        "_plane_cache_ttl",
        "_flow_count",
        "_total_transitions",
        "_total_reawakenings",
        "_descent_count",
        "_ascent_count",
        "_energy_history",
        "_max_history",
    )

    def __init__(self, field: HoneycombNeuralField, config: Optional[Dict] = None):
        cfg = config or {}
        self._field = field
        self._base_temperature = cfg.get("base_temperature", 1.0)
        self._temperature = self._base_temperature
        self._tunneling_kappa = cfg.get("tunneling_kappa", 0.5)
        self._thresholds = cfg.get("thresholds", {
            "surface_shallow": 0.0,
            "shallow_deep": 0.5,
            "deep_abyss": 1.0,
        })
        self._plane_assignment: Dict[str, str] = {}
        self._plane_cache_time: float = 0.0
        self._plane_cache_ttl: float = 60.0
        self._flow_count: int = 0
        self._total_transitions: int = 0
        self._total_reawakenings: int = 0
        self._descent_count: int = 0
        self._ascent_count: int = 0
        self._energy_history: List[Dict] = []
        self._max_history: int = 200

    def compute_internal_energy(self, weight: float, activation: float) -> float:
        raw = weight * activation
        return raw / (1.0 + raw / 10.0)

    def compute_neighborhood_entropy(self, node_id: str) -> float:
        field = self._field
        node = field._nodes.get(node_id)
        if node is None:
            return 0.0

        label_counts: Counter = Counter()
        total = 0

        for nnid in node.face_neighbors[:8]:
            nn = field._nodes.get(nnid)
            if nn and nn.is_occupied:
                for lbl in nn.labels:
                    if not lbl.startswith("__"):
                        label_counts[lbl] += 1
                        total += 1

        for nnid in node.edge_neighbors[:4]:
            nn = field._nodes.get(nnid)
            if nn and nn.is_occupied:
                for lbl in nn.labels:
                    if not lbl.startswith("__"):
                        label_counts[lbl] += 1
                        total += 1

        if total == 0:
            return 0.0

        entropy = 0.0
        for count in label_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)

        return entropy

    def compute_free_energy(self, node_id: str) -> float:
        node = self._field._nodes.get(node_id)
        if node is None or not node.is_occupied:
            return 0.0

        U = self.compute_internal_energy(node.weight, node.activation)
        S = self.compute_neighborhood_entropy(node_id)
        F = U - self._temperature * S
        return F

    def compute_well_depth(self, node_id: str) -> float:
        F = self.compute_free_energy(node_id)
        return -F

    def assign_plane(self, node_id: str) -> str:
        if node_id in self._plane_assignment:
            cached = self._plane_assignment[node_id]
            return cached

        D = self.compute_well_depth(node_id)
        tau = self._thresholds

        if D >= tau["deep_abyss"]:
            plane = "abyss"
        elif D >= tau["shallow_deep"]:
            plane = "deep"
        elif D >= tau["surface_shallow"]:
            plane = "shallow"
        else:
            plane = "surface"

        return plane

    def tunneling_probability(self, from_depth: float, to_depth: float) -> float:
        delta = to_depth - from_depth
        if delta <= 0:
            return 1.0
        return math.exp(-2.0 * self._tunneling_kappa * delta)

    def update_temperature(self, stress_level: float = 0.0, circadian_phase: str = "work"):
        circadian_factor = 1.0 if circadian_phase == "work" else 0.7
        self._temperature = self._base_temperature * (1.0 + 0.5 * stress_level) * circadian_factor

    def run_flow_cycle(self) -> Dict:
        field = self._field
        now = time.time()

        transitions = 0
        reawakenings = 0
        descents = 0
        ascents = 0

        plane_distribution = {"surface": 0, "shallow": 0, "deep": 0, "abyss": 0}
        depth_values = []
        free_energy_values = []

        occupied = list(field._occupied_ids)
        if not occupied:
            return {
                "transitions": 0, "reawakenings": 0,
                "plane_distribution": plane_distribution,
                "temperature": self._temperature,
            }

        for nid in occupied:
            node = field._nodes.get(nid)
            if not node or not node.is_occupied:
                continue

            old_plane = self._plane_assignment.get(nid, "surface")
            D = self.compute_well_depth(nid)
            tau = self._thresholds

            F = self.compute_free_energy(nid)
            free_energy_values.append(F)
            depth_values.append(D)

            if D >= tau["deep_abyss"]:
                new_plane = "abyss"
            elif D >= tau["shallow_deep"]:
                new_plane = "deep"
            elif D >= tau["surface_shallow"]:
                new_plane = "shallow"
            else:
                new_plane = "surface"

            self._plane_assignment[nid] = new_plane
            plane_distribution[new_plane] += 1

            old_idx = PLANE_INDEX.get(old_plane, 0)
            new_idx = PLANE_INDEX.get(new_plane, 0)

            if old_plane != new_plane:
                transitions += 1

                if new_idx > old_idx:
                    descents += 1
                    node.base_activation *= max(0.3, 1.0 - 0.1 * (new_idx - old_idx))
                else:
                    ascents += 1
                    node.base_activation = min(node.weight * 0.3, node.base_activation + 0.1)

            if old_plane == "abyss" and new_plane != "abyss":
                energy_gap = D - self._thresholds["deep_abyss"]
                if energy_gap < 0:
                    p_tunnel = self.tunneling_probability(D, self._thresholds["deep_abyss"])
                    if p_tunnel < 0.1:
                        self._plane_assignment[nid] = "abyss"
                        plane_distribution["abyss"] += 1
                        plane_distribution[new_plane] -= 1
                        transitions -= 1
                        ascents -= 1
                        continue

                node.activation = min(node.activation + 0.5, node.weight)
                node.base_activation = max(0.01, node.weight * 0.1)
                reawakenings += 1

        if depth_values:
            avg_depth = sum(depth_values) / len(depth_values)
            avg_free = sum(free_energy_values) / len(free_energy_values)
            self._energy_history.append({
                "time": now,
                "avg_well_depth": round(avg_depth, 4),
                "avg_free_energy": round(avg_free, 4),
                "temperature": round(self._temperature, 4),
                "transitions": transitions,
                "reawakenings": reawakenings,
            })
            if len(self._energy_history) > self._max_history:
                self._energy_history = self._energy_history[-self._max_history:]

        self._total_transitions += transitions
        self._total_reawakenings += reawakenings
        self._descent_count += descents
        self._ascent_count += ascents
        self._flow_count += 1

        return {
            "transitions": transitions,
            "reawakenings": reawakenings,
            "descents": descents,
            "ascents": ascents,
            "plane_distribution": plane_distribution,
            "temperature": round(self._temperature, 4),
            "avg_well_depth": round(sum(depth_values) / max(len(depth_values), 1), 4),
            "avg_free_energy": round(sum(free_energy_values) / max(len(free_energy_values), 1), 4),
        }

    def get_node_plane(self, node_id: str) -> str:
        return self._plane_assignment.get(node_id, "surface")

    def get_node_depth(self, node_id: str) -> float:
        return self.compute_well_depth(node_id)

    def get_node_energy_report(self, node_id: str) -> Dict:
        node = self._field._nodes.get(node_id)
        if node is None or not node.is_occupied:
            return {"error": "node not found"}

        U = self.compute_internal_energy(node.weight, node.activation)
        S = self.compute_neighborhood_entropy(node_id)
        F = self.compute_free_energy(node_id)
        D = -F

        return {
            "node_id": node_id[:8],
            "internal_energy": round(U, 4),
            "neighborhood_entropy": round(S, 4),
            "free_energy": round(F, 4),
            "well_depth": round(D, 4),
            "plane": self._plane_assignment.get(node_id, "unassigned"),
            "temperature": round(self._temperature, 4),
            "weight": round(node.weight, 4),
            "activation": round(node.activation, 4),
        }

    def get_system_energy_stats(self) -> Dict:
        field = self._field
        if not field._occupied_ids:
            return {"error": "no occupied nodes"}

        depths = []
        free_energies = []
        entropies = []

        sample_size = min(200, len(field._occupied_ids))
        sampled = list(field._occupied_ids)[:sample_size]

        for nid in sampled:
            node = field._nodes.get(nid)
            if not node or not node.is_occupied:
                continue
            U = self.compute_internal_energy(node.weight, node.activation)
            S = self.compute_neighborhood_entropy(nid)
            F = U - self._temperature * S
            depths.append(-F)
            free_energies.append(F)
            entropies.append(S)

        if not depths:
            return {"error": "no data"}

        plane_dist = {"surface": 0, "shallow": 0, "deep": 0, "abyss": 0}
        for nid in field._occupied_ids:
            p = self._plane_assignment.get(nid)
            if p in plane_dist:
                plane_dist[p] += 1

        return {
            "temperature": round(self._temperature, 4),
            "avg_well_depth": round(float(np.mean(depths)), 4),
            "std_well_depth": round(float(np.std(depths)), 4),
            "max_well_depth": round(float(np.max(depths)), 4),
            "min_well_depth": round(float(np.min(depths)), 4),
            "avg_free_energy": round(float(np.mean(free_energies)), 4),
            "avg_entropy": round(float(np.mean(entropies)), 4),
            "plane_distribution": plane_dist,
            "flow_cycles": self._flow_count,
            "total_transitions": self._total_transitions,
            "total_reawakenings": self._total_reawakenings,
            "descent_count": self._descent_count,
            "ascent_count": self._ascent_count,
            "thresholds": self._thresholds,
            "tunneling_kappa": self._tunneling_kappa,
        }

    def get_energy_history(self, last_n: int = 20) -> List[Dict]:
        return self._energy_history[-last_n:]

    def stats(self) -> Dict:
        return {
            "flow_count": self._flow_count,
            "total_transitions": self._total_transitions,
            "total_reawakenings": self._total_reawakenings,
            "descent_count": self._descent_count,
            "ascent_count": self._ascent_count,
            "temperature": round(self._temperature, 4),
            "base_temperature": self._base_temperature,
            "tunneling_kappa": self._tunneling_kappa,
            "thresholds": self._thresholds,
            "assigned_nodes": len(self._plane_assignment),
        }
