"""
Dark Plane Engine -- Thermodynamic Energy Landscape for Memory Depth

Mathematical Foundation
======================
Each memory node occupies a position in an energy landscape modeled as
a system of potential wells. The "depth" of a node's well determines
which dark plane it belongs to.

Thermodynamic Model:
    Internal Energy:
        U(n) = w(n) * a(n) / (1 + w(n) * a(n) / C)
        where C = capacity constant (default 10), providing saturation

    Neighborhood Entropy:
        S(n) = -sum_j  p_j * ln(p_j)
        where p_j is the fraction of neighbor label j among all neighbor labels

    Free Energy:
        F(n) = U(n) - T * S(n)

    Well Depth:
        D(n) = -F(n) = T*S(n) - U(n)
        Higher D = deeper well = harder to excite

Adaptive Plane Assignment (Quantile-based):
    Instead of fixed thresholds, planes are assigned by percentile of the
    well depth distribution among occupied nodes:
        Surface:  bottom 25%  (shallowest wells)
        Shallow:  25th-50th percentile
        Deep:     50th-75th percentile
        Abyss:    top 25%     (deepest wells)

    This ensures roughly equal plane populations and automatic adaptation
    as the field evolves.

Boltzmann Redistribution:
    Each flow cycle, every node has a probability of stochastic transition:
        P(plane_i | D_n) = exp(-beta * |D_n - center_i|) / Z
    where beta = 1/T, center_i = median depth of plane i, Z = partition function.
    This adds thermal noise that prevents dead-lock.

Tunneling (WKB-inspired):
    Probability of escaping from plane_old to plane_new:
        P_tunnel = exp(-2 * kappa * barrier_height)
    where barrier_height = max(0, D_median_new - D_current)
    kappa = tunneling coefficient (default 0.5)

Temperature Model:
    T = T_base * (1 + 0.5 * stress) * circadian_factor * (1 + 0.3 * activity_rate)

    - T_base: baseline temperature (configurable, default 1.0)
    - stress: from self-regulation engine [0, 1]
    - circadian_factor: 1.0 (work) / 0.7 (consolidation)
    - activity_rate: recent query/store rate, normalized [0, 1]

Energy Injection from Queries:
    When a query hits a node, it receives energy E_inject = base + relevance_score * 0.3
    This raises the node's activation, lowering its well depth D, making ascent likely.

Cross-Plane Coupling:
    Nodes in deeper planes exert a "gravitational pull" on nearby surface nodes,
    modeled as a depth-weight that biases pulse propagation toward deep nodes.
    This creates information flow from surface (active) to deep (stored) memory.

Metastable Tracking:
    Nodes in transition between planes are tracked as "metastable" for one cycle,
    receiving a bonus to activation stability to prevent oscillation.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField

PLANE_ORDER = ("surface", "shallow", "deep", "abyss")
PLANE_INDEX = {p: i for i, p in enumerate(PLANE_ORDER)}
PLANE_QUANTILES = [0.0, 0.25, 0.50, 0.75, 1.0]


class DarkPlaneEngine:
    __slots__ = (
        "_field",
        "_temperature",
        "_base_temperature",
        "_tunneling_kappa",
        "_capacity_constant",
        "_adaptive_thresholds",
        "_plane_assignment",
        "_metastable_nodes",
        "_flow_count",
        "_total_transitions",
        "_total_reawakenings",
        "_descent_count",
        "_ascent_count",
        "_energy_history",
        "_max_history",
        "_last_activity_count",
        "_last_activity_time",
        "_activity_rate",
        "_tunnel_events",
        "_boltzmann_events",
        "_energy_injections",
        "_substrate",
        "_h5_regulation",
        "_h6_cascade_strength",
    )

    def __init__(self, field: HoneycombNeuralField, config: Optional[Dict] = None):
        cfg = config or {}
        self._field = field
        self._base_temperature = cfg.get("base_temperature", 1.0)
        self._temperature = self._base_temperature
        self._tunneling_kappa = cfg.get("tunneling_kappa", 0.5)
        self._capacity_constant = cfg.get("capacity_constant", 10.0)
        self._adaptive_thresholds: Dict[str, float] = {
            "surface_shallow": 0.0,
            "shallow_deep": 0.5,
            "deep_abyss": 1.0,
        }
        self._plane_assignment: Dict[str, str] = {}
        self._metastable_nodes: Dict[str, float] = {}
        self._flow_count: int = 0
        self._total_transitions: int = 0
        self._total_reawakenings: int = 0
        self._descent_count: int = 0
        self._ascent_count: int = 0
        self._energy_history: List[Dict] = []
        self._max_history: int = 200
        self._last_activity_count: int = 0
        self._last_activity_time: float = time.time()
        self._activity_rate: float = 0.0
        self._tunnel_events: int = 0
        self._boltzmann_events: int = 0
        self._energy_injections: int = 0
        self._substrate = None
        self._h5_regulation: float = 0.0
        self._h6_cascade_strength: float = 0.0

    def set_substrate(self, substrate):
        self._substrate = substrate

    def compute_internal_energy(self, weight: float, activation: float) -> float:
        C = self._capacity_constant
        raw = weight * activation
        return raw / (1.0 + raw / C)

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
        return U - self._temperature * S

    def compute_well_depth(self, node_id: str) -> float:
        return -self.compute_free_energy(node_id)

    def _compute_all_depths(self, occupied: List[str]) -> Dict[str, float]:
        depths = {}
        for nid in occupied:
            node = self._field._nodes.get(nid)
            if node and node.is_occupied:
                depths[nid] = self.compute_well_depth(nid)
        return depths

    def _update_adaptive_thresholds(self, depths: Dict[str, float]):
        if len(depths) < 8:
            return

        values = sorted(depths.values())
        n = len(values)
        q25 = values[max(0, int(n * 0.25) - 1)]
        q50 = values[max(0, int(n * 0.50) - 1)]
        q75 = values[max(0, int(n * 0.75) - 1)]

        smoothing = 0.3
        self._adaptive_thresholds["surface_shallow"] = (
            (1 - smoothing) * self._adaptive_thresholds["surface_shallow"]
            + smoothing * q25
        )
        self._adaptive_thresholds["shallow_deep"] = (
            (1 - smoothing) * self._adaptive_thresholds["shallow_deep"]
            + smoothing * q50
        )
        self._adaptive_thresholds["deep_abyss"] = (
            (1 - smoothing) * self._adaptive_thresholds["deep_abyss"]
            + smoothing * q75
        )

    def _assign_plane_by_depth(self, D: float) -> str:
        tau = self._adaptive_thresholds
        if D >= tau["deep_abyss"]:
            return "abyss"
        elif D >= tau["shallow_deep"]:
            return "deep"
        elif D >= tau["surface_shallow"]:
            return "shallow"
        else:
            return "surface"

    def tunneling_probability(self, from_depth: float, to_depth: float) -> float:
        barrier = to_depth - from_depth
        if barrier <= 0:
            return 1.0
        return math.exp(-2.0 * self._tunneling_kappa * barrier)

    def _boltzmann_redistribute(
        self, nid: str, D: float, assigned_plane: str
    ) -> Optional[str]:
        if self._temperature < 1e-10:
            return None

        tau = self._adaptive_thresholds
        plane_centers = {
            "surface": tau["surface_shallow"] * 0.5,
            "shallow": (tau["surface_shallow"] + tau["shallow_deep"]) * 0.5,
            "deep": (tau["shallow_deep"] + tau["deep_abyss"]) * 0.5,
            "abyss": tau["deep_abyss"] + 0.5,
        }

        beta = 1.0 / self._temperature
        log_weights = {}
        for plane, center in plane_centers.items():
            dist = abs(D - center)
            log_weights[plane] = -beta * dist

        max_lw = max(log_weights.values())
        Z = 0.0
        probs = {}
        for plane, lw in log_weights.items():
            w = math.exp(lw - max_lw)
            probs[plane] = w
            Z += w

        if Z < 1e-15:
            return None

        for plane in probs:
            probs[plane] /= Z

        current_idx = PLANE_INDEX.get(assigned_plane, 0)

        transition_prob = 0.05
        if random.random() > transition_prob:
            return None

        r = random.random()
        cumulative = 0.0
        chosen = assigned_plane
        for plane in PLANE_ORDER:
            cumulative += probs.get(plane, 0.0)
            if r < cumulative:
                chosen = plane
                break

        new_idx = PLANE_INDEX.get(chosen, 0)
        if abs(new_idx - current_idx) > 1:
            if new_idx > current_idx:
                chosen = PLANE_ORDER[current_idx + 1]
            else:
                chosen = PLANE_ORDER[current_idx - 1]

        if chosen != assigned_plane:
            self._boltzmann_events += 1
            return chosen
        return None

    def _update_activity_rate(self):
        field = self._field
        now = time.time()
        current_stores = len(field._recent_stores)
        pulse_count = field._pulse_count
        total_activity = current_stores + (pulse_count - self._last_activity_count) * 0.01
        dt = max(now - self._last_activity_time, 0.1)
        raw_rate = total_activity / dt
        self._activity_rate = min(1.0, raw_rate / 5.0)
        self._last_activity_count = pulse_count
        self._last_activity_time = now

    def update_temperature(
        self,
        stress_level: float = 0.0,
        circadian_phase: str = "work",
    ):
        circadian_factor = 1.0 if circadian_phase == "work" else 0.7
        self._update_activity_rate()
        T = (
            self._base_temperature
            * (1.0 + 0.5 * stress_level)
            * circadian_factor
            * (1.0 + 0.3 * self._activity_rate)
        )
        self._temperature = max(0.1, min(5.0, T))

    def inject_query_energy(self, node_id: str, relevance: float):
        node = self._field._nodes.get(node_id)
        if node is None or not node.is_occupied:
            return

        current_plane = self._plane_assignment.get(node_id, "surface")
        if current_plane in ("deep", "abyss"):
            energy = 0.1 + relevance * 0.3
            node.activation = min(node.activation + energy, node.weight * 2.0)
            node.base_activation = max(node.base_activation, node.weight * 0.05)
            self._energy_injections += 1

            if current_plane == "abyss":
                node.hibernated = False

    def get_depth_weight_for_neighbor(self, node_id: str) -> float:
        plane = self._plane_assignment.get(node_id, "surface")
        depth_weights = {
            "surface": 1.0,
            "shallow": 1.5,
            "deep": 2.5,
            "abyss": 4.0,
        }
        return depth_weights.get(plane, 1.0)

    def run_flow_cycle(self) -> Dict:
        field = self._field
        now = time.time()

        if self._substrate is not None:
            proj = self._substrate.get_projection_data()
            self._base_temperature = 1.0 + 0.2 * proj.get("void_energy", 0.0)
            self._h5_regulation = proj.get("h5_regulation", 0.0)
            self._h6_cascade_strength = proj.get("h6_cascade_strength", 0.0)

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
                "transitions": 0,
                "reawakenings": 0,
                "plane_distribution": plane_distribution,
                "temperature": self._temperature,
            }

        all_depths = self._compute_all_depths(occupied)
        self._update_adaptive_thresholds(all_depths)

        for nid in occupied:
            node = field._nodes.get(nid)
            if not node or not node.is_occupied:
                continue

            D = all_depths.get(nid, 0.0)
            F = -D
            free_energy_values.append(F)
            depth_values.append(D)

            new_plane = self._assign_plane_by_depth(D)

            boltzmann_result = self._boltzmann_redistribute(nid, D, new_plane)
            if boltzmann_result is not None:
                new_plane = boltzmann_result

            old_plane = self._plane_assignment.get(nid, "surface")
            if self._h5_regulation > 0:
                if PLANE_INDEX.get(new_plane, 0) < PLANE_INDEX.get(old_plane, 0):
                    if random.random() < 0.3 * self._h5_regulation:
                        new_plane = old_plane
            elif self._h5_regulation < 0:
                if PLANE_INDEX.get(new_plane, 0) > PLANE_INDEX.get(old_plane, 0):
                    if random.random() < 0.3 * abs(self._h5_regulation):
                        new_plane = old_plane

            old_idx = PLANE_INDEX.get(old_plane, 0)
            new_idx = PLANE_INDEX.get(new_plane, 0)

            if old_plane != new_plane:
                if nid in self._metastable_nodes:
                    ms_time = self._metastable_nodes[nid]
                    if now - ms_time < 30.0:
                        new_plane = old_plane
                        new_idx = old_idx
                    else:
                        del self._metastable_nodes[nid]

            if old_plane != new_plane:
                transitions += 1

                if new_idx > old_idx:
                    descents += 1
                    decay_factor = max(0.3, 1.0 - 0.15 * (new_idx - old_idx))
                    node.base_activation *= decay_factor
                    if new_plane == "abyss":
                        node.hibernated = True
                else:
                    ascents += 1
                    barrier = all_depths.get(nid, 0) - self._adaptive_thresholds.get(
                        "deep_abyss" if old_plane == "abyss" else "shallow_deep", 0
                    )
                    if barrier > 0:
                        p_escape = self.tunneling_probability(
                            all_depths.get(nid, 0),
                            all_depths.get(nid, 0) - barrier,
                        )
                        if p_escape < random.random():
                            new_plane = old_plane
                            new_idx = old_idx
                            transitions -= 1
                            ascents -= 1
                            self._tunnel_events += 1
                            continue

                    recovery = min(node.weight * 0.15, 0.2)
                    node.base_activation = max(
                        node.base_activation,
                        node.base_activation + recovery,
                    )
                    if old_plane == "abyss":
                        node.hibernated = False
                        node.activation = min(node.activation + 0.3, node.weight)
                        node.base_activation = max(0.02, node.weight * 0.08)
                        reawakenings += 1

                self._metastable_nodes[nid] = now

            self._plane_assignment[nid] = new_plane
            plane_distribution[new_plane] += 1

        if depth_values:
            avg_depth = sum(depth_values) / len(depth_values)
            avg_free = sum(free_energy_values) / len(free_energy_values)
            self._energy_history.append(
                {
                    "time": now,
                    "avg_well_depth": round(avg_depth, 4),
                    "avg_free_energy": round(avg_free, 4),
                    "temperature": round(self._temperature, 4),
                    "transitions": transitions,
                    "reawakenings": reawakenings,
                    "plane_distribution": dict(plane_distribution),
                    "thresholds": dict(self._adaptive_thresholds),
                }
            )
            if len(self._energy_history) > self._max_history:
                self._energy_history = self._energy_history[-self._max_history:]

        stale_metastable = [
            nid for nid, t in self._metastable_nodes.items() if now - t > 60.0
        ]
        for nid in stale_metastable:
            del self._metastable_nodes[nid]

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
            "avg_well_depth": round(
                sum(depth_values) / max(len(depth_values), 1), 4
            ),
            "avg_free_energy": round(
                sum(free_energy_values) / max(len(free_energy_values), 1), 4
            ),
            "adaptive_thresholds": {
                k: round(v, 4) for k, v in self._adaptive_thresholds.items()
            },
            "activity_rate": round(self._activity_rate, 4),
            "metastable_count": len(self._metastable_nodes),
        }

    def assign_plane(self, node_id: str) -> str:
        if node_id in self._plane_assignment:
            return self._plane_assignment[node_id]
        D = self.compute_well_depth(node_id)
        return self._assign_plane_by_depth(D)

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
        plane = self._plane_assignment.get(node_id, "unassigned")
        is_metastable = node_id in self._metastable_nodes

        return {
            "node_id": node_id[:8],
            "internal_energy": round(U, 4),
            "neighborhood_entropy": round(S, 4),
            "free_energy": round(F, 4),
            "well_depth": round(D, 4),
            "plane": plane,
            "temperature": round(self._temperature, 4),
            "weight": round(node.weight, 4),
            "activation": round(node.activation, 4),
            "base_activation": round(node.base_activation, 4),
            "is_metastable": is_metastable,
            "depth_weight": self.get_depth_weight_for_neighbor(node_id),
        }

    def get_system_energy_stats(self) -> Dict:
        field = self._field
        if not field._occupied_ids:
            return {"error": "no occupied nodes"}

        depths = []
        free_energies = []
        entropies = []

        sample_size = min(500, len(field._occupied_ids))
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

        total = sum(plane_dist.values())
        plane_ratios = {k: round(v / max(total, 1), 4) for k, v in plane_dist.items()}

        depth_arr = np.array(depths)
        return {
            "temperature": round(self._temperature, 4),
            "avg_well_depth": round(float(np.mean(depth_arr)), 4),
            "std_well_depth": round(float(np.std(depth_arr)), 4),
            "max_well_depth": round(float(np.max(depth_arr)), 4),
            "min_well_depth": round(float(np.min(depth_arr)), 4),
            "median_well_depth": round(float(np.median(depth_arr)), 4),
            "avg_free_energy": round(float(np.mean(free_energies)), 4),
            "avg_entropy": round(float(np.mean(entropies)), 4),
            "plane_distribution": plane_dist,
            "plane_ratios": plane_ratios,
            "adaptive_thresholds": {
                k: round(v, 4) for k, v in self._adaptive_thresholds.items()
            },
            "flow_cycles": self._flow_count,
            "total_transitions": self._total_transitions,
            "total_reawakenings": self._total_reawakenings,
            "descent_count": self._descent_count,
            "ascent_count": self._ascent_count,
            "tunneling_kappa": self._tunneling_kappa,
            "activity_rate": round(self._activity_rate, 4),
            "metastable_count": len(self._metastable_nodes),
            "tunnel_events": self._tunnel_events,
            "boltzmann_events": self._boltzmann_events,
            "energy_injections": self._energy_injections,
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
            "adaptive_thresholds": {
                k: round(v, 4) for k, v in self._adaptive_thresholds.items()
            },
            "assigned_nodes": len(self._plane_assignment),
            "metastable_nodes": len(self._metastable_nodes),
            "activity_rate": round(self._activity_rate, 4),
            "tunnel_events": self._tunnel_events,
            "boltzmann_events": self._boltzmann_events,
            "energy_injections": self._energy_injections,
        }
