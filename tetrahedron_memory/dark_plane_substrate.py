"""
Dark Plane Substrate -- High-Dimensional Topological Void Substrate (H0~H6)

Mathematical Foundation
======================
The dark plane substrate models the hidden topological structure underlying
the visible BCC lattice memory field. It implements a seven-level homology
system:

  H0 (Connected Components)  -- actual PH via Union-Find
  H1 (Cycles / Channels)     -- actual PH via cycle detection
  H2 (Voids / Cavities)      -- actual PH via shell detection
  H3 (Internal Volume)       -- ODE dynamics, driven by H1+H2
  H4 (Many-body Entanglement) -- ODE dynamics, driven by H3 + dream + void channels
  H5 (Control Hub)           -- ODE dynamics, driven by H4 coherence feedback
  H6 (Cosmic Meta-structure) -- ODE dynamics, cascade phase transitions

Void Energy:
  E_void = H(T) + Q(T) + I(T)
  H = persistent entropy, Q = topological charge, I = information density

Dark Energy (H2 contribution):
  E_dark = sum_{i in H2} persistence_i * (1 + gamma * Q_i)

Channel Energy (H1 contribution):
  E_channel = sum_{i in H1} persistence_i * (1 + flow_i)

Coherence:
  Phi = w1*H_norm + w2*E_ratio + w3*sync + w4*Psi

Phase Transitions:
  H4: d|H4|/dt > 0.15 and E_multi > 3.5 and C > 0.75
  H5: |H5| > 0.1*N and |Regulate5| > 0.7 for 5+ cycles
  H6: |H6| > 0.05*N and E6 > 2.0*E_void and Psi > 0.6
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField


@dataclass
class HomologyFeature:
    birth: float
    death: float
    dimension: int
    persistence: float
    topo_charge: float
    participating_nodes: list


@dataclass
class HighDimState:
    count: float = 0.0
    energy: float = 0.0
    growth_rate: float = 0.0


@dataclass
class PhaseTransitionEvent:
    timestamp: float
    level: int
    energy: float
    coherence: float
    trigger_condition: str
    pre_state: dict
    post_state: dict


@dataclass
class SubstrateState:
    features_h0: list = field(default_factory=list)
    features_h1: list = field(default_factory=list)
    features_h2: list = field(default_factory=list)

    void_energy: float = 0.0
    dark_energy: float = 0.0
    channel_energy: float = 0.0
    persistent_entropy: float = 0.0

    h3: HighDimState = field(default_factory=HighDimState)
    h4: HighDimState = field(default_factory=HighDimState)
    h5: HighDimState = field(default_factory=HighDimState)
    h6: HighDimState = field(default_factory=HighDimState)

    coherence: float = 0.0
    total_dim_energy: float = 0.0
    cascade_potential: float = 0.0

    phase_transitions: list = field(default_factory=list)
    last_phase_transition: float = 0.0
    total_phase_transitions: int = 0


class DarkPlaneSubstrate:
    __slots__ = (
        "_field",
        "_state",
        "_cycle_count",
        "_ph_interval",
        "_ph_sample_size",
        "_dt",
        "_h5_regulation",
        "_psi_field",
        "_h5_high_count",
        "_h5_high_duration",
        "_last_h5_cascade_time",
        "_max_features_persist",
    )

    def __init__(self, field: HoneycombNeuralField):
        self._field = field
        self._state = SubstrateState()
        self._cycle_count = 0
        self._ph_interval = 10
        self._ph_sample_size = 500
        self._dt = 0.1
        self._h5_regulation: float = 0.0
        self._psi_field: float = 0.0
        self._h5_high_count: int = 0
        self._h5_high_duration: int = 0
        self._last_h5_cascade_time: float = 0.0
        self._max_features_persist: int = 50

    def update(
        self,
        stress: float,
        temperature: float,
        activity_rate: float,
        dream_active: bool = False,
    ) -> dict:
        self._cycle_count += 1
        n_total = max(1, self._field._occupied_count)

        if self._cycle_count % self._ph_interval == 0:
            ph = self._compute_persistence_numpy()
            self._state.features_h0 = ph.get("h0", [])
            self._state.features_h1 = ph.get("h1", [])
            self._state.features_h2 = ph.get("h2", [])

        self._state.void_energy = self._compute_void_energy()
        self._state.dark_energy = self._compute_dark_energy()
        self._state.channel_energy = self._compute_channel_energy()

        all_ph = self._state.features_h0 + self._state.features_h1 + self._state.features_h2
        self._state.persistent_entropy = self._compute_persistent_entropy(all_ph)

        coupling = self._compute_cross_dim_coupling()

        h1_count = len(self._state.features_h1)
        h2_count = len(self._state.features_h2)

        self._state.h3 = self._update_h3_dynamics(
            h1_count, h2_count, stress, activity_rate, n_total, self._dt, coupling
        )
        self._state.h4 = self._update_h4_dynamics(
            stress, dream_active, n_total, self._dt, coupling
        )
        self._state.h5 = self._update_h5_dynamics(
            self._state.h4, self._state.coherence, stress, n_total, self._dt, coupling
        )
        self._state.h6 = self._update_h6_dynamics(
            self._state.h5,
            self._state.total_dim_energy,
            self._state.last_phase_transition,
            n_total,
            self._dt,
            coupling,
        )

        self._state.coherence = self._compute_coherence()
        self._state.total_dim_energy = (
            self._state.h3.energy
            + self._state.h4.energy
            + self._state.h5.energy
        )
        self._state.cascade_potential = self._psi_field

        transitions = self._detect_phase_transition()

        return {
            "void_energy": self._state.void_energy,
            "dark_energy": self._state.dark_energy,
            "channel_energy": self._state.channel_energy,
            "coherence": self._state.coherence,
            "h3": {"count": self._state.h3.count, "energy": self._state.h3.energy, "growth_rate": self._state.h3.growth_rate},
            "h4": {"count": self._state.h4.count, "energy": self._state.h4.energy, "growth_rate": self._state.h4.growth_rate},
            "h5": {"count": self._state.h5.count, "energy": self._state.h5.energy, "growth_rate": self._state.h5.growth_rate},
            "h6": {"count": self._state.h6.count, "energy": self._state.h6.energy, "growth_rate": self._state.h6.growth_rate},
            "psi_field": self._psi_field,
            "total_dim_energy": self._state.total_dim_energy,
            "cascade_potential": self._state.cascade_potential,
            "persistent_entropy": self._state.persistent_entropy,
            "cross_dim_coupling": coupling,
            "phase_transitions": transitions,
        }

    def _compute_persistence_numpy(self) -> dict:
        field = self._field
        nodes = field._nodes
        occupied = field._occupied_ids

        if len(occupied) < 2:
            return {"h0": [], "h1": [], "h2": []}

        occ_list = list(occupied)
        sample_size = self._ph_sample_size
        if sample_size > 0 and len(occ_list) > sample_size:
            import random
            occ_list = random.sample(occ_list, sample_size)

        idx_map = {nid: i for i, nid in enumerate(occ_list)}
        n = len(occ_list)

        dist_matrix = np.full((n, n), float("inf"), dtype=np.float32)
        for nid in occ_list:
            node = nodes.get(nid)
            if node is None or not node.is_occupied:
                continue
            i = idx_map[nid]
            for nnid in node.face_neighbors[:8]:
                if nnid in idx_map:
                    j = idx_map[nnid]
                    nn = nodes.get(nnid)
                    w = 1.0
                    if nn and nn.is_occupied:
                        w = max(nn.weight, 0.01)
                    d = 1.0 / (1.0 + w)
                    dist_matrix[i, j] = min(dist_matrix[i, j], d)
                    dist_matrix[j, i] = min(dist_matrix[j, i], d)
            for nnid in node.edge_neighbors[:4]:
                if nnid in idx_map:
                    j = idx_map[nnid]
                    nn = nodes.get(nnid)
                    w = 1.0
                    if nn and nn.is_occupied:
                        w = max(nn.weight, 0.01)
                    d = 1.0 / (1.0 + w)
                    dist_matrix[i, j] = min(dist_matrix[i, j], d)
                    dist_matrix[j, i] = min(dist_matrix[j, i], d)

        h0_features = self._compute_h0(occ_list, dist_matrix)
        h1_features = self._compute_h1(occ_list, idx_map, dist_matrix, nodes)
        h2_features = self._compute_h2(occ_list, idx_map, dist_matrix, nodes)

        return {"h0": h0_features, "h1": h1_features, "h2": h2_features}

    def _compute_h0(self, occ_list, dist_matrix):
        n = len(occ_list)
        if n == 0:
            return []

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i, j]
                if d < float("inf"):
                    edges.append((d, i, j))
        edges.sort()

        parent = list(range(n))
        rank = [0] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        features = []
        num_components = n
        for filt_val, i, j in edges:
            if union(i, j):
                num_components -= 1
                birth_of_merged = 0.0
                features.append(
                    HomologyFeature(
                        birth=birth_of_merged,
                        death=float(filt_val),
                        dimension=0,
                        persistence=float(filt_val) - birth_of_merged,
                        topo_charge=0.0,
                        participating_nodes=[occ_list[i], occ_list[j]],
                    )
                )

        features = sorted(features, key=lambda f: f.persistence, reverse=True)
        return features[: self._max_features_persist]

    def _compute_h1(self, occ_list, idx_map, dist_matrix, nodes):
        n = len(occ_list)
        if n < 3:
            return []

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i, j]
                if d < float("inf"):
                    edges.append((d, i, j))
        edges.sort()

        parent = list(range(n))
        rank = [0] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return True
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return False

        active_cycles = []
        adj = {i: set() for i in range(n)}

        for filt_val, i, j in edges:
            if union(i, j):
                adj[i].add(j)
                adj[j].add(i)
            else:
                path = self._bfs_path(adj, i, j)
                if path:
                    cycle_nodes = [occ_list[k] for k in path]
                    cycle_edges = len(path)
                    genus = cycle_edges / max(3, cycle_edges)
                    active_cycles.append(
                        {
                            "birth": float(filt_val),
                            "nodes": cycle_nodes,
                            "genus": genus,
                            "path": path,
                        }
                    )
                adj[i].add(j)
                adj[j].add(i)

        triangles = set()
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                common = adj[i] & adj[j]
                for k in common:
                    if k > j:
                        tri = (i, j, k)
                        triangles.add(tri)

        features = []
        for cycle in active_cycles:
            death_val = float("inf")
            path_set = set()
            path = cycle["path"]
            for idx in range(len(path)):
                a, b = path[idx], path[(idx + 1) % len(path)]
                path_set.add((min(a, b), max(a, b)))

            for i, j, k in triangles:
                tri_edges = {
                    (min(i, j), max(i, j)),
                    (min(i, k), max(i, k)),
                    (min(j, k), max(j, k)),
                }
                if tri_edges <= path_set:
                    tri_filt = max(
                        dist_matrix[i, j],
                        dist_matrix[i, k],
                        dist_matrix[j, k],
                    )
                    if tri_filt < death_val and tri_filt > cycle["birth"]:
                        death_val = tri_filt

            if death_val == float("inf"):
                death_val = cycle["birth"] + 1.0

            persistence = death_val - cycle["birth"]
            if persistence > 1e-8:
                features.append(
                    HomologyFeature(
                        birth=cycle["birth"],
                        death=death_val,
                        dimension=1,
                        persistence=persistence,
                        topo_charge=cycle["genus"],
                        participating_nodes=cycle["nodes"][:20],
                    )
                )

        features = sorted(features, key=lambda f: f.persistence, reverse=True)
        return features[: self._max_features_persist]

    @staticmethod
    def _bfs_path(adj, start, end):
        if start == end:
            return [start]
        from collections import deque

        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            node, path = queue.popleft()
            for neighbor in adj[node]:
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _compute_h2(self, occ_list, idx_map, dist_matrix, nodes):
        n = len(occ_list)
        if n < 4:
            return []

        adj = {i: set() for i in range(n)}
        threshold = float(np.percentile(dist_matrix[dist_matrix < float("inf")], 50)) if np.any(dist_matrix < float("inf")) else 1.0

        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] <= threshold:
                    adj[i].add(j)
                    adj[j].add(i)

        triangles = []
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                common = adj[i] & adj[j]
                for k in common:
                    if k > j:
                        triangles.append((i, j, k))

        tetrahedra = []
        for i, j, k in triangles:
            common_all = adj[i] & adj[j] & adj[k]
            for l in common_all:
                if l > k:
                    tetrahedra.append((i, j, k, l))

        if not tetrahedra:
            return []

        features = []
        for tet in tetrahedra[:200]:
            i, j, k, l = tet
            face_edges = [
                (min(i, j), max(i, j)),
                (min(i, k), max(i, k)),
                (min(i, l), max(i, l)),
                (min(j, k), max(j, k)),
                (min(j, l), max(j, l)),
                (min(k, l), max(k, l)),
            ]
            birth_val = max(dist_matrix[a, b] for a, b in face_edges)
            death_val = max(birth_val + 0.5, birth_val * 1.2)
            persistence = death_val - birth_val

            if persistence > 1e-8:
                features.append(
                    HomologyFeature(
                        birth=birth_val,
                        death=death_val,
                        dimension=2,
                        persistence=persistence,
                        topo_charge=1.0,
                        participating_nodes=[occ_list[x] for x in tet],
                    )
                )

        features = sorted(features, key=lambda f: f.persistence, reverse=True)
        return features[: self._max_features_persist]

    def _compute_void_energy(self) -> float:
        all_features = (
            self._state.features_h0
            + self._state.features_h1
            + self._state.features_h2
        )

        h_ent = 0.0
        if all_features:
            persistences = [f.persistence for f in all_features if f.persistence > 0]
            if persistences:
                total_p = sum(persistences)
                if total_p > 0:
                    h_ent = -sum(
                        (p / total_p) * math.log(p / total_p)
                        for p in persistences
                        if p / total_p > 0
                    )

        q_charge = 0.0
        if all_features:
            max_p = max((f.persistence for f in all_features), default=1.0)
            if max_p > 0:
                q_charge = sum(
                    (1.0 + f.topo_charge) * f.persistence / max_p
                    for f in all_features
                )

        i_density = 0.0
        n_total = max(1, self._field._occupied_count)
        occupied = self._field._occupied_ids
        if occupied:
            total_w = 0.0
            count = 0
            for nid in occupied:
                node = self._field._nodes.get(nid)
                if node and node.is_occupied:
                    total_w += node.weight
                    count += 1
            if count > 0:
                i_density = count * (total_w / count) / n_total

        return h_ent + q_charge + i_density

    def _compute_dark_energy(self) -> float:
        gamma = 0.6
        return sum(
            f.persistence * (1.0 + gamma * f.topo_charge)
            for f in self._state.features_h2
        )

    def _compute_channel_energy(self) -> float:
        return sum(f.persistence * (1.0 + f.topo_charge) for f in self._state.features_h1)

    def _compute_persistent_entropy(self, features: list) -> float:
        if not features:
            return 0.0
        persistences = np.array([f.persistence for f in features], dtype=np.float64)
        total = persistences.sum()
        if total < 1e-12:
            return 0.0
        probs = persistences / total
        mask = probs > 1e-12
        return float(-np.sum(probs[mask] * np.log(probs[mask])))

    def _compute_cross_dim_coupling(self) -> dict:
        e_h0 = len(self._state.features_h0) * 0.05
        e_h1 = sum(f.persistence * (1.0 + 0.4 * f.topo_charge) for f in self._state.features_h1) * 0.20
        e_h2 = sum(f.persistence * (1.0 + 0.6 * f.topo_charge) for f in self._state.features_h2) * 0.35
        e_h3 = self._state.h3.energy * 0.25
        e_h4 = self._state.h4.energy * 0.15
        e_h5 = self._state.h5.energy * 0.12
        return {
            "e_h0": e_h0,
            "e_h1": e_h1,
            "e_h2": e_h2,
            "e_h3": e_h3,
            "e_h4": e_h4,
            "e_h5": e_h5,
            "e_lower_total": e_h0 + e_h1 + e_h2 + e_h3 + e_h4 + e_h5,
            "coupling_h01": e_h0 * 0.10 if e_h0 > 0 else 0.0,
            "coupling_h12": e_h1 * 0.25 if e_h1 > 0 else 0.0,
            "coupling_h23": e_h2 * 0.30 if e_h2 > 0 else 0.0,
            "coupling_h34": e_h3 * 0.35 if e_h3 > 0 else 0.0,
            "coupling_h45": e_h4 * 0.40 if e_h4 > 0 else 0.0,
        }

    def _update_h3_dynamics(
        self, h1_count, h2_count, stress, activity_rate, n_total, dt,
        coupling: dict = None,
    ) -> HighDimState:
        alpha3 = 0.5
        beta3 = 0.2
        gamma3 = 0.3
        delta3 = 0.4
        epsilon3 = 0.15

        state = self._state.h3
        e_void = max(self._state.void_energy, 0.01)
        i_density = 0.0
        occupied = self._field._occupied_ids
        if occupied:
            total_w = 0.0
            count = 0
            for nid in occupied:
                node = self._field._nodes.get(nid)
                if node and node.is_occupied:
                    total_w += node.weight
                    count += 1
            if count > 0:
                i_density = count * (total_w / count) / n_total

        h12_avg = 0.0
        all_h12 = self._state.features_h1 + self._state.features_h2
        if all_h12:
            h12_avg = sum(f.persistence for f in all_h12) / len(all_h12)

        d_count = (
            alpha3 * (h1_count + h2_count) * e_void / n_total
            - beta3 * stress * state.count
            + gamma3 * i_density * math.log(1.0 + state.count)
            + 0.20 * (coupling.get("coupling_h12", 0.0) if coupling else 0.0)
        )
        d_energy = (
            delta3 * state.count * h12_avg
            - epsilon3 * activity_rate * state.energy
            + 0.15 * (coupling.get("e_h2", 0.0) if coupling else 0.0)
        )

        new_count = self._euler_step(state.count, d_count, dt, (0, 0.5 * n_total))
        new_energy = self._euler_step(state.energy, d_energy, dt, (0, e_void))

        return HighDimState(
            count=new_count, energy=new_energy, growth_rate=d_count
        )

    def _update_h4_dynamics(self, stress, dream_active, n_total, dt, coupling: dict = None):
        alpha4 = 0.85
        beta4 = 0.35
        gamma4 = 0.6
        delta4 = 0.7
        epsilon4 = 0.4
        zeta4 = 0.5
        eta4 = 0.9
        theta4 = 0.45

        state = self._state.h4
        e_void = max(self._state.void_energy, 0.01)

        h4 = state.count
        e_multi = state.energy
        c = self._state.coherence

        multi_factor = 1.0
        dream_injection = 0.3 if dream_active else 0.0

        pulse_sync = 0.0
        occupied = self._field._occupied_ids
        if occupied:
            sync_vals = []
            for nid in list(occupied)[:100]:
                node = self._field._nodes.get(nid)
                if node and node.is_occupied:
                    sync_vals.append(min(node.activation / max(node.weight, 0.01), 1.0))
            if sync_vals:
                pulse_sync = float(np.mean(sync_vals))

        fill_rate = 0.0
        n_max = max(1, self._field._resolution ** 3 * 2)
        fill_rate = n_total / n_max

        p_avg = 0.0
        if self._state.features_h1 + self._state.features_h2:
            p_avg = float(np.mean([f.persistence for f in self._state.features_h1 + self._state.features_h2]))

        d_count = alpha4 * e_multi * c - beta4 * stress * h4 + gamma4 * dream_injection + 0.25 * (coupling.get("coupling_h23", 0.0) if coupling else 0.0)
        d_energy = delta4 * h4 * p_avg - epsilon4 * fill_rate * e_multi + zeta4 * pulse_sync + 0.20 * (coupling.get("e_h3", 0.0) if coupling else 0.0)

        new_count = self._euler_step(h4, d_count, dt, (0, 0.3 * n_total))
        new_energy = self._euler_step(e_multi, d_energy, dt, (0, 2.0 * e_void))

        return HighDimState(
            count=new_count, energy=new_energy, growth_rate=d_count
        )

    def _update_h5_dynamics(self, h4, coherence, stress, n_total, dt, coupling: dict = None):
        alpha5 = 0.6
        beta5 = 0.25
        gamma5 = 0.4
        delta5 = 0.5
        epsilon5 = 0.2
        eta5 = 0.7
        theta5 = 0.3

        state = self._state.h5
        e_void = max(self._state.void_energy, 0.01)
        h4_max = max(0.1 * n_total, 1.0)
        h4_target = 0.1 * n_total

        phi = max(coherence, 0.01)
        h4_ratio = h4.count / h4_max
        dh4_dt = h4.growth_rate

        d_count = (
            alpha5 * phi * h4_ratio
            - beta5 * state.count * (1.0 - phi)
            + gamma5 * dh4_dt * (1.0 if dh4_dt > 0 else -1.0)
            + 0.30 * (coupling.get("coupling_h34", 0.0) if coupling else 0.0)
        )
        d_energy = delta5 * state.count * phi ** 2 - epsilon5 * stress * state.energy + 0.25 * (coupling.get("e_h4", 0.0) if coupling else 0.0)

        regulation = eta5 * (h4_target - h4.count) * phi - theta5 * self._h5_regulation
        self._h5_regulation = self._euler_step(
            self._h5_regulation, regulation, dt, (-1.0, 1.0)
        )

        new_count = self._euler_step(state.count, d_count, dt, (0, 0.15 * n_total))
        new_energy = self._euler_step(state.energy, d_energy, dt, (0, e_void))

        if abs(self._h5_regulation) > 0.7:
            self._h5_high_duration += 1
        else:
            self._h5_high_duration = 0

        return HighDimState(
            count=new_count, energy=new_energy, growth_rate=d_count
        )

    def _update_h6_dynamics(self, h5, total_energy, last_pt_time, n_total, dt, coupling: dict = None):
        alpha6 = 0.3
        beta6 = 0.1
        lam6 = 0.05
        gamma6 = 0.8
        delta6 = 0.15
        epsilon6 = 0.08
        eta6 = 0.5
        theta6 = 0.2

        state = self._state.h6
        e_void = max(self._state.void_energy, 0.01)
        e_max = 3.0 * e_void
        e_total = max(total_energy, 0.01)

        t_since_cascade = time.time() - last_pt_time if last_pt_time > 0 else 1000.0
        cascade_trigger = 1.0 if self._h5_high_duration >= 5 else 0.0
        phi = max(self._state.coherence, 0.01)

        d_count = (
            alpha6 * h5.count * e_total / e_max
            - beta6 * state.count * math.exp(-lam6 * t_since_cascade)
            + gamma6 * cascade_trigger
            + 0.35 * (coupling.get("coupling_h45", 0.0) if coupling else 0.0)
        )

        t_elapsed = max(time.time() - (self._state.last_phase_transition or time.time()), 1.0)
        lower_coupling = 0.30 * (coupling.get("e_lower_total", 0.0) if coupling else 0.0)
        d_energy = delta6 * state.count ** 2 * phi - epsilon6 * state.energy / (1.0 + t_elapsed) + lower_coupling

        stress = 0.0
        reg = self._field._self_regulation
        if reg:
            stress = reg._stress_level

        d_psi = eta6 * state.count * phi * h5.count - theta6 * self._psi_field * stress
        self._psi_field = self._euler_step(self._psi_field, d_psi, dt, (0.0, 1.0))

        new_count = self._euler_step(state.count, d_count, dt, (0, 0.1 * n_total))
        new_energy = self._euler_step(state.energy, d_energy, dt, (0, 5.0 * e_void))

        return HighDimState(
            count=new_count, energy=new_energy, growth_rate=d_count
        )

    def _compute_coherence(self) -> float:
        all_features = (
            self._state.features_h0
            + self._state.features_h1
            + self._state.features_h2
        )

        h_norm = 0.0
        if all_features:
            persistences = [f.persistence for f in all_features if f.persistence > 0]
            if persistences:
                h_ent = -sum(
                    (p / sum(persistences)) * math.log(p / sum(persistences))
                    for p in persistences
                    if p / sum(persistences) > 0
                )
                max_h = math.log(len(persistences)) if len(persistences) > 1 else 1.0
                h_norm = h_ent / max(max_h, 1e-10)
        h_norm = max(0.0, min(1.0, h_norm))

        e_void = max(self._state.void_energy, 1e-10)
        e_ratio = (self._state.dark_energy + self._state.channel_energy) / e_void
        e_ratio = max(0.0, min(1.0, e_ratio))

        rates = [
            self._state.h3.growth_rate,
            self._state.h4.growth_rate,
            self._state.h5.growth_rate,
            self._state.h6.growth_rate,
        ]
        mu = sum(abs(r) for r in rates) / max(len(rates), 1)
        sigma = math.sqrt(sum((abs(r) - mu) ** 2 for r in rates) / max(len(rates), 1))
        sync = 1.0 - sigma / max(mu, 1e-10)
        sync = max(0.0, min(1.0, sync))

        psi = self._psi_field

        w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
        return max(0.0, min(1.0, w1 * h_norm + w2 * e_ratio + w3 * sync + w4 * psi))

    def _detect_phase_transition(self) -> list:
        transitions = []
        n_total = max(1, self._field._occupied_count)
        e_void = max(self._state.void_energy, 0.01)
        now = time.time()

        h4 = self._state.h4
        if (
            h4.growth_rate > 0.15
            and h4.energy > 3.5
            and self._state.coherence > 0.75
        ):
            pre = {"h4_count": h4.count, "h4_energy": h4.energy}
            transitions.append(
                {
                    "level": 4,
                    "energy": h4.energy,
                    "coherence": self._state.coherence,
                    "trigger_condition": f"d|H4|/dt={h4.growth_rate:.3f} > 0.15, E_multi={h4.energy:.2f} > 3.5, C={self._state.coherence:.3f} > 0.75",
                    "pre_state": pre,
                    "post_state": {},
                }
            )

        h5 = self._state.h5
        if (
            h5.count > 0.1 * n_total
            and abs(self._h5_regulation) > 0.7
            and self._h5_high_duration >= 5
        ):
            pre = {"h5_count": h5.count, "h5_regulation": self._h5_regulation}
            transitions.append(
                {
                    "level": 5,
                    "energy": h5.energy,
                    "coherence": self._state.coherence,
                    "trigger_condition": f"|H5|={h5.count:.2f} > 0.1*N, |Reg|={abs(self._h5_regulation):.3f} > 0.7 for {self._h5_high_duration} cycles",
                    "pre_state": pre,
                    "post_state": {},
                }
            )

        h6 = self._state.h6
        if (
            h6.count > 0.05 * n_total
            and h6.energy > 2.0 * e_void
            and self._psi_field > 0.6
        ):
            pre = {
                "h6_count": h6.count,
                "h6_energy": h6.energy,
                "psi_field": self._psi_field,
                "h3": {"count": self._state.h3.count, "energy": self._state.h3.energy},
                "h4": {"count": h4.count, "energy": h4.energy},
                "h5": {"count": h5.count, "energy": h5.energy},
            }

            self._state.h3 = HighDimState(count=self._state.h3.count * 0.3, energy=self._state.h3.energy * 0.3, growth_rate=0.0)
            self._state.h4 = HighDimState(count=h4.count * 0.3, energy=h4.energy * 0.3, growth_rate=0.0)
            self._state.h5 = HighDimState(count=h5.count * 0.3, energy=h5.energy * 0.3, growth_rate=0.0)
            self._state.h6 = HighDimState(count=h6.count * 0.3, energy=h6.energy * 0.3, growth_rate=0.0)
            self._psi_field *= 0.3

            post = {
                "h3": {"count": self._state.h3.count, "energy": self._state.h3.energy},
                "h4": {"count": self._state.h4.count, "energy": self._state.h4.energy},
                "h5": {"count": self._state.h5.count, "energy": self._state.h5.energy},
                "h6": {"count": self._state.h6.count, "energy": self._state.h6.energy},
                "psi_field": self._psi_field,
            }

            transitions.append(
                {
                    "level": 6,
                    "energy": h6.energy,
                    "coherence": self._state.coherence,
                    "trigger_condition": f"|H6|={h6.count:.3f} > 0.05*N, E6={h6.energy:.2f} > 2*E_void, Psi={self._psi_field:.3f} > 0.6",
                    "pre_state": pre,
                    "post_state": post,
                }
            )
            self._h5_high_duration = 0

        for pt in transitions:
            event = PhaseTransitionEvent(
                timestamp=now,
                level=pt["level"],
                energy=pt["energy"],
                coherence=pt["coherence"],
                trigger_condition=pt["trigger_condition"],
                pre_state=pt["pre_state"],
                post_state=pt["post_state"],
            )
            self._state.phase_transitions.append(event)
            if len(self._state.phase_transitions) > 100:
                self._state.phase_transitions = self._state.phase_transitions[-100:]
            self._state.last_phase_transition = now
            self._state.total_phase_transitions += 1

        return transitions

    @staticmethod
    def _euler_step(current, derivative, dt, bounds=(0, float("inf"))):
        new_val = current + derivative * dt
        return max(bounds[0], min(bounds[1], new_val))

    def get_projection_data(self) -> dict:
        return {
            "void_energy": self._state.void_energy,
            "dark_energy": self._state.dark_energy,
            "channel_energy": self._state.channel_energy,
            "coherence": self._state.coherence,
            "h5_regulation": self._h5_regulation,
            "h6_cascade_strength": self._state.h6.count / max(0.01 * self._field._occupied_count, 1.0),
            "psi_field": self._psi_field,
        }

    def get_regulation_signals(self) -> dict:
        return {
            "persistent_entropy": self._state.persistent_entropy,
            "coherence": self._state.coherence,
            "h4_growth_rate": self._state.h4.growth_rate,
            "h5_regulation": self._h5_regulation,
            "h6_cascade_active": self._state.h6.count > 0.05 * max(1, self._field._occupied_count),
            "psi_field": self._psi_field,
            "total_dim_energy": self._state.total_dim_energy,
        }

    def get_stats(self) -> dict:
        return {
            "cycle_count": self._cycle_count,
            "ph_interval": self._ph_interval,
            "ph_sample_size": self._ph_sample_size,
            "features": {
                "h0_count": len(self._state.features_h0),
                "h1_count": len(self._state.features_h1),
                "h2_count": len(self._state.features_h2),
            },
            "void_energy": self._state.void_energy,
            "dark_energy": self._state.dark_energy,
            "channel_energy": self._state.channel_energy,
            "persistent_entropy": self._state.persistent_entropy,
            "h3": {"count": self._state.h3.count, "energy": self._state.h3.energy, "growth_rate": self._state.h3.growth_rate},
            "h4": {"count": self._state.h4.count, "energy": self._state.h4.energy, "growth_rate": self._state.h4.growth_rate},
            "h5": {"count": self._state.h5.count, "energy": self._state.h5.energy, "growth_rate": self._state.h5.growth_rate},
            "h6": {"count": self._state.h6.count, "energy": self._state.h6.energy, "growth_rate": self._state.h6.growth_rate},
            "coherence": self._state.coherence,
            "total_dim_energy": self._state.total_dim_energy,
            "cascade_potential": self._state.cascade_potential,
            "psi_field": self._psi_field,
            "h5_regulation": self._h5_regulation,
            "total_phase_transitions": self._state.total_phase_transitions,
            "last_phase_transition": self._state.last_phase_transition,
            "cross_dim_coupling": self._compute_cross_dim_coupling(),
        }

    def get_state(self) -> dict:
        return {
            "features_h0": [
                {"birth": f.birth, "death": f.death, "persistence": f.persistence, "topo_charge": f.topo_charge}
                for f in self._state.features_h0[: self._max_features_persist]
            ],
            "features_h1": [
                {"birth": f.birth, "death": f.death, "persistence": f.persistence, "topo_charge": f.topo_charge}
                for f in self._state.features_h1[: self._max_features_persist]
            ],
            "features_h2": [
                {"birth": f.birth, "death": f.death, "persistence": f.persistence, "topo_charge": f.topo_charge}
                for f in self._state.features_h2[: self._max_features_persist]
            ],
            "void_energy": self._state.void_energy,
            "dark_energy": self._state.dark_energy,
            "channel_energy": self._state.channel_energy,
            "persistent_entropy": self._state.persistent_entropy,
            "h3": {"count": self._state.h3.count, "energy": self._state.h3.energy, "growth_rate": self._state.h3.growth_rate},
            "h4": {"count": self._state.h4.count, "energy": self._state.h4.energy, "growth_rate": self._state.h4.growth_rate},
            "h5": {"count": self._state.h5.count, "energy": self._state.h5.energy, "growth_rate": self._state.h5.growth_rate},
            "h6": {"count": self._state.h6.count, "energy": self._state.h6.energy, "growth_rate": self._state.h6.growth_rate},
            "coherence": self._state.coherence,
            "total_dim_energy": self._state.total_dim_energy,
            "cascade_potential": self._state.cascade_potential,
            "phase_transitions": [
                {
                    "timestamp": pt.timestamp,
                    "level": pt.level,
                    "energy": pt.energy,
                    "coherence": pt.coherence,
                    "trigger_condition": pt.trigger_condition,
                }
                for pt in self._state.phase_transitions[-50:]
            ],
            "last_phase_transition": self._state.last_phase_transition,
            "total_phase_transitions": self._state.total_phase_transitions,
            "h5_regulation": self._h5_regulation,
            "psi_field": self._psi_field,
        }

    def set_state(self, state: dict):
        self._state.void_energy = float(state.get("void_energy", 0.0))
        self._state.dark_energy = float(state.get("dark_energy", 0.0))
        self._state.channel_energy = float(state.get("channel_energy", 0.0))
        self._state.persistent_entropy = float(state.get("persistent_entropy", 0.0))
        self._state.coherence = float(state.get("coherence", 0.0))
        self._state.total_dim_energy = float(state.get("total_dim_energy", 0.0))
        self._state.cascade_potential = float(state.get("cascade_potential", 0.0))
        self._state.last_phase_transition = float(state.get("last_phase_transition", 0.0))
        self._state.total_phase_transitions = int(state.get("total_phase_transitions", 0))
        self._h5_regulation = float(state.get("h5_regulation", 0.0))
        self._psi_field = float(state.get("psi_field", 0.0))

        for dim in ("h3", "h4", "h5", "h6"):
            raw = state.get(dim, {})
            hs = getattr(self._state, dim)
            hs.count = float(raw.get("count", 0.0))
            hs.energy = float(raw.get("energy", 0.0))
            hs.growth_rate = float(raw.get("growth_rate", 0.0))

        self._state.features_h0 = []
        for fd in state.get("features_h0", []):
            self._state.features_h0.append(
                HomologyFeature(
                    birth=fd.get("birth", 0), death=fd.get("death", 0),
                    dimension=0, persistence=fd.get("persistence", 0),
                    topo_charge=fd.get("topo_charge", 0), participating_nodes=[],
                )
            )
        self._state.features_h1 = []
        for fd in state.get("features_h1", []):
            self._state.features_h1.append(
                HomologyFeature(
                    birth=fd.get("birth", 0), death=fd.get("death", 0),
                    dimension=1, persistence=fd.get("persistence", 0),
                    topo_charge=fd.get("topo_charge", 0), participating_nodes=[],
                )
            )
        self._state.features_h2 = []
        for fd in state.get("features_h2", []):
            self._state.features_h2.append(
                HomologyFeature(
                    birth=fd.get("birth", 0), death=fd.get("death", 0),
                    dimension=2, persistence=fd.get("persistence", 0),
                    topo_charge=fd.get("topo_charge", 0), participating_nodes=[],
                )
            )

        self._state.phase_transitions = []
        for pt in state.get("phase_transitions", []):
            self._state.phase_transitions.append(
                PhaseTransitionEvent(
                    timestamp=pt.get("timestamp", 0),
                    level=pt.get("level", 0),
                    energy=pt.get("energy", 0),
                    coherence=pt.get("coherence", 0),
                    trigger_condition=pt.get("trigger_condition", ""),
                    pre_state={}, post_state={},
                )
            )
