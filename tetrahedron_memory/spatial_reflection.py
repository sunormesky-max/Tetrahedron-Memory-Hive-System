from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np


class SpatialReflectionField:
    """
    Spatial Reflection Force Field for the BCC Honeycomb Neural Field.

    Computes a per-node potential energy based on local geometric harmony:
    - Label coherence with neighbors (semantic gravity)
    - Tetrahedral cell quality (geometric fitness)
    - Local density balance (crowding repulsion)
    - Crystal channel alignment (pathway resonance)
    - Pulse accumulator pressure (neural tension)

    The field does NOT move nodes (BCC lattice positions are fixed).
    Instead, it modulates:
    - Pulse propagation direction weights (anisotropic decay)
    - Dream source selection (high-tension = creative potential)
    - Query scoring (low-energy = high confidence)
    - Memory placement preference (equilibrium positions)
    - Phase transition detection (energy regime changes)
    """

    def __init__(self):
        self._node_energy: Dict[str, float] = {}
        self._node_gradient: Dict[str, np.ndarray] = {}
        self._field_entropy = 1.0
        self._total_reflections = 0
        self._last_reflection_time = 0.0
        self._energy_history: List[float] = []
        self._phase_state = "fluid"
        self._phase_transitions = 0

    def compute_node_energy(self, field, nid: str) -> float:
        node = field._nodes.get(nid)
        if node is None:
            return 0.5
        e_label = self._label_coherence(field, nid, node)
        e_cell = self._cell_quality_energy(field, nid)
        e_density = self._density_balance(field, nid, node)
        e_crystal = self._crystal_resonance(field, nid, node)
        energy = 0.30 * e_label + 0.25 * e_cell + 0.25 * e_density + 0.20 * e_crystal
        return min(1.0, max(0.0, energy))

    def _label_coherence(self, field, nid: str, node) -> float:
        if not node.is_occupied or not node.labels:
            return 0.5
        node_labels = set(l for l in node.labels if not l.startswith("__"))
        if not node_labels:
            return 0.5
        coherent = 0
        total = 0
        for fnid in node.face_neighbors[:8]:
            fn = field._nodes.get(fnid)
            if fn and fn.is_occupied:
                fn_labels = set(l for l in fn.labels if not l.startswith("__"))
                if fn_labels:
                    overlap = len(node_labels & fn_labels)
                    union = len(node_labels | fn_labels)
                    coherent += overlap / max(union, 1)
                    total += 1
        if total == 0:
            return 0.5
        return coherent / total

    def _cell_quality_energy(self, field, nid: str) -> float:
        cells = field._cell_map.get_cells_for_node(nid)
        if not cells:
            return 0.5
        best_quality = max(c.quality for c in cells)
        return 1.0 - best_quality

    def _density_balance(self, field, nid: str, node) -> float:
        occupied_weights = []
        total_count = 0
        for fnid in node.face_neighbors[:8]:
            fn = field._nodes.get(fnid)
            if fn:
                total_count += 1
                if fn.is_occupied:
                    occupied_weights.append(fn.weight)
        if total_count == 0:
            return 0.5
        density = len(occupied_weights) / total_count
        if density > 0.8 and occupied_weights:
            return min(1.0, float(np.std(occupied_weights) / max(np.mean(occupied_weights), 0.1)))
        return abs(density - 0.5) * 2.0

    def _crystal_resonance(self, field, nid: str, node) -> float:
        if not node.crystal_channels:
            return 0.5
        crystal_sum = sum(node.crystal_channels.values())
        return min(1.0, crystal_sum / 5.0)

    def run_reflection_cycle(self, field) -> Dict[str, Any]:
        with field._lock:
            old_entropy = self._field_entropy
            energies = []
            gradient_sum = np.zeros(3, dtype=np.float32)

            for nid, node in field._nodes.items():
                if not node.is_occupied:
                    continue
                energy = self.compute_node_energy(field, nid)
                self._node_energy[nid] = energy
                energies.append(energy)

                if node.is_occupied and len(node.face_neighbors) > 0:
                    gradient = np.zeros(3, dtype=np.float32)
                    for fnid in node.face_neighbors[:6]:
                        fn = field._nodes.get(fnid)
                        if fn:
                            fn_e = self._node_energy.get(fnid, 0.5)
                            direction = fn.position - node.position
                            dist = float(np.linalg.norm(direction))
                            if dist > 0:
                                gradient += direction / dist * (fn_e - energy)
                    self._node_gradient[nid] = gradient
                    gradient_sum += np.abs(gradient)

            if energies:
                self._field_entropy = float(np.std(energies))
            else:
                self._field_entropy = 0.0

            avg_energy = float(np.mean(energies)) if energies else 0.5
            self._energy_history.append(avg_energy)
            if len(self._energy_history) > 100:
                self._energy_history = self._energy_history[-50:]

            prev_phase = self._phase_state
            if avg_energy < 0.2 and self._field_entropy < 0.1:
                self._phase_state = "crystalline"
            elif avg_energy < 0.35 and self._field_entropy < 0.2:
                self._phase_state = "ordered"
            elif avg_energy > 0.6 and self._field_entropy > 0.3:
                self._phase_state = "turbulent"
            else:
                self._phase_state = "fluid"

            if prev_phase != self._phase_state:
                self._phase_transitions += 1

            self._total_reflections += 1
            self._last_reflection_time = time.time()

            high_tension = [nid for nid, e in self._node_energy.items() if e > 0.7]
            low_tension = [nid for nid, e in self._node_energy.items() if e < 0.2]

            self._apply_field_adjustments(field, high_tension, low_tension)

            return {
                "avg_energy": round(avg_energy, 4),
                "field_entropy": round(self._field_entropy, 4),
                "phase_state": self._phase_state,
                "phase_changed": prev_phase != self._phase_state,
                "high_tension_nodes": len(high_tension),
                "low_tension_nodes": len(low_tension),
                "total_reflections": self._total_reflections,
                "gradient_magnitude": round(float(np.mean(np.abs(gradient_sum))), 4),
            }

    def _apply_field_adjustments(self, field, high_tension: List[str], low_tension: List[str]):
        for nid in high_tension[:5]:
            node = field._nodes.get(nid)
            if node and node.is_occupied:
                node.base_activation = min(node.base_activation + 0.01, 1.0)
                if node.weight > 0.5:
                    geo_quality = field._compute_node_geometric_quality(nid)
                    if geo_quality < 0.4:
                        for enid in node.edge_neighbors[:4]:
                            en = field._nodes.get(enid)
                            if en and not en.is_occupied:
                                for fnid in en.face_neighbors[:4]:
                                    fn = field._nodes.get(fnid)
                                    if fn and fn.is_occupied:
                                        shared = len(set(node.labels) & set(fn.labels))
                                        if shared > 0:
                                            field._hebbian.record_path([nid, enid, fnid], True, 0.05 * shared)
                                        break
                    else:
                        for fnid in node.face_neighbors[:3]:
                            fn = field._nodes.get(fnid)
                            if fn and fn.is_occupied:
                                shared = len(set(node.labels) & set(fn.labels))
                                if shared > 0:
                                    field._hebbian.record_path([nid, fnid], True, 0.1 * shared)

        for nid in low_tension[:5]:
            node = field._nodes.get(nid)
            if node and node.is_occupied:
                node.base_activation = max(node.base_activation - 0.005, 0.01)
                geo_q = field._compute_node_geometric_quality(nid)
                node.metadata["geometric_quality"] = geo_q

    def get_node_energy(self, nid: str) -> float:
        return self._node_energy.get(nid, 0.5)

    def get_pulse_direction_bias(self, field, from_id: str, to_id: str) -> float:
        from_e = self._node_energy.get(from_id, 0.5)
        to_e = self._node_energy.get(to_id, 0.5)
        gradient = self._node_gradient.get(from_id)
        from_node = field._nodes.get(from_id)
        to_node = field._nodes.get(to_id)
        if gradient is not None and from_node is not None and to_node is not None:
            direction = to_node.position - from_node.position
            dist = float(np.linalg.norm(direction))
            if dist > 0:
                alignment = float(np.dot(gradient, direction / dist))
                return 1.0 + alignment * 0.3
        return 1.0 + (to_e - from_e) * 0.2

    def get_dream_tension(self, nid: str) -> float:
        return self._node_energy.get(nid, 0.5)

    def get_spatial_quality(self, field, nid: str) -> float:
        energy = self._node_energy.get(nid, 0.5)
        return 1.0 - energy

    def stats(self) -> Dict[str, Any]:
        return {
            "phase_state": self._phase_state,
            "phase_transitions": self._phase_transitions,
            "field_entropy": round(self._field_entropy, 4),
            "total_reflections": self._total_reflections,
            "avg_energy": round(float(np.mean(list(self._node_energy.values()))), 4) if self._node_energy else 0.5,
            "energy_history_len": len(self._energy_history),
        }
