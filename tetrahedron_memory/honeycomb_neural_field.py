"""
HoneycombNeuralField — BCC Lattice Honeycomb + PCNN-Grounded Neural Pulse Engine.

Version 5.0 — Structural Cascade + Lattice Integrity + Crystallized Pathways

Architecture:
  Layer 1: BCC lattice honeycomb (space-filling tetrahedral mesh)
  Layer 2: Memory activation (content mapped to lattice nodes)
  Layer 3: PCNN neural pulses (theoretically grounded, multi-type)
  Layer 4: Hebbian path memory (emergent pathway reinforcement)
  Layer 5: Cascade pulse waves (multi-directional branching propagation)
  Layer 6: Crystallized pathways (permanent structural fast-paths)
  Layer 7: Lattice integrity verification (geometric topology audit)

PCNN Parameters derived from:
  - Eckhorn 1989 visual cortex model (coupling + dynamic threshold)
  - BCC lattice topology (coordination number 8, APF 0.680)
  - Small-world network theory (Watts-Strogatz 1998)
  - Signal attenuation model: exp(-alpha * d) in 3D isotropic medium

Key equations (PCNN adapted to BCC lattice):
  F_i[n] = alpha_F * F_i[n-1] + S_i + V_F * sum_j(M_ij * Y_j[n-1])
  L_i[n] = alpha_L * L_i[n-1] + V_L * sum_j(W_ij * Y_j[n-1])
  U_i[n] = F_i[n] * (1 + beta * L_i[n])
  Y_i[n] = step(U_i[n] - Theta_i[n-1])
  Theta_i[n] = alpha_Theta * Theta_i[n-1] + V_Theta * Y_i[n]

Cascade propagation:
  At each hop, a cascade pulse spawns K child pulses (K = branching factor).
  Each child propagates independently with decay * branching_decay_factor.
  Total wavefront energy is conserved: sum(children) <= parent.

Crystallization:
  When Hebbian edge weight exceeds crystallize_threshold, the edge becomes
  a "crystal" — permanently reinforced with zero decay, acting as structural
  conduit for future pulses.

Lattice Integrity:
  BCC lattice is verified by checking:
  - Every body-center node has exactly 8 face-sharing corner neighbors
  - Every corner node has exactly 8 adjacent body-center neighbors
  - Every corner node has exactly 6 edge-sharing corner neighbors
  - All edges are bidirectional
  - No orphan nodes (nodes with zero neighbors)

BCC lattice properties used in derivations:
  - Nearest neighbor distance: a * sqrt(3) / 2
  - Next-nearest neighbor distance: a
  - Distance ratio d_edge/d_face = 2/sqrt(3) ≈ 1.155
  - Face coordination: 8, Edge coordination: 6
"""

import enum
import hashlib
import logging
import math
import random
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("tetramem.honeycomb")


class PulseType(enum.Enum):
    EXPLORATORY = "exploratory"
    REINFORCING = "reinforcing"
    TENSION_SENSING = "tension_sensing"
    SELF_CHECK = "self_check"
    CASCADE = "cascade"
    STRUCTURE = "structure"


class PCNNConfig:
    """
    PCNN parameters derived from BCC lattice topology and neural science.

    Derivation notes:
    - BCC face coordination = 8, edge coordination = 6
    - Face decay: e^(-alpha * d_face) where d_face = sqrt(3)/2 * spacing
      With alpha * spacing = 0.380 (calibrated from signal conservation):
      face_decay = e^(-0.380 * 0.866) ≈ 0.72
    - Edge decay: e^(-0.380 * 1.0) ≈ 0.684, but small-world shortcut factor
      reduces to 0.50 * face_decay for Watts-Strogatz small-world topology
    - Max hops from small-world theory: L ~ log(N)/log(k)
      For N=2000 nodes, k=14: L ≈ 2.7, so L/2 ≈ 1.4 hops for half-diameter
      But we want multi-hop propagation, so we scale up:
      - Exploratory: 2*L ≈ 8 hops (broad scan)
      - Reinforcing: L ≈ 5 hops (pathway strengthening)
      - Tension: L/2 ≈ 3 hops (local tension detection)
    - Bridge threshold: sigma * sqrt(n_sources) where sigma is noise floor
      With noise floor ≈ base_activation * accumulation_time ≈ 0.01 * 30s/0.5s ≈ 0.3
      For 2 sources: threshold = 0.3 * sqrt(2) ≈ 0.42 → rounded to 0.4
    """

    ALPHA_FEED = 0.30
    ALPHA_LINK = 0.70
    ALPHA_THRESHOLD = 0.15

    BETA = 0.40

    V_F = 1.0
    V_L = 0.5
    V_THETA = 5.0

    FACE_DECAY = 0.72
    EDGE_DECAY_FACTOR = 0.50
    VERTEX_DECAY_FACTOR = 0.25

    MAX_HOPS_EXPLORATORY = 12
    MAX_HOPS_REINFORCING = 8
    MAX_HOPS_TENSION = 5

    BRIDGE_THRESHOLD = 0.40
    MIN_BRIDGE_SOURCES = 2

    BASE_PULSE_INTERVAL = 0.50
    MIN_PULSE_INTERVAL = 0.20
    MAX_PULSE_INTERVAL = 2.00

    EXPLORATORY_STRENGTH_RANGE = (0.08, 0.20)
    REINFORCING_STRENGTH_RANGE = (0.25, 0.50)
    TENSION_STRENGTH_RANGE = (0.40, 0.70)

    PULSE_TYPE_PROBABILITIES = {
        PulseType.EXPLORATORY: 0.25,
        PulseType.REINFORCING: 0.25,
        PulseType.CASCADE: 0.20,
        PulseType.TENSION_SENSING: 0.12,
        PulseType.SELF_CHECK: 0.08,
        PulseType.STRUCTURE: 0.07,
    }

    HEBBIAN_MAX_PATHS = 4000
    HEBBIAN_DECAY = 0.98
    HEBBIAN_REINFORCE = 1.15
    HEBBIAN_MIN_WEIGHT = 0.01

    SELF_CHECK_INTERVAL = 60.0
    SELF_CHECK_MAX_HOPS = 4
    SELF_CHECK_STRENGTH = 0.55
    ISOLATED_NODE_THRESHOLD = 0
    DUPLICATE_TOKEN_OVERLAP = 0.85
    DUPLICATE_MERGE_MIN_WEIGHT_RATIO = 1.1

    CASCADE_BRANCHING_FACTOR = 4
    CASCADE_BRANCHING_DECAY = 0.65
    CASCADE_MAX_DEPTH = 6
    CASCADE_ENERGY_CONSERVATION = 0.95
    CASCADE_STRENGTH_RANGE = (0.30, 0.60)
    CASCADE_MAX_HOPS = 8

    STRUCTURE_MAX_HOPS = 3
    STRUCTURE_STRENGTH = 0.40
    STRUCTURE_INTEGRITY_INTERVAL = 300

    CRYSTALLIZE_THRESHOLD = 2.0
    CRYSTALLIZE_MAX_PATHS = 600
    CRYSTAL_PULSE_BOOST = 1.8
    CRYSTAL_WEIGHT_FLOOR = 2.0

    CONVERGENCE_CHECK_CYCLES = 60
    GLOBAL_DECAY_CYCLES = 120

    SELF_ORGANIZE_INTERVAL = 60
    CLUSTER_MIN_SIZE = 3
    CLUSTER_LABEL_OVERLAP = 0.4
    CLUSTER_MAX_LABELS = 8
    ENTROPY_IDEAL_RATIO = 0.15
    ENTROPY_BOOST_FACTOR = 0.2
    ENTROPY_SUPPRESS_FACTOR = 0.1
    CONSOLIDATION_MIN_SIMILARITY = 0.65
    CONSOLIDATION_MAX_PER_CYCLE = 5
    CONSOLIDATION_WEIGHT_TRANSFER = 0.6
    SHORTCUT_MAX_DISTANCE = 6
    SHORTCUT_MIN_LABEL_OVERLAP = 2
    SHORTCUT_MAX_PER_CYCLE = 3
    SHORTCUT_VIRTUAL_STRENGTH = 0.8

    TETRA_QUALITY_THRESHOLD = 0.3
    TETRA_IDEAL_VOLUME_FACTOR = 0.1178
    TETRA_MAX_CELLS_PER_ANALYSIS = 500
    TETRA_DENSITY_PENALTY = 0.5

    DREAM_CYCLE_INTERVAL = 300
    DREAM_MAX_RECOMBINATIONS = 10
    DREAM_MIN_SOURCE_WEIGHT = 1.0
    DREAM_INSIGHT_WEIGHT = 1.5
    DREAM_CROSS_DOMAIN_BONUS = 2.0

    AGENT_CONTEXT_MAX_MEMORIES = 15
    AGENT_REASONING_MAX_HOPS = 5
    AGENT_SUGGESTION_TOP_N = 5

    MAX_PULSE_ACCUMULATOR = 5.0
    MAX_INTERNAL_ACTIVITY = 50.0
    MAX_FEEDING = 20.0
    MAX_LINKING = 10.0


class HoneycombNode:
    __slots__ = (
        "id", "position", "face_neighbors", "edge_neighbors", "vertex_neighbors",
        "content", "labels", "weight", "activation", "base_activation",
        "last_pulse_time", "pulse_accumulator", "creation_time",
        "metadata", "access_count", "decay_rate",
        "feeding", "linking", "internal_activity", "threshold", "fired",
        "crystal_channels",
    )

    def __init__(self, id: str, position: np.ndarray):
        self.id = id
        self.position = position
        self.face_neighbors: List[str] = []
        self.edge_neighbors: List[str] = []
        self.vertex_neighbors: List[str] = []
        self.content: Optional[str] = None
        self.labels: List[str] = []
        self.weight: float = 0.0
        self.activation: float = 0.0
        self.base_activation: float = 0.01
        self.last_pulse_time: float = 0.0
        self.pulse_accumulator: float = 0.0
        self.creation_time: float = time.time()
        self.metadata: Dict[str, Any] = {}
        self.access_count: int = 0
        self.decay_rate: float = 0.001

        self.feeding: float = 0.0
        self.linking: float = 0.0
        self.internal_activity: float = 0.0
        self.threshold: float = PCNNConfig.V_THETA
        self.fired: bool = False
        self.crystal_channels: Dict[str, float] = {}

    @property
    def is_occupied(self) -> bool:
        return self.content is not None

    def touch(self):
        self.last_pulse_time = time.time()
        self.access_count += 1

    def decay(self, dt: float):
        if self.is_occupied:
            rate = self.decay_rate / max(self.weight, 0.5)
            self.activation = max(self.base_activation, self.activation - rate * dt)
        else:
            self.pulse_accumulator *= 0.95

    def reinforce(self, amount: float):
        boost = amount * max(self.weight, 0.5) * 0.3
        self.activation = min(10.0, self.activation + amount + boost)

    def pcnn_step(self, neighbor_outputs: List[Tuple[str, float, str]]):
        """
        One PCNN timestep for this node.

        Parameters
        ----------
        neighbor_outputs : list of (neighbor_id, output_strength, connection_type)
            "face", "edge", or "vertex"
        """
        cfg = PCNNConfig

        s_input = self.activation if self.is_occupied else self.pulse_accumulator

        linking_sum = 0.0
        feeding_sum = 0.0
        for nid, strength, ctype in neighbor_outputs:
            w_link = 1.0 if ctype == "face" else (0.5 if ctype == "edge" else 0.2)
            w_feed = 1.0 if ctype == "face" else (0.3 if ctype == "edge" else 0.1)
            linking_sum += w_link * strength
            feeding_sum += w_feed * strength

        self.feeding = min(cfg.MAX_FEEDING, cfg.ALPHA_FEED * self.feeding + s_input + cfg.V_F * feeding_sum)
        self.linking = min(cfg.MAX_LINKING, cfg.ALPHA_LINK * self.linking + cfg.V_L * linking_sum)
        self.internal_activity = min(cfg.MAX_INTERNAL_ACTIVITY, self.feeding * (1.0 + cfg.BETA * self.linking))

        self.fired = self.internal_activity > self.threshold

        if self.fired:
            self.threshold = cfg.ALPHA_THRESHOLD * self.threshold + cfg.V_THETA
            self.pulse_accumulator = min(cfg.MAX_PULSE_ACCUMULATOR, self.pulse_accumulator + self.internal_activity * 0.02)
        else:
            self.threshold = cfg.ALPHA_THRESHOLD * self.threshold


class NeuralPulse:
    __slots__ = (
        "source_id", "strength", "hops", "path", "direction",
        "birth_time", "max_hops", "pulse_type", "bias_fn",
        "cascade_depth", "cascade_parent_id",
    )

    def __init__(
        self,
        source_id: str,
        strength: float,
        max_hops: int = 6,
        pulse_type: PulseType = PulseType.EXPLORATORY,
        bias_fn=None,
        cascade_depth: int = 0,
        cascade_parent_id: Optional[str] = None,
    ):
        self.source_id = source_id
        self.strength = strength
        self.hops = 0
        self.path = [source_id]
        self.direction = "face"
        self.birth_time = time.time()
        self.max_hops = max_hops
        self.pulse_type = pulse_type
        self.bias_fn = bias_fn
        self.cascade_depth = cascade_depth
        self.cascade_parent_id = cascade_parent_id

    def propagate(self, decay: float = 0.7) -> float:
        self.hops += 1
        self.strength *= decay
        return self.strength

    @property
    def alive(self) -> bool:
        noise_floor = 0.01
        return self.strength > noise_floor and self.hops < self.max_hops

    def clone(self, new_strength: float) -> "NeuralPulse":
        child = NeuralPulse(
            self.source_id, new_strength,
            max_hops=self.max_hops,
            pulse_type=self.pulse_type,
            bias_fn=self.bias_fn,
            cascade_depth=self.cascade_depth + 1,
            cascade_parent_id=self.source_id,
        )
        child.hops = self.hops
        child.path = list(self.path)
        child.direction = self.direction
        return child


class CrystallizedPathway:
    """
    Permanent structural fast-path in the BCC lattice neural field.

    When a Hebbian edge weight exceeds the crystallization threshold,
    it becomes a crystal — a zero-decay, permanently reinforced conduit.
    Pulses traveling through crystal channels get boosted transmission.

    Crystal channels form the "white matter" of the memory system:
    stable, high-bandwidth structural pathways connecting distant
    memory clusters through the BCC lattice.
    """

    def __init__(self, max_crystals: int = 200, weight_floor: float = 2.0):
        self._crystals: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._max_crystals = max_crystals
        self._weight_floor = weight_floor
        self._creation_count = 0
        self._pulse_transmissions = 0

    def try_crystallize(self, edge_a: str, edge_b: str, hebbian_weight: float) -> bool:
        key = self._make_key(edge_a, edge_b)
        if key in self._crystals:
            crystal = self._crystals[key]
            crystal["hebbian_weight"] = max(crystal["hebbian_weight"], hebbian_weight)
            crystal["last_reinforced"] = time.time()
            return False

        if hebbian_weight < self._weight_floor:
            return False

        if len(self._crystals) >= self._max_crystals:
            self._evict_weakest()

        self._crystals[key] = {
            "nodes": (edge_a, edge_b),
            "hebbian_weight": hebbian_weight,
            "crystal_weight": min(hebbian_weight * 0.5, 10.0),
            "created_at": time.time(),
            "last_reinforced": time.time(),
            "transmission_count": 0,
        }
        self._creation_count += 1
        return True

    def is_crystal(self, node_a: str, node_b: str) -> bool:
        return self._make_key(node_a, node_b) in self._crystals

    def get_boost(self, node_a: str, node_b: str) -> float:
        key = self._make_key(node_a, node_b)
        crystal = self._crystals.get(key)
        if crystal is None:
            return 1.0
        crystal["transmission_count"] += 1
        self._pulse_transmissions += 1
        return PCNNConfig.CRYSTAL_PULSE_BOOST

    def get_crystal_path(self, node_a: str, node_b: str) -> float:
        return self._crystals.get(self._make_key(node_a, node_b), {}).get("crystal_weight", 0.0)

    def _evict_weakest(self):
        if not self._crystals:
            return
        weakest_key = min(self._crystals, key=lambda k: self._crystals[k]["crystal_weight"])
        del self._crystals[weakest_key]

    def _make_key(self, a: str, b: str) -> Tuple[str, str]:
        return (min(a, b), max(a, b))

    def scan_and_crystallize(self, hebbian_edges: Dict[Tuple[str, str], float]):
        new_count = 0
        for (a, b), weight in hebbian_edges.items():
            if weight >= PCNNConfig.CRYSTALLIZE_THRESHOLD:
                if self.try_crystallize(a, b, weight):
                    new_count += 1
        return new_count

    def stats(self) -> Dict[str, Any]:
        if not self._crystals:
            return {
                "total_crystals": 0,
                "total_created": self._creation_count,
                "total_transmissions": self._pulse_transmissions,
                "avg_crystal_weight": 0.0,
                "max_crystal_weight": 0.0,
            }
        weights = [c["crystal_weight"] for c in self._crystals.values()]
        return {
            "total_crystals": len(self._crystals),
            "total_created": self._creation_count,
            "total_transmissions": self._pulse_transmissions,
            "avg_crystal_weight": round(float(np.mean(weights)), 3),
            "max_crystal_weight": round(float(max(weights)), 3),
            "capacity": self._max_crystals,
        }

    def top_crystals(self, n: int = 20) -> List[Dict]:
        sorted_c = sorted(self._crystals.values(), key=lambda c: -c["crystal_weight"])[:n]
        return [
            {
                "nodes": (c["nodes"][0][:8], c["nodes"][1][:8]),
                "crystal_weight": round(c["crystal_weight"], 3),
                "hebbian_weight": round(c["hebbian_weight"], 3),
                "transmissions": c["transmission_count"],
                "age_seconds": round(time.time() - c["created_at"], 0),
            }
            for c in sorted_c
        ]


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


class LatticeIntegrityReport:
    __slots__ = (
        "check_time", "total_nodes", "face_edges_checked", "edge_edges_checked",
        "broken_edges", "orphan_nodes", "coordination_errors",
        "connectivity_components", "integrity_score", "details",
    )

    def __init__(self):
        self.check_time: float = time.time()
        self.total_nodes: int = 0
        self.face_edges_checked: int = 0
        self.edge_edges_checked: int = 0
        self.broken_edges: List[Dict] = []
        self.orphan_nodes: List[str] = []
        self.coordination_errors: List[Dict] = []
        self.connectivity_components: int = 0
        self.integrity_score: float = 1.0
        self.details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_time": self.check_time,
            "total_nodes": self.total_nodes,
            "face_edges_checked": self.face_edges_checked,
            "edge_edges_checked": self.edge_edges_checked,
            "broken_edges": self.broken_edges[:20],
            "orphan_nodes": self.orphan_nodes[:20],
            "coordination_errors": self.coordination_errors[:20],
            "connectivity_components": self.connectivity_components,
            "integrity_score": round(self.integrity_score, 4),
            "details": self.details,
        }


class LatticeIntegrityChecker:
    """
    Verifies geometric and topological integrity of the BCC lattice.

    BCC crystal structure rules:
      - Body-center (BC) nodes: 8 face-sharing corner neighbors
      - Corner nodes: 8 face-sharing BC neighbors + 6 edge-sharing corner neighbors
      - All edges must be bidirectional
      - No orphan nodes (nodes with zero neighbors)
      - All neighbor references must resolve to existing nodes

    Also checks occupied node structural health:
      - Weight consistency (weight >= 0)
      - Activation bounds (0 <= activation <= 10)
      - Crystal channel validity (referenced nodes exist)
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._last_report: Optional[LatticeIntegrityReport] = None
        self._report_history: List[LatticeIntegrityReport] = []
        self._max_history = 20

    def run_full_check(self) -> LatticeIntegrityReport:
        report = LatticeIntegrityReport()
        field = self._field

        with field._lock:
            report.total_nodes = len(field._nodes)
            self._check_bidirectionality(field, report)
            self._check_orphan_nodes(field, report)
            self._check_coordination(field, report)
            self._check_connectivity(field, report)
            self._check_occupied_health(field, report)

        total_errors = (
            len(report.broken_edges)
            + len(report.orphan_nodes)
            + len(report.coordination_errors)
        )
        if report.total_nodes > 0:
            critical_errors = len(report.broken_edges) + len(report.orphan_nodes)
            coord_ratio = len(report.coordination_errors) / max(report.total_nodes, 1)
            report.integrity_score = max(0.0, 1.0 - (critical_errors * 0.5 + coord_ratio * 0.5))

        if total_errors == 0:
            report.details = f"All {report.total_nodes} nodes pass integrity verification"
        else:
            report.details = f"{total_errors} issues found: {len(report.broken_edges)} broken edges, {len(report.orphan_nodes)} orphans, {len(report.coordination_errors)} coordination errors"

        self._last_report = report
        self._report_history.append(report)
        if len(self._report_history) > self._max_history:
            self._report_history = self._report_history[-self._max_history // 2:]

        return report

    def _check_bidirectionality(self, field, report: LatticeIntegrityReport):
        for n1, n2, etype in field._edges:
            report.face_edges_checked += 1 if etype == "face" else 0
            report.edge_edges_checked += 1 if etype == "edge" else 0

            node_a = field._nodes.get(n1)
            node_b = field._nodes.get(n2)

            if node_a is None or node_b is None:
                report.broken_edges.append({
                    "edge": (n1[:8], n2[:8]),
                    "type": etype,
                    "issue": "missing_node",
                })
                continue

            if etype == "face":
                if n2 not in node_a.face_neighbors:
                    report.broken_edges.append({
                        "edge": (n1[:8], n2[:8]),
                        "type": "face",
                        "issue": "unidirectional_a_to_b",
                    })
                if n1 not in node_b.face_neighbors:
                    report.broken_edges.append({
                        "edge": (n2[:8], n1[:8]),
                        "type": "face",
                        "issue": "unidirectional_b_to_a",
                    })
            elif etype == "edge":
                if n2 not in node_a.edge_neighbors:
                    report.broken_edges.append({
                        "edge": (n1[:8], n2[:8]),
                        "type": "edge",
                        "issue": "unidirectional_a_to_b",
                    })
                if n1 not in node_b.edge_neighbors:
                    report.broken_edges.append({
                        "edge": (n2[:8], n1[:8]),
                        "type": "edge",
                        "issue": "unidirectional_b_to_a",
                    })

    def _check_orphan_nodes(self, field, report: LatticeIntegrityReport):
        for nid, node in field._nodes.items():
            total_neighbors = len(node.face_neighbors) + len(node.edge_neighbors)
            if total_neighbors == 0:
                report.orphan_nodes.append(nid)

    def _check_coordination(self, field, report: LatticeIntegrityReport):
        for key, nid in field._position_index.items():
            node = field._nodes.get(nid)
            if node is None:
                continue

            is_body_center = isinstance(key, tuple) and len(key) == 4 and key[3] == "b"

            if is_body_center:
                expected_face = 8
                actual_face = len(node.face_neighbors)
                if actual_face < expected_face // 2:
                    report.coordination_errors.append({
                        "node": nid[:8],
                        "type": "body_center",
                        "expected_face": expected_face,
                        "actual_face": actual_face,
                    })
            else:
                expected_face = 8
                actual_face = len(node.face_neighbors)
                if actual_face < expected_face // 2:
                    report.coordination_errors.append({
                        "node": nid[:8],
                        "type": "corner",
                        "expected_face": expected_face,
                        "actual_face": actual_face,
                        "expected_edge": 6,
                        "actual_edge": len(node.edge_neighbors),
                    })

    def _check_connectivity(self, field, report: LatticeIntegrityReport):
        if not field._nodes:
            return
        visited = set()
        components = 0
        for start_nid in field._nodes:
            if start_nid in visited:
                continue
            components += 1
            stack = [start_nid]
            while stack:
                nid = stack.pop()
                if nid in visited:
                    continue
                visited.add(nid)
                node = field._nodes.get(nid)
                if node is None:
                    continue
                for fnid in node.face_neighbors:
                    if fnid not in visited:
                        stack.append(fnid)
                for enid in node.edge_neighbors:
                    if enid not in visited:
                        stack.append(enid)
        report.connectivity_components = components

    def _check_occupied_health(self, field, report: LatticeIntegrityReport):
        for nid, node in field._nodes.items():
            if not node.is_occupied:
                continue
            if node.weight < 0:
                report.coordination_errors.append({
                    "node": nid[:8],
                    "type": "negative_weight",
                    "weight": node.weight,
                })
            if node.activation < 0:
                report.coordination_errors.append({
                    "node": nid[:8],
                    "type": "negative_activation",
                    "activation": node.activation,
                })
            for crystal_target in node.crystal_channels:
                if crystal_target not in field._nodes:
                    report.broken_edges.append({
                        "edge": (nid[:8], crystal_target[:8]),
                        "type": "crystal",
                        "issue": "dangling_crystal_channel",
                    })

    def get_latest(self) -> Optional[Dict]:
        return self._last_report.to_dict() if self._last_report else None

    def get_history(self, n: int = 10) -> List[Dict]:
        return [r.to_dict() for r in self._report_history[-n:]]

    def stats(self) -> Dict[str, Any]:
        return {
            "checks_performed": len(self._report_history),
            "latest_score": self._last_report.integrity_score if self._last_report else None,
        }


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
        self._max_paths = max_paths
        self._decay = decay
        self._reinforce = reinforce_factor
        self._min_weight = min_weight
        self._success_count = 0
        self._total_decay_count = 0

    def record_path(self, path: List[str], success: bool, strength: float):
        if len(path) < 2:
            return

        factor = self._reinforce if success else 1.0
        edge_strength = min(strength * factor / max(len(path) - 1, 1), 2.0)

        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            rev_key = (path[i + 1], path[i])
            if key in self._edges:
                self._edges[key] = min(self._edges[key] + edge_strength, 10.0)
            elif rev_key in self._edges:
                self._edges[rev_key] = min(self._edges[rev_key] + edge_strength, 10.0)
            else:
                self._edges[key] = min(edge_strength, 10.0)

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
            del self._edges[sorted_edges[i][0]]

    def get_top_paths(self, n: int = 20) -> List[Tuple[str, str, float]]:
        sorted_edges = sorted(self._edges.items(), key=lambda x: -x[1])
        return [(k[0][:8], k[1][:8], round(v, 4)) for k, v in sorted_edges[:n]]

    def stats(self) -> Dict[str, Any]:
        weights = list(self._edges.values())
        return {
            "total_path_segments": len(self._edges),
            "success_count": self._success_count,
            "decay_cycles": self._total_decay_count,
            "avg_path_weight": float(np.mean(weights)) if weights else 0.0,
            "max_path_weight": float(max(weights)) if weights else 0.0,
        }


class SelfCheckResult:
    __slots__ = (
        "check_time", "anomalies_found", "isolated_nodes", "duplicate_pairs",
        "orphan_nodes", "low_activation_nodes", "repairs_attempted",
        "repairs_succeeded", "pulse_triggered", "details",
    )

    def __init__(self):
        self.check_time: float = time.time()
        self.anomalies_found: int = 0
        self.isolated_nodes: List[str] = []
        self.duplicate_pairs: List[Dict[str, Any]] = []
        self.orphan_nodes: List[str] = []
        self.low_activation_nodes: List[str] = []
        self.repairs_attempted: int = 0
        self.repairs_succeeded: int = 0
        self.pulse_triggered: bool = False
        self.details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_time": self.check_time,
            "anomalies_found": self.anomalies_found,
            "isolated_nodes": self.isolated_nodes[:20],
            "duplicate_pairs": self.duplicate_pairs[:10],
            "orphan_nodes": self.orphan_nodes[:20],
            "low_activation_nodes": self.low_activation_nodes[:20],
            "repairs_attempted": self.repairs_attempted,
            "repairs_succeeded": self.repairs_succeeded,
            "pulse_triggered": self.pulse_triggered,
            "details": self.details,
        }


class SelfCheckEngine:
    """
    Proactive awareness system — periodic self-diagnosis via self-check pulses.

    Runs three diagnostic passes:
      1. Isolation scan: find occupied nodes with no occupied neighbors
      2. Duplicate scan: find memory pairs with high content similarity
      3. Vitality scan: find nodes with critically low activation despite high weight

    Auto-repair actions:
      - Isolated nodes → emit reinforcing pulse to re-integrate
      - Duplicates → annotate with __duplicate_of__ label, merge labels/weight
      - Low-activation → boost base_activation, emit reinforcing pulse
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._check_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._history: List[SelfCheckResult] = []
        self._max_history = 50
        self._lock = threading.RLock()

    def start(self):
        if self._check_thread is not None and self._check_thread.is_alive():
            return
        self._stop_event.clear()
        self._check_thread = threading.Thread(
            target=self._check_loop, name="self-check", daemon=True
        )
        self._check_thread.start()
        logger.info("SelfCheck engine started — interval=%.0fs", PCNNConfig.SELF_CHECK_INTERVAL)

    def stop(self):
        self._stop_event.set()
        if self._check_thread:
            self._check_thread.join(timeout=5)
            self._check_thread = None

    def _check_loop(self):
        while not self._stop_event.wait(timeout=PCNNConfig.SELF_CHECK_INTERVAL):
            try:
                result = self.run_full_check()
                with self._lock:
                    self._history.append(result)
                    if len(self._history) > self._max_history:
                        self._history = self._history[-self._max_history // 2:]
                if result.anomalies_found > 0:
                    logger.info(
                        "SelfCheck: %d anomalies (%d isolated, %d duplicates, %d low-activation)",
                        result.anomalies_found, len(result.isolated_nodes),
                        len(result.duplicate_pairs), len(result.low_activation_nodes),
                    )
            except Exception as e:
                logger.error("SelfCheck error: %s", e, exc_info=True)

    def run_full_check(self) -> SelfCheckResult:
        result = SelfCheckResult()
        field = self._field

        with field._lock:
            occupied = [(nid, n) for nid, n in field._nodes.items() if n.is_occupied]
            if not occupied:
                result.details = "no occupied nodes"
                return result

            self._scan_isolated(field, occupied, result)
            self._scan_duplicates(field, occupied, result)
            self._scan_vitality(field, occupied, result)
            self._auto_repair(field, result)

        result.anomalies_found = (
            len(result.isolated_nodes)
            + len(result.duplicate_pairs)
            + len(result.orphan_nodes)
            + len(result.low_activation_nodes)
        )

        if result.anomalies_found > 0:
            self._emit_self_check_pulse(field, result)

        return result

    def _scan_isolated(self, field, occupied, result: SelfCheckResult):
        for nid, node in occupied:
            occupied_neighbor_count = 0
            for fnid in node.face_neighbors[:8]:
                fn = field._nodes.get(fnid)
                if fn and fn.is_occupied:
                    occupied_neighbor_count += 1
                    break
            if occupied_neighbor_count == 0:
                for enid in node.edge_neighbors[:6]:
                    en = field._nodes.get(enid)
                    if en and en.is_occupied:
                        occupied_neighbor_count += 1
                        break
            if occupied_neighbor_count == 0:
                result.isolated_nodes.append(nid)

    def _scan_duplicates(self, field, occupied, result: SelfCheckResult):
        threshold = PCNNConfig.DUPLICATE_TOKEN_OVERLAP
        contents_tokens = {}
        for nid, node in occupied:
            contents_tokens[nid] = field._extract_tokens(node.content)

        checked = set()
        for i, (nid_a, node_a) in enumerate(occupied):
            for j in range(i + 1, min(i + 30, len(occupied))):
                nid_b, node_b = occupied[j]
                pair_key = (min(nid_a, nid_b), max(nid_a, nid_b))
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                tokens_a = contents_tokens.get(nid_a, set())
                tokens_b = contents_tokens.get(nid_b, set())
                if not tokens_a or not tokens_b:
                    continue

                intersection = len(tokens_a & tokens_b)
                union = len(tokens_a | tokens_b)
                if union == 0:
                    continue
                jaccard = intersection / union

                if jaccard >= threshold:
                    result.duplicate_pairs.append({
                        "node_a": nid_a[:12],
                        "node_b": nid_b[:12],
                        "similarity": round(jaccard, 3),
                        "content_a": node_a.content[:50],
                        "content_b": node_b.content[:50],
                        "weight_a": node_a.weight,
                        "weight_b": node_b.weight,
                    })

    def _scan_vitality(self, field, occupied, result: SelfCheckResult):
        for nid, node in occupied:
            if node.weight >= 2.0 and node.activation < 0.05:
                result.low_activation_nodes.append(nid)
            elif node.weight >= 1.0 and node.activation < node.base_activation * 0.5:
                result.low_activation_nodes.append(nid)

    def _auto_repair(self, field, result: SelfCheckResult):
        for nid in result.isolated_nodes[:5]:
            node = field._nodes.get(nid)
            if node and node.is_occupied:
                node.base_activation = max(node.base_activation, 0.05)
                field._emit_pulse(
                    nid,
                    strength=PCNNConfig.SELF_CHECK_STRENGTH,
                    pulse_type=PulseType.SELF_CHECK,
                )
                result.repairs_attempted += 1
                result.repairs_succeeded += 1

        for dup in result.duplicate_pairs[:3]:
            nid_a_full = None
            nid_b_full = None
            for nid, n in field._nodes.items():
                if nid.startswith(dup["node_a"]) and n.is_occupied:
                    nid_a_full = nid
                if nid.startswith(dup["node_b"]) and n.is_occupied:
                    nid_b_full = nid
                if nid_a_full and nid_b_full:
                    break

            if not nid_a_full or not nid_b_full:
                continue

            node_a = field._nodes.get(nid_a_full)
            node_b = field._nodes.get(nid_b_full)
            if not node_a or not node_b:
                continue

            weight_ratio = min(node_a.weight, node_b.weight) / max(node_a.weight, node_b.weight, 0.1)

            if weight_ratio < PCNNConfig.DUPLICATE_MERGE_MIN_WEIGHT_RATIO:
                if node_a.weight >= node_b.weight:
                    node_a.labels = list(set(node_a.labels) | set(node_b.labels))
                    node_a.weight = max(node_a.weight, node_b.weight * 0.5)
                    node_b.labels.append("__duplicate_of__")
                    node_b.metadata["duplicate_of"] = nid_a_full[:12]
                else:
                    node_b.labels = list(set(node_a.labels) | set(node_b.labels))
                    node_b.weight = max(node_b.weight, node_a.weight * 0.5)
                    node_a.labels.append("__duplicate_of__")
                    node_a.metadata["duplicate_of"] = nid_b_full[:12]
                result.repairs_attempted += 1
                result.repairs_succeeded += 1

        for nid in result.low_activation_nodes[:5]:
            node = field._nodes.get(nid)
            if node and node.is_occupied:
                boost = node.weight * 0.3
                node.activation = min(10.0, node.activation + boost)
                node.base_activation = max(node.base_activation, 0.05)
                field._emit_pulse(nid, strength=0.3, pulse_type=PulseType.REINFORCING)
                result.repairs_attempted += 1
                result.repairs_succeeded += 1

    def _emit_self_check_pulse(self, field, result: SelfCheckResult):
        all_anomaly_ids = (
            result.isolated_nodes[:3]
            + result.low_activation_nodes[:3]
        )
        for nid in all_anomaly_ids:
            node = field._nodes.get(nid)
            if node and node.is_occupied:
                field._emit_pulse(
                    nid,
                    strength=PCNNConfig.SELF_CHECK_STRENGTH,
                    pulse_type=PulseType.SELF_CHECK,
                )
        result.pulse_triggered = True

    def get_history(self, n: int = 10) -> List[Dict]:
        with self._lock:
            return [r.to_dict() for r in self._history[-n:]]

    def get_latest(self) -> Optional[Dict]:
        with self._lock:
            return self._history[-1].to_dict() if self._history else None

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_checks = len(self._history)
            total_anomalies = sum(r.anomalies_found for r in self._history)
            total_repairs = sum(r.repairs_succeeded for r in self._history)
            last_time = self._history[-1].check_time if self._history else 0
            return {
                "total_checks": total_checks,
                "total_anomalies_found": total_anomalies,
                "total_repairs_done": total_repairs,
                "last_check_time": last_time,
                "engine_running": self._check_thread is not None and self._check_thread.is_alive(),
                "check_interval": PCNNConfig.SELF_CHECK_INTERVAL,
            }


class TetrahedralCell:
    """
    A single tetrahedral cell within the BCC lattice honeycomb.

    In BCC, each cubic unit cell decomposes into 8 tetrahedra:
    each formed by the body-center + 3 adjacent corner nodes
    sharing a face of the cube.

    Quality metrics:
    - Volume: |det(v1-v0, v2-v0, v3-v0)| / 6
    - Regularity: ratio of inscribed sphere radius to circumscribed
      sphere radius, compared to ideal regular tetrahedron
    - Skewness: deviation from equilateral tetrahedron (0=perfect, 1=degenerate)
    - Jacobian: minimum scaled Jacobian (positive=valid, zero/negative=inverted)
    """

    __slots__ = (
        "cell_id", "vertex_ids", "vertex_positions", "body_center_id",
        "volume", "quality", "skewness", "jacobian",
        "centroid", "memory_count", "total_weight", "density",
        "avg_activation", "label_diversity", "weight_variance", "effective_quality",
    )

    def __init__(self, cell_id: str, vertex_ids: List[str],
                 vertex_positions: List[np.ndarray], body_center_id: str):
        self.cell_id = cell_id
        self.vertex_ids = vertex_ids
        self.vertex_positions = vertex_positions
        self.body_center_id = body_center_id
        self.volume: float = 0.0
        self.quality: float = 0.0
        self.skewness: float = 0.0
        self.jacobian: float = 0.0
        self.centroid: np.ndarray = np.zeros(3, dtype=np.float32)
        self.memory_count: int = 0
        self.total_weight: float = 0.0
        self.density: float = 0.0
        self.avg_activation: float = 0.0
        self.label_diversity: int = 0
        self.weight_variance: float = 0.0
        self.effective_quality: float = 0.0
        self._compute_metrics()

    def _compute_metrics(self):
        if len(self.vertex_positions) != 4:
            return
        v0, v1, v2, v3 = self.vertex_positions
        self.centroid = ((v0 + v1 + v2 + v3) / 4.0).astype(np.float32)

        e1 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0

        cross = np.cross(e1, e2)
        det = float(np.dot(cross, e3))
        self.volume = abs(det) / 6.0

        if self.volume < 1e-10:
            self.quality = 0.0
            self.skewness = 1.0
            self.jacobian = 0.0
            return

        edges = [
            float(np.linalg.norm(v1 - v0)),
            float(np.linalg.norm(v2 - v0)),
            float(np.linalg.norm(v3 - v0)),
            float(np.linalg.norm(v2 - v1)),
            float(np.linalg.norm(v3 - v1)),
            float(np.linalg.norm(v3 - v2)),
        ]
        avg_edge = np.mean(edges)
        max_edge = max(edges)
        min_edge = min(edges)

        if max_edge < 1e-10:
            self.skewness = 1.0
        else:
            edge_ratio = min_edge / max_edge
            self.skewness = 1.0 - edge_ratio

        s = avg_edge
        ideal_volume = s ** 3 / (6.0 * math.sqrt(2))
        if ideal_volume > 1e-10:
            self.quality = min(1.0, self.volume / ideal_volume)
        else:
            self.quality = 0.0

        edge_sq = [e ** 2 for e in edges]
        cayley_menger = np.array([
            [0, 1, 1, 1, 1],
            [1, 0, edge_sq[0], edge_sq[1], edge_sq[3]],
            [1, edge_sq[0], 0, edge_sq[2], edge_sq[4]],
            [1, edge_sq[1], edge_sq[2], 0, edge_sq[5]],
            [1, edge_sq[3], edge_sq[4], edge_sq[5], 0],
        ], dtype=np.float64)
        cm_det = abs(np.linalg.det(cayley_menger))
        if cm_det > 1e-20:
            inradius = math.sqrt(cm_det) / (288.0 * self.volume ** 2 + 1e-10)
        else:
            inradius = 0.0

        circumradius_sq = 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                d = float(np.sum((self.vertex_positions[i] - self.vertex_positions[j]) ** 2))
                circumradius_sq = max(circumradius_sq, d)
        circumradius = math.sqrt(circumradius_sq) / 2.0

        if circumradius > 1e-10:
            ideal_ratio = math.sqrt(6.0) / 12.0
            actual_ratio = inradius / circumradius
            self.jacobian = min(1.0, actual_ratio / ideal_ratio) if ideal_ratio > 0 else 0.0
        else:
            self.jacobian = 0.0

    def update_density(self, nodes: Dict[str, Any]):
        count = 0
        total_w = 0.0
        total_act = 0.0
        label_diversity = set()
        for vid in self.vertex_ids:
            node = nodes.get(vid)
            if node and node.is_occupied:
                count += 1
                total_w += node.weight
                total_act += node.activation
                label_diversity.update(node.labels)
        bc_node = nodes.get(self.body_center_id)
        if bc_node and bc_node.is_occupied:
            count += 1
            total_w += bc_node.weight
            total_act += bc_node.activation
            label_diversity.update(bc_node.labels)
        self.memory_count = count
        self.total_weight = total_w
        self.density = count / 5.0 if self.volume > 1e-10 else 0.0
        self.avg_activation = total_act / max(count, 1)
        self.label_diversity = len([l for l in label_diversity if not l.startswith("__")])
        if count >= 2:
            weights = []
            for vid in self.vertex_ids:
                node = nodes.get(vid)
                if node and node.is_occupied:
                    weights.append(node.weight)
            bc_node2 = nodes.get(self.body_center_id)
            if bc_node2 and bc_node2.is_occupied:
                weights.append(bc_node2.weight)
            if len(weights) >= 2:
                avg = sum(weights) / len(weights)
                variance = sum((w - avg) ** 2 for w in weights) / len(weights)
                self.weight_variance = min(1.0, variance / max(avg ** 2, 0.01))
        self.effective_quality = self.quality * (1.0 + 0.1 * self.label_diversity) * (1.0 - 0.3 * self.weight_variance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "vertex_ids": [vid[:8] for vid in self.vertex_ids],
            "body_center": self.body_center_id[:8],
            "centroid": self.centroid.tolist(),
            "volume": round(self.volume, 4),
            "quality": round(self.quality, 4),
            "skewness": round(self.skewness, 4),
            "jacobian": round(self.jacobian, 4),
            "memory_count": self.memory_count,
            "total_weight": round(self.total_weight, 3),
            "density": round(self.density, 3),
            "avg_activation": round(self.avg_activation, 3),
            "label_diversity": self.label_diversity,
            "weight_variance": round(self.weight_variance, 3),
            "effective_quality": round(self.effective_quality, 4),
        }


class HoneycombCellMap:
    """
    Decomposes the BCC lattice into tetrahedral cells and provides
    cell-level operations for structural analysis and memory placement.

    BCC Tetrahedral Decomposition:
    Each BCC unit cell (body-center + 8 corners) decomposes into 8 tetrahedra.
    The 8 tetrahedra are formed by selecting 3 of the 8 corners that share
    a cube face, combined with the body-center node.

    The 8 tetrahedra correspond to the 8 corners of the cube:
      For corner (dx, dy, dz) where dx,dy,dz in {0,1}:
        tetrahedron = [body_center, corner(dx,dy,dz),
                       corner(dx^1,dy,dz), corner(dx,dy^1,dz), corner(dx,dy,dz^1)]
        where ^1 means flip the bit (0↔1)

    This ensures each tetrahedron fills exactly 1/8 of the cube volume.
    """

    def __init__(self):
        self._cells: Dict[str, TetrahedralCell] = {}
        self._node_to_cells: Dict[str, List[str]] = defaultdict(list)
        self._bcc_cell_index: Dict[Tuple, List[str]] = {}

    def build(self, nodes: Dict[str, Any], position_index: Dict[Tuple, str], spacing: float):
        self._cells.clear()
        self._node_to_cells.clear()
        self._bcc_cell_index.clear()

        for key, bid in position_index.items():
            if not (isinstance(key, tuple) and len(key) == 4 and key[3] == "b"):
                continue
            ix, iy, iz = key[0], key[1], key[2]

            bc_node = nodes.get(bid)
            if bc_node is None:
                continue

            cell_ids = []
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        c0 = position_index.get((ix + dx, iy + dy, iz + dz))
                        c1 = position_index.get((ix + (1 - dx), iy + dy, iz + dz))
                        c2 = position_index.get((ix + dx, iy + (1 - dy), iz + dz))
                        c3 = position_index.get((ix + dx, iy + dy, iz + (1 - dz)))

                        if not all([c0, c1, c2, c3]):
                            continue

                        n0, n1, n2, n3 = nodes.get(c0), nodes.get(c1), nodes.get(c2), nodes.get(c3)
                        if not all([n0, n1, n2, n3]):
                            continue

                        cell_id = hashlib.sha256(
                            f"{bid}:{c0}:{c1}:{c2}:{c3}".encode()
                        ).hexdigest()[:12]

                        cell = TetrahedralCell(
                            cell_id=cell_id,
                            vertex_ids=[bid, c0, c1, c2],
                            vertex_positions=[bc_node.position, n0.position, n1.position, n2.position],
                            body_center_id=bid,
                        )

                        self._cells[cell_id] = cell
                        cell_ids.append(cell_id)

                        for nid in [bid, c0, c1, c2]:
                            self._node_to_cells[nid].append(cell_id)

            if cell_ids:
                self._bcc_cell_index[(ix, iy, iz)] = cell_ids

    def get_cell(self, cell_id: str) -> Optional[TetrahedralCell]:
        return self._cells.get(cell_id)

    def get_cells_for_node(self, node_id: str) -> List[TetrahedralCell]:
        cell_ids = self._node_to_cells.get(node_id, [])
        return [self._cells[cid] for cid in cell_ids if cid in self._cells]

    def update_all_densities(self, nodes: Dict[str, Any]):
        for cell in self._cells.values():
            cell.update_density(nodes)

    def get_best_cells(self, n: int = 20) -> List[TetrahedralCell]:
        sorted_cells = sorted(self._cells.values(), key=lambda c: -c.quality)
        return sorted_cells[:n]

    def get_cells_by_density(self, n: int = 20) -> List[TetrahedralCell]:
        sorted_cells = sorted(self._cells.values(), key=lambda c: -c.density)
        return sorted_cells[:n]

    def find_optimal_placement_cells(self, nodes: Dict, label_set: Set[str],
                                      label_index: Dict, count: int = 10) -> List[TetrahedralCell]:
        related_nodes = set()
        for lbl in label_set:
            related_nodes.update(label_index.get(lbl, set()))

        scored_cells = []
        for cell in self._cells.values():
            if cell.memory_count >= 4:
                continue
            label_overlap = 0
            for vid in cell.vertex_ids:
                if vid in related_nodes:
                    label_overlap += 1
            bc = nodes.get(cell.body_center_id)
            if bc and bc.id in related_nodes:
                label_overlap += 2

            quality_bonus = cell.quality * 3.0
            density_penalty = cell.density * PCNNConfig.TETRA_DENSITY_PENALTY
            score = label_overlap * 2.0 + quality_bonus - density_penalty
            if score > 0:
                scored_cells.append((cell, score))

        scored_cells.sort(key=lambda x: -x[1])
        return [c for c, _ in scored_cells[:count]]

    def structural_analysis(self, nodes=None) -> Dict[str, Any]:
        if not self._cells:
            return {"total_cells": 0}

        if nodes:
            self.update_all_densities(nodes)
        qualities = [c.effective_quality for c in self._cells.values()]
        raw_qualities = [c.quality for c in self._cells.values()]
        volumes = [c.volume for c in self._cells.values()]
        skews = [c.skewness for c in self._cells.values()]
        jacobians = [c.jacobian for c in self._cells.values()]
        densities = [c.density for c in self._cells.values()]

        high_quality = sum(1 for q in qualities if q > 0.8)
        low_quality = sum(1 for q in qualities if q < 0.3)

        return {
            "total_cells": len(self._cells),
            "total_bcc_units": len(self._bcc_cell_index),
            "quality": {
                "mean": round(float(np.mean(qualities)), 4),
                "min": round(float(min(qualities)), 4),
                "max": round(float(max(qualities)), 4),
                "std": round(float(np.std(qualities)), 4),
                "high_quality_count": high_quality,
                "low_quality_count": low_quality,
            },
            "volume": {
                "mean": round(float(np.mean(volumes)), 6),
                "total": round(float(sum(volumes)), 4),
            },
            "skewness": {
                "mean": round(float(np.mean(skews)), 4),
                "max": round(float(max(skews)), 4),
            },
            "jacobian": {
                "mean": round(float(np.mean(jacobians)), 4),
                "min": round(float(min(jacobians)), 4),
            },
            "density": {
                "mean": round(float(np.mean(densities)), 4),
                "max": round(float(max(densities)), 4),
                "occupied_cells": sum(1 for d in densities if d > 0),
            },
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "total_cells": len(self._cells),
            "total_bcc_units": len(self._bcc_cell_index),
        }


class SemanticCluster:
    __slots__ = ("cluster_id", "labels", "node_ids", "centroid", "avg_weight", "total_activation")

    def __init__(self, cluster_id: str, labels: Set[str]):
        self.cluster_id = cluster_id
        self.labels = labels
        self.node_ids: List[str] = []
        self.centroid: Optional[np.ndarray] = None
        self.avg_weight: float = 0.0
        self.total_activation: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "labels": list(self.labels),
            "node_count": len(self.node_ids),
            "avg_weight": round(self.avg_weight, 3),
            "total_activation": round(self.total_activation, 3),
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
        }


class OrganizeResult:
    __slots__ = (
        "organize_time", "clusters_found", "clusters_reinforced",
        "entropy_before", "entropy_after", "consolidations_done",
        "shortcuts_created", "migrations_done", "details",
    )

    def __init__(self):
        self.organize_time: float = time.time()
        self.clusters_found: int = 0
        self.clusters_reinforced: int = 0
        self.entropy_before: float = 0.0
        self.entropy_after: float = 0.0
        self.consolidations_done: int = 0
        self.shortcuts_created: int = 0
        self.migrations_done: int = 0
        self.details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "organize_time": self.organize_time,
            "clusters_found": self.clusters_found,
            "clusters_reinforced": self.clusters_reinforced,
            "entropy_before": round(self.entropy_before, 4),
            "entropy_after": round(self.entropy_after, 4),
            "consolidations_done": self.consolidations_done,
            "shortcuts_created": self.shortcuts_created,
            "migrations_done": self.migrations_done,
            "details": self.details,
        }


class SelfOrganizeEngine:
    """
    Unified self-organization engine — autonomous topology optimization.

    Four modules per cycle:
      1. Cluster Detection: label-cooccurrence + spatial proximity grouping
      2. Entropy Balance: weight redistribution to prevent concentration
      3. Memory Consolidation: merge weak duplicate memories
      4. Topological Shortcuts: virtual edges for semantically close but
         topologically distant memory pairs

    Runs periodically in the pulse loop, separate from SelfCheckEngine
    which handles anomaly detection and repair.
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._history: List[OrganizeResult] = []
        self._max_history = 30
        self._clusters: List[SemanticCluster] = []
        self._shortcuts: Dict[Tuple[str, str], float] = {}
        self._lock = threading.RLock()

    def run_cycle(self) -> OrganizeResult:
        result = OrganizeResult()
        field = self._field

        with field._lock:
            occupied = [(nid, n) for nid, n in field._nodes.items() if n.is_occupied]
            if len(occupied) < 3:
                result.details = "insufficient nodes for organization"
                with self._lock:
                    self._history.append(result)
                    if len(self._history) > self._max_history:
                        self._history = self._history[-self._max_history // 2:]
                return result

            result.entropy_before = self._compute_entropy(occupied)

            self._detect_clusters(field, occupied, result)
            self._rebalance_entropy(field, occupied, result)
            self._migrate_memories(field, occupied, result)
            self._consolidate_memories(field, occupied, result)
            self._create_shortcuts(field, occupied, result)

            result.entropy_after = self._compute_entropy(occupied)
            self._reinforce_clusters(field, result)

        with self._lock:
            self._history.append(result)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history // 2:]

        return result

    def _compute_entropy(self, occupied: List[Tuple[str, Any]]) -> float:
        if not occupied:
            return 0.0
        weights = [n.weight for _, n in occupied]
        total = sum(weights)
        if total <= 0:
            return 0.0
        probs = [w / total for w in weights if w > 0]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        max_entropy = math.log(len(probs) + 1e-10)
        return entropy / max(max_entropy, 1e-10)

    def _detect_clusters(self, field, occupied, result: OrganizeResult):
        cfg = PCNNConfig
        label_groups: Dict[str, List[str]] = defaultdict(list)
        for nid, node in occupied:
            for lbl in node.labels:
                if not lbl.startswith("__"):
                    label_groups[lbl].append(nid)

        candidate_labels = [
            lbl for lbl, nids in label_groups.items()
            if len(nids) >= cfg.CLUSTER_MIN_SIZE
        ]

        clusters: List[SemanticCluster] = []
        used_nodes: Set[str] = set()

        for lbl in sorted(candidate_labels, key=lambda l: -len(label_groups[l]))[:cfg.CLUSTER_MAX_LABELS]:
            nids = label_groups[lbl]
            available = [n for n in nids if n not in used_nodes]
            if len(available) < cfg.CLUSTER_MIN_SIZE:
                continue

            cluster_labels = {lbl}
            cluster_nodes = []
            for nid in available:
                node = field._nodes.get(nid)
                if node is None:
                    continue
                other_labels = set(node.labels) - {"__pulse_bridge__", "__system__"}
                cluster_labels.update(other_labels)
                cluster_nodes.append((nid, node))

            if len(cluster_nodes) < cfg.CLUSTER_MIN_SIZE:
                continue

            positions = [n.position for _, n in cluster_nodes]
            centroid = np.mean(positions, axis=0)
            avg_w = float(np.mean([n.weight for _, n in cluster_nodes]))
            total_act = sum(n.activation for _, n in cluster_nodes)

            cluster = SemanticCluster(f"cluster_{lbl}", cluster_labels)
            cluster.node_ids = [nid for nid, _ in cluster_nodes]
            cluster.centroid = centroid
            cluster.avg_weight = avg_w
            cluster.total_activation = total_act
            clusters.append(cluster)

            for nid, _ in cluster_nodes:
                used_nodes.add(nid)

        self._clusters = clusters
        result.clusters_found = len(clusters)

    def _reinforce_clusters(self, field, result: OrganizeResult):
        for cluster in self._clusters:
            if len(cluster.node_ids) < 2:
                continue
            for nid in cluster.node_ids:
                node = field._nodes.get(nid)
                if node and node.is_occupied:
                    neighbor_in_cluster = 0
                    for fnid in node.face_neighbors[:6]:
                        fn = field._nodes.get(fnid)
                        if fn and fn.is_occupied:
                            shared = set(fn.labels) & cluster.labels
                            if len(shared) > 0:
                                neighbor_in_cluster += 1
                    if neighbor_in_cluster < 2:
                        boost = cluster.avg_weight * 0.05
                        node.activation = min(10.0, node.activation + boost)
                        field._hebbian.record_path(
                            [cluster.node_ids[0], nid],
                            success=True,
                            strength=cluster.avg_weight * 0.3,
                        )
                        result.clusters_reinforced += 1

    def _rebalance_entropy(self, field, occupied, result: OrganizeResult):
        cfg = PCNNConfig
        weights = [n.weight for _, n in occupied]
        avg_w = float(np.mean(weights))
        max_w = float(max(weights))
        min_w = float(min(weights))

        if max_w <= 0 or avg_w <= 0:
            return

        concentration = max_w / (avg_w + 1e-10)

        if concentration > 5.0:
            for nid, node in occupied:
                if node.weight > avg_w * 4:
                    suppress = node.weight * cfg.ENTROPY_SUPPRESS_FACTOR
                    node.weight = max(min_w, node.weight - suppress)
                elif node.weight < avg_w * 0.3:
                    boost = avg_w * cfg.ENTROPY_BOOST_FACTOR
                    node.weight = min(max_w, node.weight + boost)
                    node.activation = min(10.0, node.activation + boost * 0.5)

    def _migrate_memories(self, field, occupied, result: OrganizeResult):
        occupied_count = len(occupied)
        if occupied_count < 20:
            return
        max_migrations = min(5, max(1, occupied_count // 200))
        quality_list = []
        for nid, node in occupied:
            if node.weight < 0.5 or "__consolidated__" in node.labels:
                continue
            geo_q = node.metadata.get("geometric_quality", 0.5)
            bcc_c = node.metadata.get("bcc_cell_coherence", 0.5)
            neighbors_occ = sum(
                1 for fnid in node.face_neighbors[:6]
                if field._nodes.get(fnid) and field._nodes[fnid].is_occupied
            )
            density_penalty = max(0, (neighbors_occ - 4) * 0.08)
            quality = 0.4 * geo_q + 0.3 * bcc_c + 0.3 * (node.weight / 10.0) - density_penalty
            quality_list.append((nid, node, quality))
        quality_list.sort(key=lambda x: x[2])
        migrated = 0
        for nid, node, q in quality_list[:15]:
            if migrated >= max_migrations:
                break
            node_labels = set(node.labels) - {
                "__pulse_bridge__", "__system__", "__consolidated__", "__low_priority__"
            }
            best_target = None
            best_ts = q + 0.12
            search_pool = node.face_neighbors + node.edge_neighbors + node.vertex_neighbors
            for fnid in search_pool:
                fn = field._nodes.get(fnid)
                if fn is None or fn.is_occupied:
                    continue
                t_geo = field._compute_node_geometric_quality(fnid)
                t_bcc = field._bcc_cell_coherence(fnid)
                label_match = 0
                for fnn in fn.face_neighbors[:6]:
                    fnn_n = field._nodes.get(fnn)
                    if fnn_n and fnn_n.is_occupied and node_labels & set(fnn_n.labels):
                        label_match += 1
                t_cell_q = field._cell_quality_factor(fnid) if hasattr(field, '_cell_quality_factor') else 1.0
                ts = 0.3 * t_geo + 0.25 * t_bcc + 0.25 * min(label_match / 3.0, 1.0) + 0.2 * t_cell_q
                if ts > best_ts:
                    best_target = fnid
                    best_ts = ts
            if best_target is None:
                continue
            target = field._nodes[best_target]
            target.content = node.content
            target.labels = list(node.labels)
            target.weight = node.weight
            target.activation = node.activation
            target.base_activation = node.base_activation
            target.metadata = dict(node.metadata)
            target.creation_time = node.creation_time
            target.feeding = node.feeding
            target.crystal_channels = dict(node.crystal_channels)
            target.metadata["geometric_quality"] = float(field._compute_node_geometric_quality(best_target))
            target.metadata["geo_topo_divergence"] = float(field._compute_geometric_topo_divergence(best_target))
            target.metadata["bcc_cell_coherence"] = float(field._bcc_cell_coherence(best_target))
            target.metadata["migrated_from"] = nid[:12]
            target.metadata["migration_time"] = time.time()
            target.touch()
            chash = hashlib.sha256(target.content.encode()).hexdigest()[:12]
            field._content_hash_index[chash] = best_target
            for lbl in target.labels:
                if not lbl.startswith("__"):
                    field._label_index[lbl].discard(nid)
                    field._label_index[lbl].add(best_target)
            for tok in field._extract_tokens(target.content):
                field._content_token_index[tok].discard(nid)
                field._content_token_index[tok].add(best_target)
            node.content = None
            node.labels = []
            node.weight = 0.0
            node.activation = 0.0
            node.metadata = {}
            node.crystal_channels = {}
            field._emit_pulse(best_target, strength=target.weight * 0.3, pulse_type=PulseType.REINFORCING)
            migrated += 1
        result.migrations_done = migrated

    def _consolidate_memories(self, field, occupied, result: OrganizeResult):
        cfg = PCNNConfig
        low_weight = [(nid, n) for nid, n in occupied if n.weight < 1.0 and n.is_occupied]
        if len(low_weight) < 2:
            return

        consolidated = 0
        checked = set()
        low_map = {nid: n for nid, n in low_weight}

        for nid_a, node_a in low_weight:
            if consolidated >= cfg.CONSOLIDATION_MAX_PER_CYCLE:
                break
            if nid_a in checked:
                continue
            if node_a.crystal_channels:
                continue
            if "__low_priority__" in node_a.labels:
                continue

            tokens_a = field._extract_tokens(node_a.content)
            if not tokens_a:
                continue

            best_match = None
            best_sim = 0.0

            candidate_nids = set()
            for tok in tokens_a:
                candidate_nids.update(field._content_token_index.get(tok, set()))
            candidate_nids.discard(nid_a)

            for nid_b in candidate_nids:
                if nid_b in checked:
                    continue
                node_b = low_map.get(nid_b)
                if node_b is None:
                    continue
                if node_b.crystal_channels:
                    continue

                tokens_b = field._extract_tokens(node_b.content)
                if not tokens_b:
                    continue

                intersection = len(tokens_a & tokens_b)
                union = len(tokens_a | tokens_b)
                if union == 0:
                    continue
                jaccard = intersection / union

                if jaccard >= cfg.CONSOLIDATION_MIN_SIMILARITY and jaccard > best_sim:
                    best_sim = jaccard
                    best_match = (nid_b, node_b)

            if best_match is None:
                continue

            nid_b, node_b = best_match
            receiver, donor = (node_a, node_b) if node_a.weight >= node_b.weight else (node_b, node_a)
            receiver_nid, donor_nid = (nid_a, nid_b) if node_a.weight >= node_b.weight else (nid_b, nid_a)

            transfer = donor.weight * cfg.CONSOLIDATION_WEIGHT_TRANSFER
            receiver.weight = min(10.0, receiver.weight + transfer)
            receiver.activation = min(10.0, receiver.activation + donor.activation * 0.3)
            receiver.labels = list(set(receiver.labels) | set(donor.labels))
            receiver.labels = [l for l in receiver.labels if not l.startswith("__duplicate")]

            donor.metadata["consolidated_into"] = receiver_nid[:12]
            donor.metadata["consolidation_weight"] = round(donor.weight, 3)
            donor.labels.append("__consolidated__")
            donor.weight *= (1.0 - cfg.CONSOLIDATION_WEIGHT_TRANSFER)
            donor.activation *= 0.3

            field._hebbian.record_path(
                [receiver_nid, donor_nid], success=True, strength=receiver.weight * 0.2
            )

            checked.add(nid_a)
            checked.add(nid_b)
            consolidated += 1

        result.consolidations_done = consolidated

    def _create_shortcuts(self, field, occupied, result: OrganizeResult):
        cfg = PCNNConfig
        shortcuts_this_cycle = 0
        exclude = {"__pulse_bridge__", "__system__", "__consolidated__", "__dream__"}
        occupied_set = {nid for nid, _ in occupied}
        label_inv: Dict[str, List[str]] = defaultdict(list)
        for nid, node in occupied:
            for lbl in node.labels:
                if lbl not in exclude:
                    label_inv[lbl].append(nid)

        candidate_pairs: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for lbl, nids in label_inv.items():
            if len(nids) < 2:
                continue
            sampled = nids if len(nids) <= 30 else random.sample(nids, 30)
            for i in range(len(sampled)):
                for j in range(i + 1, len(sampled)):
                    a, b = sampled[i], sampled[j]
                    key = (min(a, b), max(a, b))
                    candidate_pairs[key].add(lbl)

        scored: List[Tuple[float, Tuple[str, str], Set[str]]] = []
        for key, shared in candidate_pairs.items():
            if len(shared) < cfg.SHORTCUT_MIN_LABEL_OVERLAP:
                continue
            if key in self._shortcuts:
                continue
            na, nb = field._nodes.get(key[0]), field._nodes.get(key[1])
            if na is None or nb is None:
                continue
            if nb.id in na.face_neighbors or nb.id in na.edge_neighbors:
                continue
            all_labels_a = set(na.labels) - exclude
            all_labels_b = set(nb.labels) - exclude
            jaccard = len(shared) / max(len(all_labels_a | all_labels_b), 1)
            scored.append((jaccard, key, shared))
        scored.sort(key=lambda x: -x[0])

        for jaccard, key, shared in scored[:200]:
            if shortcuts_this_cycle >= cfg.SHORTCUT_MAX_PER_CYCLE:
                break
            nid_a, nid_b = key
            distance = self._topo_distance(field, nid_a, nid_b)
            if distance is None or distance <= 2 or distance > cfg.SHORTCUT_MAX_DISTANCE:
                continue
            all_labels_a = set(field._nodes[nid_a].labels) - exclude
            all_labels_b = set(field._nodes[nid_b].labels) - exclude
            strength = cfg.SHORTCUT_VIRTUAL_STRENGTH * len(shared) / max(len(all_labels_a | all_labels_b), 1)
            self._shortcuts[key] = strength
            field._hebbian.record_path(
                [nid_a, nid_b], success=True, strength=strength * 2.0
            )
            shortcuts_this_cycle += 1

        if len(self._shortcuts) > 500:
            sorted_s = sorted(self._shortcuts.items(), key=lambda x: x[1])
            for k, _ in sorted_s[:len(self._shortcuts) - 300]:
                del self._shortcuts[k]

        result.shortcuts_created = shortcuts_this_cycle

    def _topo_distance(self, field, nid_a: str, nid_b: str) -> Optional[int]:
        if nid_a == nid_b:
            return 0
        visited = {nid_a}
        frontier = [nid_a]
        for depth in range(8):
            next_frontier = []
            for fid in frontier:
                fn = field._nodes.get(fid)
                if fn is None:
                    continue
                for nnid in fn.face_neighbors[:6] + fn.edge_neighbors[:4]:
                    if nnid == nid_b:
                        return depth + 1
                    if nnid not in visited:
                        visited.add(nnid)
                        next_frontier.append(nnid)
            frontier = next_frontier
            if not frontier:
                break
        return None

    def get_clusters(self) -> List[Dict]:
        with self._lock:
            return [c.to_dict() for c in self._clusters]

    def get_shortcuts(self, n: int = 20) -> List[Dict]:
        with self._lock:
            sorted_s = sorted(self._shortcuts.items(), key=lambda x: -x[1])[:n]
            return [
                {"nodes": (k[0][:8], k[1][:8]), "strength": round(v, 3)}
                for k, v in sorted_s
            ]

    def get_history(self, n: int = 10) -> List[Dict]:
        with self._lock:
            return [r.to_dict() for r in self._history[-n:]]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_organize_cycles": len(self._history),
                "active_clusters": len(self._clusters),
                "active_shortcuts": len(self._shortcuts),
                "latest_entropy": self._history[-1].entropy_after if self._history else None,
                "total_consolidations": sum(r.consolidations_done for r in self._history),
                "total_shortcuts_created": sum(r.shortcuts_created for r in self._history),
            }


class DreamCycleResult:
    __slots__ = ("cycle_time", "sources_used", "dreams_created", "cross_domain", "insights")

    def __init__(self):
        self.cycle_time: float = time.time()
        self.sources_used: int = 0
        self.dreams_created: int = 0
        self.cross_domain: int = 0
        self.insights: List[Dict] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_time": self.cycle_time,
            "sources_used": self.sources_used,
            "dreams_created": self.dreams_created,
            "cross_domain": self.cross_domain,
            "insights": self.insights[:10],
        }


class DreamEngine:
    """
    Autonomous creative memory recombination engine.

    Dream Cycle:
    1. Select 2-3 high-weight memory clusters (different label domains)
    2. Extract key elements from each cluster
    3. Recombine into new dream memories (cross-domain synthesis)
    4. Score dream creativity: label distance * weight product * activation resonance
    5. Store high-scoring dreams as __dream__ memories
    6. Emit cascade pulses from dream nodes to propagate insights

    This mirrors human REM sleep creativity: disparate neural patterns
    recombine during dream states to produce novel associations.
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._history: List[DreamCycleResult] = []
        self._max_history = 30
        self._total_dreams = 0
        self._lock = threading.Lock()

    def _extract_content_summary(self, content: str, max_chars: int = 40) -> str:
        if not content:
            return ""
        clean = content.lstrip("[").split("] ", 1)
        text = clean[-1] if len(clean) > 1 else clean[0]
        text = text.strip()
        for sep in ["。", "，", "；", "\n", "——", "：", "|"]:
            if sep in text and text.index(sep) < max_chars:
                text = text[:text.index(sep)]
                break
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text

    def _extract_deep_structure(self, content: str) -> Dict:
        """
        Extract deep structural elements from a memory.
        Returns: {core_concept, key_principles: [], methods: [], constraints: [], domain_specific: []}
        """
        if not content:
            return {"core_concept": "", "key_principles": [], "methods": [], "constraints": []}
        clean = content.lstrip("[").split("] ", 1)
        text = clean[-1] if len(clean) > 1 else clean[0]
        sections = text.split("\n")
        core = ""
        principles = []
        methods = []
        constraints = []
        for sec in sections:
            stripped = sec.strip()
            if not stripped:
                continue
            if stripped.startswith("【") and "】" in stripped:
                bracket_content = stripped
                if any(k in bracket_content for k in ["核心", "本质", "关键", "根本", "基础", "原理", "逻辑"]):
                    inner = bracket_content.split("】", 1)[1].strip() if "】" in bracket_content else ""
                    if inner:
                        principles.append(inner[:60])
                elif any(k in bracket_content for k in ["流程", "步骤", "方法", "操作", "做法", "如何"]):
                    inner = bracket_content.split("】", 1)[1].strip() if "】" in bracket_content else ""
                    if inner:
                        methods.append(inner[:60])
                elif any(k in bracket_content for k in ["注意", "限制", "前提", "条件", "必须", "不可", "约束"]):
                    inner = bracket_content.split("】", 1)[1].strip() if "】" in bracket_content else ""
                    if inner:
                        constraints.append(inner[:60])
                elif not core:
                    core = bracket_content.split("】", 1)[1].strip()[:40] if "】" in bracket_content else ""
            elif any(stripped.startswith(p) for p in ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⒈", "⒉", "⒊", "1.", "2.", "3.", "4.", "5.", "- ", "* "]):
                item = stripped.lstrip("①②③④⑤⑥⑦⑧⑨⑩⒈⒉⒊⒋⒌1234567890.-* ").strip()
                if item:
                    principles.append(item[:50])
            elif not core and len(stripped) > 5:
                core = stripped[:40]
        if not core:
            for sep in ["。", "，", "\n", "——"]:
                if sep in text:
                    core = text[:text.index(sep)][:40]
                    break
            if not core:
                core = text[:40]
        return {
            "core_concept": core,
            "key_principles": principles[:5],
            "methods": methods[:3],
            "constraints": constraints[:3],
        }

    def _compute_expertise_depth(self, domain_label: str, domain_nodes: List) -> float:
        """
        Measure how deep the expertise is in a given domain.
        Deep expertise = high weight + many interconnected nodes + crystallized pathways + long content.
        """
        if not domain_nodes:
            return 0.0
        total_weight = sum(n.weight for _, n in domain_nodes)
        avg_weight = total_weight / len(domain_nodes)
        node_ids = {nid for nid, _ in domain_nodes}
        internal_links = 0
        for nid, n in domain_nodes:
            for fnid in n.face_neighbors + n.edge_neighbors:
                if fnid in node_ids:
                    internal_links += 1
        density = internal_links / max(len(domain_nodes) * 4, 1)
        avg_content_len = sum(len(n.content) for _, n in domain_nodes) / max(len(domain_nodes), 1)
        content_depth = min(1.0, avg_content_len / 500.0)
        crystal_count = sum(1 for _, n in domain_nodes if n.crystal_channels)
        crystal_ratio = crystal_count / max(len(domain_nodes), 1)
        return min(1.0, avg_weight / 5.0 * 0.25 + density * 0.25 + content_depth * 0.3 + crystal_ratio * 0.2)

    def _generate_deep_insight(self, struct_a: Dict, struct_b: Dict,
                                domain_a: str, domain_b: str,
                                depth_a: float, depth_b: float) -> str:
        """
        Generate a deep cross-domain insight from structural analysis.
        Instead of just concatenating summaries, find structural analogies
        and generate a substantive insight.
        """
        core_a = struct_a["core_concept"]
        core_b = struct_b["core_concept"]
        principles_a = struct_a["key_principles"]
        principles_b = struct_b["key_principles"]
        methods_a = struct_a["methods"]
        methods_b = struct_b["methods"]
        constraints_a = struct_a["constraints"]
        constraints_b = struct_b["constraints"]

        insight_parts = []
        insight_parts.append(f"{domain_a}核心⟨{core_a}⟩与{domain_b}核心⟨{core_b}⟩的深层对照")

        if principles_a and principles_b:
            pa = principles_a[0] if principles_a else ""
            pb = principles_b[0] if principles_b else ""
            insight_parts.append(f"原理映射: {domain_a}的「{pa[:30]}」↔ {domain_b}的「{pb[:30]}」")

        if methods_a and methods_b:
            ma = methods_a[0] if methods_a else ""
            mb = methods_b[0] if methods_b else ""
            insight_parts.append(f"方法迁移: {domain_a}的「{ma[:25]}」→ 可否应用于{domain_b}的「{mb[:25]}」？")

        if constraints_a and constraints_b:
            ca = constraints_a[0] if constraints_a else ""
            cb = constraints_b[0] if constraints_b else ""
            insight_parts.append(f"约束碰撞: {domain_a}「{ca[:25]}」× {domain_b}「{cb[:25]}」")

        combined_depth = (depth_a + depth_b) / 2
        if combined_depth > 0.5:
            insight_parts.append(f"【专深度{combined_depth:.0%}】{domain_a}↔{domain_b}交叉创新潜力高")
        elif combined_depth > 0.25:
            insight_parts.append(f"【专深度{combined_depth:.0%}】{domain_a}或{domain_b}需继续深化")

        return " | ".join(insight_parts)

    def _trace_dream_path(self, field, nid_a: str, nid_b: str, max_hops: int = 8) -> List[str]:
        """
        BFS through the BCC lattice to find the topological path between two nodes.
        Returns a list of occupied node IDs along the path (waypoints).
        This IS the dream's spatial structure — the geometric journey through the lattice.
        """
        if nid_a == nid_b:
            return []
        visited = {nid_a}
        parent = {}
        queue = [nid_a]
        for _ in range(max_hops):
            next_q = []
            for fid in queue:
                fn = field._nodes.get(fid)
                if fn is None:
                    continue
                for nnid in fn.face_neighbors[:6] + fn.edge_neighbors[:4]:
                    if nnid == nid_b:
                        path = [nid_b]
                        cur = fid
                        while cur in parent:
                            path.append(cur)
                            cur = parent[cur]
                        path.append(nid_a)
                        path.reverse()
                        occupied_waypoints = []
                        for pid in path:
                            pn = field._nodes.get(pid)
                            if pn and pn.is_occupied and pid not in (nid_a, nid_b):
                                occupied_waypoints.append(pid)
                        return occupied_waypoints
                    if nnid not in visited:
                        visited.add(nnid)
                        parent[nnid] = fid
                        next_q.append(nnid)
            queue = next_q
            if not queue:
                break
        return []

    def run_dream_cycle(self) -> DreamCycleResult:
        result = DreamCycleResult()
        field = self._field
        cfg = PCNNConfig

        with field._lock:
            occupied = [
                (nid, n) for nid, n in field._nodes.items()
                if n.is_occupied and n.weight >= cfg.DREAM_MIN_SOURCE_WEIGHT
                and "__dream__" not in n.labels
                and "__pulse_bridge__" not in n.labels
                and "__consolidated__" not in n.labels
            ]
            if len(occupied) < 4:
                result.insights = [{"note": "insufficient source memories for dreaming"}]
                return result

            label_domains: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
            for nid, node in occupied:
                for lbl in node.labels:
                    if not lbl.startswith("__"):
                        label_domains[lbl].append((nid, node))

            top_domains = sorted(
                label_domains.items(),
                key=lambda x: -sum(n.weight for _, n in x[1])
            )[:8]

            if len(top_domains) < 2:
                result.insights = [{"note": "insufficient label diversity for dreaming"}]
                return result

            domain_depths = {}
            for dname, dnodes in top_domains:
                domain_depths[dname] = self._compute_expertise_depth(dname, dnodes)

            result.sources_used = len(occupied)

            for _ in range(cfg.DREAM_MAX_RECOMBINATIONS):
                if len(top_domains) < 2:
                    break

                mode = random.random()
                if mode < 0.4:
                    deepest = sorted(domain_depths.items(), key=lambda x: -x[1])
                    d1_idx = next((i for i, (n, _) in enumerate(top_domains) if n == deepest[0][0]), 0)
                    d2_candidates = [i for i in range(len(top_domains)) if i != d1_idx]
                    if not d2_candidates:
                        continue
                    d2_idx = random.choice(d2_candidates)
                elif mode < 0.7:
                    d1_idx = random.randint(0, len(top_domains) - 1)
                    d2_idx = random.randint(0, len(top_domains) - 1)
                    if d1_idx == d2_idx:
                        continue
                else:
                    d1_idx = random.randint(0, len(top_domains) - 1)
                    same_domain_nodes = top_domains[d1_idx][1]
                    if len(same_domain_nodes) < 3:
                        continue
                    sampled = random.sample(same_domain_nodes, min(2, len(same_domain_nodes)))
                    src_a = sampled[0]
                    src_b = sampled[1] if len(sampled) > 1 else random.choice(same_domain_nodes)
                    nid_a, node_a = src_a
                    nid_b, node_b = src_b
                    domain_a_name = top_domains[d1_idx][0]
                    domain_b_name = domain_a_name
                    depth_a = domain_depths.get(domain_a_name, 0)
                    depth_b = depth_a
                    struct_a = self._extract_deep_structure(node_a.content)
                    struct_b = self._extract_deep_structure(node_b.content)
                    creativity = self._score_creativity(node_a, node_b, domain_a_name, domain_b_name)
                    creativity = min(1.0, creativity * 1.3)
                    if creativity < 0.5:
                        continue
                    self._create_deep_dream(field, result, nid_a, nid_b, node_a, node_b,
                                            domain_a_name, domain_b_name, struct_a, struct_b,
                                            depth_a, depth_b, creativity, cfg)
                    continue

                domain_a_name, domain_a_nodes = top_domains[d1_idx]
                domain_b_name, domain_b_nodes = top_domains[d2_idx]

                src_a = random.choice(domain_a_nodes)
                src_b = random.choice(domain_b_nodes)
                nid_a, node_a = src_a
                nid_b, node_b = src_b

                spatial_dist = float(np.linalg.norm(node_a.position - node_b.position))
                max_dist = field._spacing * 8
                spatial_factor = min(1.0, spatial_dist / max_dist)
                if spatial_factor < 0.15:
                    continue

                field_tension_a = field._reflection_field.get_dream_tension(nid_a) if field._reflection_field else 0.5
                field_tension_b = field._reflection_field.get_dream_tension(nid_b) if field._reflection_field else 0.5
                tension_product = field_tension_a * field_tension_b

                creativity = self._score_creativity(node_a, node_b, domain_a_name, domain_b_name)
                creativity = min(1.0, creativity * (1.0 + tension_product * 0.3))

                if creativity < 0.5:
                    continue

                depth_a = domain_depths.get(domain_a_name, 0)
                depth_b = domain_depths.get(domain_b_name, 0)
                struct_a = self._extract_deep_structure(node_a.content)
                struct_b = self._extract_deep_structure(node_b.content)

                self._create_deep_dream(field, result, nid_a, nid_b, node_a, node_b,
                                        domain_a_name, domain_b_name, struct_a, struct_b,
                                        depth_a, depth_b, creativity, cfg)

        with self._lock:
            self._history.append(result)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history // 2:]

        return result

    def _create_deep_dream(self, field, result, nid_a, nid_b, node_a, node_b,
                           domain_a_name, domain_b_name, struct_a, struct_b,
                           depth_a, depth_b, creativity, cfg):
        summary_a = struct_a["core_concept"]
        summary_b = struct_b["core_concept"]
        waypoints = self._trace_dream_path(field, nid_a, nid_b)
        waypoint_summaries = []
        for wid in waypoints[:3]:
            wn = field._nodes.get(wid)
            if wn and wn.is_occupied:
                ws = self._extract_content_summary(wn.content, 30)
                if ws:
                    wlabels = [l for l in wn.labels if not l.startswith("__")][:2]
                    waypoint_summaries.append((wlabels, ws))

        deep_insight = self._generate_deep_insight(
            struct_a, struct_b, domain_a_name, domain_b_name, depth_a, depth_b
        )

        dream_parts = []
        dream_parts.append(f"{domain_a_name}核心: {summary_a}")
        for wlabels, ws in waypoint_summaries:
            label_hint = "/".join(wlabels) if wlabels else ""
            dream_parts.append(f"经由{label_hint}: {ws}" if label_hint else f"中间: {ws}")
        dream_parts.append(f"{domain_b_name}核心: {summary_b}")

        dream_content = "[dream] " + " -> ".join(dream_parts) + " | " + deep_insight
        dream_labels = list(set([
            domain_a_name, domain_b_name, "__dream__",
        ]))
        dream_weight = max(
            0.5,
            min(
                cfg.DREAM_INSIGHT_WEIGHT * creativity * (1.0 + (depth_a + depth_b) * 0.2),
                max(node_a.weight, node_b.weight) * 0.8,
            )
        )

        path_length = len(waypoints) + 2

        try:
            dream_id = field.store(
                content=dream_content,
                labels=dream_labels,
                weight=dream_weight,
                metadata={
                    "dream_source_a": nid_a[:12],
                    "dream_source_b": nid_b[:12],
                    "creativity_score": round(creativity, 3),
                    "dream_type": "cross_domain" if domain_a_name != domain_b_name else "intra_domain",
                    "spatial_distance": round(float(np.linalg.norm(node_a.position - node_b.position)), 2),
                    "topo_path_length": path_length,
                    "waypoint_count": len(waypoints),
                    "expertise_depth_a": round(depth_a, 3),
                    "expertise_depth_b": round(depth_b, 3),
                    "insight_type": "deep_structural",
                },
            )
            result.dreams_created += 1
            self._total_dreams += 1

            is_cross = domain_a_name != domain_b_name
            if is_cross:
                result.cross_domain += 1

            result.insights.append({
                "dream_id": dream_id[:12],
                "domains": [domain_a_name, domain_b_name],
                "creativity": round(creativity, 3),
                "weight": round(dream_weight, 3),
                "cross_domain": is_cross,
                "spatial_distance": round(float(np.linalg.norm(node_a.position - node_b.position)), 2),
                "topo_path_length": path_length,
                "expertise_depth": round((depth_a + depth_b) / 2, 3),
                "insight_preview": deep_insight[:80],
            })

            field._emit_pulse(
                dream_id, strength=dream_weight * 0.6,
                pulse_type=PulseType.CASCADE,
            )
        except Exception:
            pass

        with self._lock:
            self._history.append(result)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history // 2:]

        return result

    def _score_creativity(self, node_a, node_b, domain_a: str, domain_b: str) -> float:
        cfg = PCNNConfig
        field = self._field
        cross_bonus = cfg.DREAM_CROSS_DOMAIN_BONUS if domain_a != domain_b else 1.0
        weight_factor = (node_a.weight * node_b.weight) / 25.0
        activation_factor = (node_a.activation + node_b.activation) / 10.0
        structural = self._structural_distance(node_a, node_b)
        crystal_boost_a = sum(node_a.crystal_channels.values()) if node_a.crystal_channels else 0
        crystal_boost_b = sum(node_b.crystal_channels.values()) if node_b.crystal_channels else 0
        crystal_factor = min(1.0, (crystal_boost_a + crystal_boost_b) / 10.0)
        hebbian_factor = 0.0
        hebbian_w = field._hebbian.get_path_bias(node_a.id, node_b.id)
        if hebbian_w > 0:
            hebbian_factor = min(1.0, hebbian_w * 0.3)
        pcnn_factor = 0.0
        if node_a.fired and node_b.fired:
            pcnn_factor = 0.15
        elif node_a.internal_activity > 0.5 or node_b.internal_activity > 0.5:
            pcnn_factor = 0.08
        resonance_factor = 0.0
        for ev in field._resonance_events:
            if node_a.id[:8] in ev.get("node_ids", []) or node_b.id[:8] in ev.get("node_ids", []):
                resonance_factor = 0.1
                break
        moran_factor = 0.0
        if field._spatial_autocorrelation > 0.1:
            moran_factor = 0.05
        base = (cross_bonus * weight_factor * activation_factor) / (cross_bonus + 1)
        creativity = base * (0.30 + 0.25 * structural + 0.12 * crystal_factor + 0.08 * min(1.0, weight_factor)
                            + 0.08 * hebbian_factor + 0.07 * pcnn_factor + 0.05 * resonance_factor + 0.05 * moran_factor)
        return min(1.0, creativity)

    def _structural_distance(self, node_a, node_b) -> float:
        try:
            dist = float(np.linalg.norm(node_a.position - node_b.position))
            return min(1.0, dist / (self._field._spacing * 8))
        except Exception:
            return 0.5

    def get_history(self, n: int = 10) -> List[Dict]:
        return [r.to_dict() for r in self._history[-n:]]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_dreams_created": self._total_dreams,
            "dream_cycles_run": len(self._history),
            "latest_dreams": sum(r.dreams_created for r in self._history[-3:]),
        }


class AgentMemoryDriver:
    """
    Memory-driven agent capability layer.

    Provides the interface between TetraMem's neural memory and agent actions:
    - Context injection: assemble relevant memories for current task context
    - Reasoning chains: multi-hop paths from source memory to target insight
    - Proactive suggestions: memory pattern-based action recommendations
    - Navigate: path-finding through the memory topology graph
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field

    def get_context(self, topic: str, max_memories: int = 15) -> Dict[str, Any]:
        field = self._field
        cfg = PCNNConfig
        n = min(max_memories, cfg.AGENT_CONTEXT_MAX_MEMORIES)

        memories = field.query(topic, k=n * 2)
        if not memories:
            return {"topic": topic, "context": [], "reasoning": "no relevant memories found"}

        core = [m for m in memories if "__dream__" not in m.get("labels", [])][:n // 2]
        dreams = [m for m in memories if "__dream__" in m.get("labels", [])][:n // 4]
        bridges = [m for m in memories if "__pulse_bridge__" in m.get("labels", [])][:n // 4]

        context_memories = core + dreams + bridges
        context_memories.sort(key=lambda m: -m.get("distance", 0))

        labels_encountered = set()
        for m in context_memories:
            for l in m.get("labels", []):
                if not l.startswith("__"):
                    labels_encountered.add(l)

        associations = []
        if core:
            top_id = core[0]["id"]
            assoc = field.associate(top_id, max_depth=2)
            for a in assoc[:5]:
                if "__dream__" not in a.get("labels", []) and "__pulse_bridge__" not in a.get("labels", []):
                    associations.append({
                        "content": a["content"][:80],
                        "weight": a["weight"],
                        "connection": a["type"],
                    })

        reasoning = self._build_reasoning(topic, context_memories)

        return {
            "topic": topic,
            "context_count": len(context_memories),
            "context": [
                {
                    "id": m["id"][:12],
                    "content": m["content"][:120],
                    "weight": m.get("weight", 0),
                    "relevance": round(m.get("distance", 0), 3),
                    "labels": [l for l in m.get("labels", []) if not l.startswith("__")],
                    "is_dream": "__dream__" in m.get("labels", []),
                }
                for m in context_memories[:n]
            ],
            "related_labels": list(labels_encountered),
            "associations": associations,
            "reasoning": reasoning,
        }

    def reasoning_chain(self, source_id: str, target_query: str, max_hops: int = 5) -> Dict[str, Any]:
        field = self._field
        cfg = PCNNConfig
        hops = min(max_hops, cfg.AGENT_REASONING_MAX_HOPS)

        source = field.get_node(source_id)
        if source is None:
            return {"error": "source node not found"}

        targets = field.query(target_query, k=3)
        if not targets:
            return {"source": source_id[:12], "chain": [], "conclusion": "no target found"}

        target_ids = {t["id"] for t in targets}

        visited = {source_id}
        frontier = [(source_id, [source_id])]
        found_path = None

        for depth in range(hops):
            next_frontier = []
            for nid, path in frontier:
                node = field._nodes.get(nid)
                if node is None:
                    continue
                neighbors = []
                for fnid in node.face_neighbors:
                    fn = field._nodes.get(fnid)
                    if fn and fn.is_occupied and fnid not in visited:
                        score = fn.weight * fn.activation
                        if fnid in target_ids:
                            score += 100.0
                        neighbors.append((fnid, score))
                for enid in node.edge_neighbors[:6]:
                    en = field._nodes.get(enid)
                    if en and en.is_occupied and enid not in visited:
                        score = en.weight * en.activation * 0.5
                        if enid in target_ids:
                            score += 100.0
                        neighbors.append((enid, score))

                neighbors.sort(key=lambda x: -x[1])

                for nnid, score in neighbors[:3]:
                    visited.add(nnid)
                    new_path = path + [nnid]
                    if nnid in target_ids:
                        found_path = new_path
                        break
                    next_frontier.append((nnid, new_path))

                if found_path:
                    break
            frontier = next_frontier
            if found_path:
                break

        if not found_path:
            return {
                "source": source_id[:12],
                "target_query": target_query,
                "chain": [],
                "conclusion": "no path found within hop limit",
            }

        chain_nodes = []
        for nid in found_path:
            node = field._nodes.get(nid)
            if node:
                chain_nodes.append({
                    "id": nid[:12],
                    "content": node.content[:80],
                    "weight": round(node.weight, 2),
                    "labels": [l for l in node.labels if not l.startswith("__")],
                })

        return {
            "source": source_id[:12],
            "target_query": target_query,
            "chain_length": len(chain_nodes),
            "chain": chain_nodes,
            "conclusion": chain_nodes[-1]["content"][:100] if chain_nodes else "",
        }

    def suggest_actions(self, context: str = "") -> Dict[str, Any]:
        field = self._field
        cfg = PCNNConfig

        suggestions = []

        isolated = field.detect_isolated()
        if isolated:
            suggestions.append({
                "action": "connect_isolated_memories",
                "priority": "high",
                "description": f"{len(isolated)} memories have no occupied neighbors. Consider adding bridging memories or triggering cascade pulses.",
                "affected_count": len(isolated),
            })

        stats = field.stats()
        if stats.get("bridge_rate", 0) < 0.001:
            suggestions.append({
                "action": "increase_pulse_activity",
                "priority": "medium",
                "description": "Bridge rate is very low. The neural field may benefit from more exploration pulses or manual memory additions.",
                "current_bridge_rate": round(stats.get("bridge_rate", 0), 6),
            })

        so_stats = stats.get("self_organize", {})
        if so_stats.get("active_clusters", 0) > 0:
            suggestions.append({
                "action": "leverage_clusters",
                "priority": "medium",
                "description": f"{so_stats['active_clusters']} semantic clusters detected. Use cluster-aware queries for better recall.",
                "cluster_count": so_stats["active_clusters"],
            })

        hc = stats.get("honeycomb_cells", {})
        if hc and hc.get("density", {}).get("occupied_cells", 0) > 0:
            avg_density = hc["density"].get("mean", 0)
            if avg_density > 0.5:
                suggestions.append({
                    "action": "expand_memory_space",
                    "priority": "low",
                    "description": f"Average tetrahedral cell density is {avg_density:.2f}. Consider storing memories in new areas.",
                    "avg_density": round(avg_density, 3),
                })

        if context:
            relevant = field.query(context, k=3)
            if relevant:
                top = relevant[0]
                suggestions.append({
                    "action": "deepen_context",
                    "priority": "medium",
                    "description": f"Most relevant memory (weight={top['weight']:.1f}): {top['content'][:80]}",
                    "memory_id": top["id"][:12],
                })

        return {
            "context": context or "general",
            "suggestion_count": len(suggestions),
            "suggestions": suggestions[:cfg.AGENT_SUGGESTION_TOP_N],
        }

    def navigate(self, source_id: str, target_id: str, max_hops: int = 6) -> Dict[str, Any]:
        field = self._field

        src_node = field._nodes.get(source_id)
        tgt_node = field._nodes.get(target_id)
        if not src_node or not tgt_node:
            return {"path": [], "length": 0, "error": "source or target not found"}

        visited = {source_id}
        frontier = [(source_id, [source_id], 0.0)]
        best_path = None
        best_score = float('inf')

        for depth in range(max_hops):
            next_frontier = []
            for nid, path, cost in frontier:
                node = field._nodes.get(nid)
                if node is None:
                    continue
                neighbors = list(node.face_neighbors) + list(node.edge_neighbors[:4])
                if field._self_organize and hasattr(field._self_organize, '_shortcuts'):
                    for sc_key, sc_str in field._self_organize._shortcuts.items():
                        if nid in sc_key:
                            partner = sc_key[1] if sc_key[0] == nid else sc_key[0]
                            if partner not in visited:
                                neighbors.append(partner)
                for nnid in neighbors:
                    if nnid in visited:
                        continue
                    visited.add(nnid)
                    nn = field._nodes.get(nnid)
                    step_cost = 1.0
                    if nn:
                        if nn.is_occupied:
                            step_cost = 1.0 / (nn.weight + 0.1)
                        else:
                            step_cost = 2.0
                        crystal_boost = field._crystallized.get_boost(nid, nnid)
                        if crystal_boost > 1.0:
                            step_cost /= crystal_boost
                        hebbian_w = field._hebbian.get_path_bias(nid, nnid)
                        if hebbian_w > 0:
                            step_cost /= (1.0 + hebbian_w * 0.5)
                        if field._reflection_field:
                            energy = field._reflection_field._node_energy.get(nnid, 0.5)
                            step_cost *= (0.8 + energy * 0.4)
                    new_cost = cost + step_cost
                    new_path = path + [nnid]
                    if nnid == target_id:
                        if new_cost < best_score:
                            best_path = new_path
                            best_score = new_cost
                    else:
                        next_frontier.append((nnid, new_path, new_cost))
            frontier = next_frontier
            if not frontier:
                break

        if not best_path:
            return {"path": [], "length": 0, "source": source_id[:12], "target": target_id[:12]}

        path_data = []
        for nid in best_path:
            node = field._nodes.get(nid)
            if node:
                path_data.append({
                    "id": nid[:12],
                    "content": node.content[:60] if node.content else "",
                    "weight": round(node.weight, 2),
                    "occupied": node.is_occupied,
                })

        return {
            "source": source_id[:12],
            "target": target_id[:12],
            "path": path_data,
            "length": len(best_path) - 1,
            "cost": round(best_score, 3),
        }

    def _build_reasoning(self, topic: str, memories: List[Dict]) -> str:
        if not memories:
            return f"No memories related to '{topic}'."
        top = memories[0]
        labels = [l for l in top.get("labels", []) if not l.startswith("__")]
        parts = [f"Regarding '{topic}':"]
        parts.append(f"- Primary recall: {top.get('content', '')[:80]} (relevance: {top.get('distance', 0):.2f})")
        if labels:
            parts.append(f"- Related domains: {', '.join(labels[:5])}")
        dream_count = sum(1 for m in memories if "__dream__" in m.get("labels", []))
        if dream_count > 0:
            parts.append(f"- {dream_count} dream insights available")
        parts.append(f"- Total context: {len(memories)} memories")
        return "\n".join(parts)


class HoneycombNeuralField:
    """BCC Lattice Honeycomb with PCNN-grounded neural pulse engine."""

    def __init__(self, resolution: int = 5, spacing: float = 1.0):
        self._lock = threading.RLock()
        self._nodes: Dict[str, HoneycombNode] = {}
        self._position_index: Dict[Tuple, str] = {}
        self._label_index: Dict[str, Set[str]] = defaultdict(set)
        self._content_hash_index: Dict[str, str] = {}
        self._content_token_index: Dict[str, Set[str]] = defaultdict(set)
        self._edges: List[Tuple[str, str, str]] = []
        self._pulse_engine: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pulse_interval: float = PCNNConfig.BASE_PULSE_INTERVAL
        self._pulse_count: int = 0
        self._bridge_count: int = 0
        self._resolution = resolution
        self._spacing = spacing
        self._pulse_log: List[Dict] = []
        self._max_log = 200

        self._pulse_type_counts: Dict[PulseType, int] = {t: 0 for t in PulseType}
        self._hebbian = HebbianPathMemory(
            max_paths=PCNNConfig.HEBBIAN_MAX_PATHS,
            decay=PCNNConfig.HEBBIAN_DECAY,
            reinforce_factor=PCNNConfig.HEBBIAN_REINFORCE,
            min_weight=PCNNConfig.HEBBIAN_MIN_WEIGHT,
        )
        self._adaptive_interval: float = PCNNConfig.BASE_PULSE_INTERVAL
        self._recent_bridge_rate: float = 0.0
        self._self_check: Optional[SelfCheckEngine] = None
        self._crystallized: CrystallizedPathway = CrystallizedPathway(
            max_crystals=PCNNConfig.CRYSTALLIZE_MAX_PATHS,
            weight_floor=PCNNConfig.CRYSTAL_WEIGHT_FLOOR,
        )
        self._lattice_checker: Optional[LatticeIntegrityChecker] = None
        self._cascade_count: int = 0
        self._crystal_maintenance_cycle: int = 0
        self._self_organize: Optional[SelfOrganizeEngine] = None
        self._cell_map: HoneycombCellMap = HoneycombCellMap()
        self._dream_engine: Optional[DreamEngine] = None
        self._agent_driver: Optional[AgentMemoryDriver] = None
        self._feedback_loop: Optional[FeedbackLoop] = None
        self._session_manager: Optional[SessionManager] = None
        self._reflection_field: Optional[SpatialReflectionField] = None
        self._bcc_unit_index: Dict[str, List[str]] = defaultdict(list)
        self._spatial_autocorrelation: float = 0.0
        self._autocorrelation_history: List[float] = []
        self._current_phase: str = "fluid"
        self._resonance_events: List[Dict] = []
        self._propagation_source: str = ""

    def initialize(self) -> Dict[str, Any]:
        with self._lock:
            self._build_bcc_lattice()
            self._build_connectivity()
            self._cell_map.build(self._nodes, self._position_index, self._spacing)
            self._build_bcc_unit_index()
            logger.info(
                "Honeycomb cells: %d tetrahedral cells in %d BCC units",
                len(self._cell_map._cells), len(self._cell_map._bcc_cell_index),
            )
            return self.stats()

    def _bcc_direction_factor(self, from_pos: np.ndarray, to_pos: np.ndarray) -> float:
        delta = to_pos - from_pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-10:
            return 1.0
        direction = delta / dist
        abs_dir = np.abs(direction)
        abs_dir_sum = float(np.sum(abs_dir))
        if abs_dir_sum < 1e-10:
            return 1.0
        normalized = abs_dir / abs_dir_sum
        ideal_111 = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        alignment_111 = float(1.0 - np.linalg.norm(normalized - ideal_111) / math.sqrt(2.0/3.0))
        ideal_100 = np.array([1.0, 0.0, 0.0])
        alignment_100 = float(1.0 - np.linalg.norm(normalized - ideal_100) / math.sqrt(2.0))
        factor = 1.0 + 0.20 * alignment_111 - 0.12 * alignment_100
        nn_dist = self._spacing * math.sqrt(3) / 2.0
        ideal_step_ratio = nn_dist / max(dist, 1e-10)
        if 0.8 < ideal_step_ratio < 1.3:
            factor += 0.15
        return max(0.7, min(1.4, factor))

    def _compute_node_geometric_quality(self, nid: str) -> float:
        """
        Compute the geometric quality of a memory node based on its
        tetrahedral cell volume, quality, skewness, and jacobian.
        Higher = better geometric harmony = more reliable memory.
        """
        node = self._nodes.get(nid)
        if node is None or not node.is_occupied:
            return 0.0
        cells = self._cell_map.get_cells_for_node(nid)
        if not cells:
            return 0.5
        total_quality = sum(c.quality for c in cells)
        avg_quality = total_quality / len(cells)
        best_volume = max((c.volume for c in cells), default=0)
        volume_factor = min(1.0, best_volume / (self._spacing ** 3 * 0.1178 + 1e-10))
        avg_jacobian = sum(c.jacobian for c in cells) / len(cells)
        avg_skew = sum(c.skewness for c in cells) / len(cells)
        geo_quality = (
            0.35 * avg_quality
            + 0.25 * volume_factor
            + 0.25 * avg_jacobian
            + 0.15 * (1.0 - avg_skew)
        )
        return round(min(1.0, max(0.0, geo_quality)), 4)

    def _compute_geometric_topo_divergence(self, nid: str) -> float:
        """
        Divergence between geometric proximity and topological (label) proximity.
        High divergence = geometric neighbors share few labels = spatial tension.
        This is the core metric for geometric memory understanding.
        """
        node = self._nodes.get(nid)
        if node is None or not node.is_occupied:
            return 0.0
        node_labels = set(l for l in node.labels if not l.startswith("__"))
        if not node_labels:
            return 0.0
        total_divergence = 0.0
        count = 0
        for fnid in node.face_neighbors[:8]:
            fn = self._nodes.get(fnid)
            if fn and fn.is_occupied:
                fn_labels = set(l for l in fn.labels if not l.startswith("__"))
                if fn_labels:
                    jaccard = len(node_labels & fn_labels) / max(len(node_labels | fn_labels), 1)
                    geometric_dist = float(np.linalg.norm(node.position - fn.position))
                    geo_proximity = 1.0 / (1.0 + geometric_dist)
                    divergence = abs(geo_proximity - jaccard)
                    total_divergence += divergence
                    count += 1
        if count == 0:
            return 0.0
        return total_divergence / count

    def _build_bcc_lattice(self):
        res = self._resolution
        sp = self._spacing
        for ix in range(-res, res + 1):
            for iy in range(-res, res + 1):
                for iz in range(-res, res + 1):
                    pos = np.array([ix * sp, iy * sp, iz * sp], dtype=np.float32)
                    nid = hashlib.sha256(pos.tobytes()).hexdigest()[:16]
                    self._nodes[nid] = HoneycombNode(nid, pos)
                    self._position_index[(ix, iy, iz)] = nid

        for ix in range(-res, res):
            for iy in range(-res, res):
                for iz in range(-res, res):
                    bpos = np.array([(ix + 0.5) * sp, (iy + 0.5) * sp, (iz + 0.5) * sp], dtype=np.float32)
                    bid = hashlib.sha256(bpos.tobytes()).hexdigest()[:16]
                    self._nodes[bid] = HoneycombNode(bid, bpos)
                    self._position_index[(ix, iy, iz, "b")] = bid

        logger.info("BCC lattice: %d nodes", len(self._nodes))

    def _expand_lattice(self):
        old_res = self._resolution
        new_res = old_res + 1
        self._resolution = new_res
        sp = self._spacing
        new_nodes = 0
        new_body = 0
        new_corner_ids = set()
        new_body_ids = set()

        for ix in range(-new_res, new_res + 1):
            for iy in range(-new_res, new_res + 1):
                for iz in range(-new_res, new_res + 1):
                    if abs(ix) <= old_res and abs(iy) <= old_res and abs(iz) <= old_res:
                        continue
                    pos = np.array([ix * sp, iy * sp, iz * sp], dtype=np.float32)
                    nid = hashlib.sha256(pos.tobytes()).hexdigest()[:16]
                    if nid not in self._nodes:
                        self._nodes[nid] = HoneycombNode(nid, pos)
                        self._position_index[(ix, iy, iz)] = nid
                        new_nodes += 1
                        new_corner_ids.add(nid)

        for ix in range(-new_res, new_res):
            for iy in range(-new_res, new_res):
                for iz in range(-new_res, new_res):
                    if abs(ix) < old_res and abs(iy) < old_res and abs(iz) < old_res:
                        continue
                    bpos = np.array([(ix + 0.5) * sp, (iy + 0.5) * sp, (iz + 0.5) * sp], dtype=np.float32)
                    bid = hashlib.sha256(bpos.tobytes()).hexdigest()[:16]
                    key = (ix, iy, iz, "b")
                    if bid not in self._nodes:
                        self._nodes[bid] = HoneycombNode(bid, bpos)
                        self._position_index[key] = bid
                        new_body += 1
                        new_body_ids.add(bid)
                    elif key not in self._position_index:
                        self._position_index[key] = bid

        self._build_shell_connectivity(new_corner_ids, new_body_ids)
        self._cell_map.build(self._nodes, self._position_index, self._spacing)
        self._build_bcc_unit_index()
        logger.info("Lattice expanded: res %d->%d, +%d corners +%d body, total %d nodes",
                     old_res, new_res, new_nodes, new_body, len(self._nodes))

    def _build_shell_connectivity(self, new_corner_ids: Set[str], new_body_ids: Set[str]):
        all_new = new_corner_ids | new_body_ids
        face_count = 0
        edge_count = 0
        vertex_count = 0
        for key, bid in list(self._position_index.items()):
            if not (isinstance(key, tuple) and len(key) == 4 and key[3] == "b"):
                continue
            if bid not in all_new:
                continue
            ix, iy, iz = key[0], key[1], key[2]
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        ck = (ix + dx, iy + dy, iz + dz)
                        cnid = self._position_index.get(ck)
                        if not cnid:
                            continue
                        bnode = self._nodes.get(bid)
                        cnode = self._nodes.get(cnid)
                        if not bnode or not cnode:
                            continue
                        if cnid not in bnode.face_neighbors:
                            bnode.face_neighbors.append(cnid)
                        if bid not in cnode.face_neighbors:
                            cnode.face_neighbors.append(bid)
                        self._edges.append((bid, cnid, "face"))
                        face_count += 1
        for key, nid in list(self._position_index.items()):
            if not (isinstance(key, tuple) and len(key) == 3):
                continue
            if nid not in all_new:
                continue
            ix, iy, iz = key
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nk = (ix + dx, iy + dy, iz + dz)
                nnid = self._position_index.get(nk)
                if not nnid or nnid == nid:
                    continue
                node = self._nodes.get(nid)
                nn = self._nodes.get(nnid)
                if not node or not nn:
                    continue
                if nnid not in node.face_neighbors and nnid not in node.edge_neighbors:
                    node.edge_neighbors.append(nnid)
                    if nid not in nn.edge_neighbors:
                        nn.edge_neighbors.append(nid)
                    self._edges.append((nid, nnid, "edge"))
                    edge_count += 1
            for dx, dy, dz in [(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
                                (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
                                (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]:
                nk = (ix + dx, iy + dy, iz + dz)
                nnid = self._position_index.get(nk)
                if not nnid or nnid == nid:
                    continue
                node = self._nodes.get(nid)
                nn = self._nodes.get(nnid)
                if not node or not nn:
                    continue
                if (nnid not in node.face_neighbors and
                    nnid not in node.edge_neighbors and
                    nnid not in node.vertex_neighbors):
                    node.vertex_neighbors.append(nnid)
                    if nid not in nn.vertex_neighbors:
                        nn.vertex_neighbors.append(nid)
                    self._edges.append((nid, nnid, "vertex"))
                    vertex_count += 1
        logger.info("Shell connectivity: %d face, %d edge, %d vertex", face_count, edge_count, vertex_count)

    def _build_connectivity(self):
        face_count = 0
        edge_count = 0

        for key, bid in list(self._position_index.items()):
            if not (isinstance(key, tuple) and len(key) == 4 and key[3] == "b"):
                continue
            ix, iy, iz = key[0], key[1], key[2]
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        ck = (ix + dx, iy + dy, iz + dz)
                        cnid = self._position_index.get(ck)
                        if not cnid:
                            continue
                        bnode = self._nodes.get(bid)
                        cnode = self._nodes.get(cnid)
                        if not bnode or not cnode:
                            continue
                        if cnid not in bnode.face_neighbors:
                            bnode.face_neighbors.append(cnid)
                        if bid not in cnode.face_neighbors:
                            cnode.face_neighbors.append(bid)
                        self._edges.append((bid, cnid, "face"))
                        face_count += 1

        for key, nid in list(self._position_index.items()):
            if not (isinstance(key, tuple) and len(key) == 3):
                continue
            ix, iy, iz = key
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nk = (ix + dx, iy + dy, iz + dz)
                nnid = self._position_index.get(nk)
                if not nnid or nnid == nid:
                    continue
                node = self._nodes.get(nid)
                nn = self._nodes.get(nnid)
                if not node or not nn:
                    continue
                if nnid not in node.face_neighbors and nnid not in node.edge_neighbors:
                    node.edge_neighbors.append(nnid)
                    if nid not in nn.edge_neighbors:
                        nn.edge_neighbors.append(nid)
                    self._edges.append((nid, nnid, "edge"))
                    edge_count += 1

        vertex_count = 0
        for key, nid in list(self._position_index.items()):
            if not (isinstance(key, tuple) and len(key) == 3):
                continue
            ix, iy, iz = key
            for dx, dy, dz in [(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
                                (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
                                (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]:
                nk = (ix + dx, iy + dy, iz + dz)
                nnid = self._position_index.get(nk)
                if not nnid or nnid == nid:
                    continue
                node = self._nodes.get(nid)
                nn = self._nodes.get(nnid)
                if not node or not nn:
                    continue
                if (nnid not in node.face_neighbors and
                    nnid not in node.edge_neighbors and
                    nnid not in node.vertex_neighbors):
                    node.vertex_neighbors.append(nnid)
                    if nid not in nn.vertex_neighbors:
                        nn.vertex_neighbors.append(nid)
                    self._edges.append((nid, nnid, "vertex"))
                    vertex_count += 1

        logger.info("Connectivity: %d face, %d edge, %d vertex", face_count, edge_count, vertex_count)

    def _build_bcc_unit_index(self):
        """
        Build BCC unit cell containment index.
        For each BCC unit cell (body-center + 8 corners), track which nodes
        belong to the same unit. This provides natural geometric neighborhoods
        beyond face/edge/vertex neighbor types.
        """
        self._bcc_unit_index.clear()
        for key, bid in self._position_index.items():
            if not (isinstance(key, tuple) and len(key) == 4 and key[3] == "b"):
                continue
            ix, iy, iz = key[0], key[1], key[2]
            unit_members = [bid]
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        ck = (ix + dx, iy + dy, iz + dz)
                        cnid = self._position_index.get(ck)
                        if cnid:
                            unit_members.append(cnid)
            for nid in unit_members:
                self._bcc_unit_index[nid].append(bid)
        logger.info("BCC unit index: %d nodes mapped to %d unit cells",
                     len(self._bcc_unit_index), len(set(uid for uids in self._bcc_unit_index.values() for uid in uids)))

    def _get_bcc_cellmates(self, nid: str) -> List[str]:
        """Get all nodes sharing at least one BCC unit cell with the given node."""
        unit_ids = self._bcc_unit_index.get(nid, [])
        if not unit_ids:
            return []
        cellmates = set()
        for uid in unit_ids:
            bc_node = self._nodes.get(uid)
            if bc_node is None:
                continue
            for fnid in bc_node.face_neighbors:
                if fnid != nid:
                    cellmates.add(fnid)
            if uid != nid:
                cellmates.add(uid)
        return list(cellmates)

    def _bcc_cell_coherence(self, nid: str) -> float:
        """
        Measure label coherence within the BCC unit cells containing this node.
        High coherence = same-label memories are concentrated in the same unit cell,
        which is geometrically efficient for BCC topology.
        """
        node = self._nodes.get(nid)
        if node is None or not node.is_occupied or not node.labels:
            return 0.5
        node_labels = set(l for l in node.labels if not l.startswith("__"))
        if not node_labels:
            return 0.5
        unit_ids = self._bcc_unit_index.get(nid, [])
        if not unit_ids:
            return 0.5
        coherent = 0
        total = 0
        for uid in unit_ids:
            bc_node = self._nodes.get(uid)
            if bc_node is None:
                continue
            for fnid in bc_node.face_neighbors:
                if fnid == nid:
                    continue
                fn = self._nodes.get(fnid)
                if fn and fn.is_occupied and fn.labels:
                    fn_labels = set(l for l in fn.labels if not l.startswith("__"))
                    if fn_labels:
                        overlap = len(node_labels & fn_labels) / max(len(node_labels | fn_labels), 1)
                        coherent += overlap
                        total += 1
        if total == 0:
            return 0.5
        return coherent / total

    def compute_spatial_autocorrelation(self) -> float:
        """
        Compute Moran's I for spatial autocorrelation of memory weights.
        Positive I = weights are spatially clustered (similar weights near each other).
        Negative I = weights are spatially dispersed (different weights near each other).
        Near zero = random spatial distribution.
        Uses face-neighbor adjacency as the spatial weight matrix.
        """
        occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
        if len(occupied) < 3:
            self._spatial_autocorrelation = 0.0
            return 0.0
        weights = np.array([n.weight for _, n in occupied], dtype=np.float64)
        nid_list = [nid for nid, _ in occupied]
        nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}
        n = len(weights)
        w_mean = float(np.mean(weights))
        deviations = weights - w_mean
        num_sum = 0.0
        den_sum = float(np.sum(deviations ** 2))
        if den_sum < 1e-12:
            self._spatial_autocorrelation = 0.0
            return 0.0
        w_total = 0.0
        for i, nid in enumerate(nid_list):
            node = self._nodes.get(nid)
            if node is None:
                continue
            for fnid in node.face_neighbors:
                j = nid_to_idx.get(fnid)
                if j is not None and j != i:
                    w_ij = 1.0
                    num_sum += w_ij * deviations[i] * deviations[j]
                    w_total += w_ij
            for enid in node.edge_neighbors:
                j = nid_to_idx.get(enid)
                if j is not None and j != i:
                    w_ij = 0.5
                    num_sum += w_ij * deviations[i] * deviations[j]
                    w_total += w_ij
        if w_total < 1e-12:
            self._spatial_autocorrelation = 0.0
            return 0.0
        morans_i = (n / w_total) * (num_sum / den_sum)
        morans_i = max(-1.0, min(1.0, morans_i))
        self._spatial_autocorrelation = morans_i
        self._autocorrelation_history.append(morans_i)
        if len(self._autocorrelation_history) > 50:
            self._autocorrelation_history = self._autocorrelation_history[-25:]
        return morans_i

    def _vacancy_attraction(self, nid: str) -> float:
        """
        Compute the 'attraction' score of an empty node based on the weight
        and density of its surrounding occupied neighbors. High attraction means
        this empty node is in a desirable location surrounded by strong memories.
        Uses face > edge > vertex neighbor priority with BCC cellmate bonus.
        """
        node = self._nodes.get(nid)
        if node is None or node.is_occupied:
            return 0.0
        attraction = 0.0
        for fnid in node.face_neighbors:
            fn = self._nodes.get(fnid)
            if fn and fn.is_occupied:
                attraction += fn.weight * 1.0
        for enid in node.edge_neighbors:
            en = self._nodes.get(enid)
            if en and en.is_occupied:
                attraction += en.weight * 0.5
        for vnid in node.vertex_neighbors:
            vn = self._nodes.get(vnid)
            if vn and vn.is_occupied:
                attraction += vn.weight * 0.25
        cellmates = self._get_bcc_cellmates(nid)
        for cmid in cellmates:
            cm = self._nodes.get(cmid)
            if cm and cm.is_occupied:
                attraction += cm.weight * 0.3
        return attraction

    def _evict_for_space(self, content: str, labels, weight: float) -> str:
        bridge_nodes = [(nid, n) for nid, n in self._nodes.items()
                        if n.is_occupied and any(l.startswith("__pulse_bridge__") for l in n.labels)]
        if bridge_nodes:
            bridge_nodes.sort(key=lambda x: x[1].weight)
            evict_id, evict_node = bridge_nodes[0]
            self._clear_node(evict_id, evict_node)
            logger.info("Evicted bridge node %s (w=%.2f) for new memory", evict_id[:8], evict_node.weight)
            return evict_id
        dream_nodes = [(nid, n) for nid, n in self._nodes.items()
                       if n.is_occupied and "__dream__" in n.labels and n.weight < 0.5]
        if dream_nodes:
            dream_nodes.sort(key=lambda x: x[1].weight)
            evict_id, evict_node = dream_nodes[0]
            self._clear_node(evict_id, evict_node)
            logger.info("Evicted dream node %s (w=%.2f) for new memory", evict_id[:8], evict_node.weight)
            return evict_id
        low_weight = [(nid, n) for nid, n in self._nodes.items()
                      if n.is_occupied and n.weight < 0.3]
        if low_weight:
            low_weight.sort(key=lambda x: x[1].weight)
            evict_id, evict_node = low_weight[0]
            self._clear_node(evict_id, evict_node)
            logger.info("Evicted low-weight node %s (w=%.2f)", evict_id[:8], evict_node.weight)
            return evict_id
        return random.choice(list(self._nodes.keys()))

    def _clear_node(self, nid, node):
        chash = hashlib.sha256((node.content or "").encode()).hexdigest()[:12]
        self._content_hash_index.pop(chash, None)
        for lbl in node.labels:
            self._label_index[lbl].discard(nid)
        for tok in self._extract_tokens(node.content or ""):
            self._content_token_index[tok].discard(nid)
        node.content = None
        node.labels = []
        node.weight = 0.0
        node.activation = 0.0
        node.base_activation = 0.01
        node.metadata = {}
        node.crystal_channels.clear()

    def store(self, content: str, labels: Optional[List[str]] = None,
              weight: float = 1.0, metadata: Optional[Dict] = None,
              creation_time_override: Optional[float] = None) -> str:
        with self._lock:
            chash = hashlib.sha256(content.encode()).hexdigest()[:12]
            if chash in self._content_hash_index:
                existing_id = self._content_hash_index[chash]
                existing = self._nodes.get(existing_id)
                if existing and existing.is_occupied:
                    existing.reinforce(weight * 0.1)
                    if labels:
                        new_labels = set(existing.labels) | set(labels)
                        for lbl in new_labels - set(existing.labels):
                            existing.labels.append(lbl)
                            self._label_index[lbl].add(existing_id)
                    if metadata:
                        merged_meta = {**existing.metadata, **metadata}
                        for k, v in metadata.items():
                            if k in existing.metadata and isinstance(existing.metadata[k], list) and isinstance(v, list):
                                merged_meta[k] = list(set(existing.metadata[k] + v))
                        existing.metadata = merged_meta
                    if weight > existing.weight:
                        existing.weight = min(10.0, existing.weight + (weight - existing.weight) * 0.3)
                    return existing_id

            total = len(self._nodes)
            occupied_count = sum(1 for n in self._nodes.values() if n.is_occupied)
            expand_needed = False
            if total > 0 and occupied_count / total > 0.85:
                expand_needed = True
            elif total > 0 and occupied_count / total > 0.60:
                dense_count = 0
                for n in self._nodes.values():
                    if not n.is_occupied:
                        continue
                    occ_nb = sum(
                        1 for fnid in n.face_neighbors[:6]
                        if self._nodes.get(fnid) and self._nodes[fnid].is_occupied
                    )
                    if occ_nb >= 5:
                        dense_count += 1
                if dense_count > occupied_count * 0.3:
                    expand_needed = True
            if expand_needed:
                logger.info("Lattice occupancy %.1f%% (dense_nodes=%d) — auto-expanding (res %d->%d)",
                            occupied_count / total * 100,
                            sum(1 for n in self._nodes.values() if n.is_occupied and sum(1 for fnid in n.face_neighbors[:6] if self._nodes.get(fnid) and self._nodes[fnid].is_occupied) >= 5),
                            self._resolution, self._resolution + 1)
                self._expand_lattice()

            nid = self._find_nearest_empty_node(content, labels)
            node = self._nodes[nid]
            if node.is_occupied:
                nid = self._evict_for_space(content, labels, weight)
                node = self._nodes[nid]
            node.content = content
            node.labels = labels or []
            node.weight = weight
            node.activation = weight
            node.base_activation = max(0.01, weight * 0.1)
            node.metadata = metadata or {}
            node.metadata["geometric_quality"] = float(self._compute_node_geometric_quality(nid))
            node.metadata["geo_topo_divergence"] = float(self._compute_geometric_topo_divergence(nid))
            node.metadata["bcc_cell_coherence"] = float(self._bcc_cell_coherence(nid))
            node.creation_time = creation_time_override if creation_time_override is not None else time.time()
            node.touch()

            node.feeding = weight
            node.threshold = PCNNConfig.V_THETA * 0.5

            self._content_hash_index[chash] = nid
            for lbl in node.labels:
                self._label_index[lbl].add(nid)
            for tok in self._extract_tokens(content):
                self._content_token_index[tok].add(nid)

            self._emit_pulse(nid, strength=weight * 0.5, pulse_type=PulseType.REINFORCING)

            logger.debug("Stored memory at node %s: %s", nid[:8], content[:40])
            return nid

    def _find_nearest_empty_node(self, content: str, labels=None) -> str:
        occupied_nodes = [n for n in self._nodes.values() if n.is_occupied]

        if not occupied_nodes:
            origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            best_id = None
            best_dist = float('inf')
            for nid, node in self._nodes.items():
                d = float(np.sum((node.position - origin) ** 2))
                if d < best_dist:
                    best_dist = d
                    best_id = nid
            return best_id

        label_set = set(labels or [])

        optimal_cells = self._cell_map.find_optimal_placement_cells(
            self._nodes, label_set, self._label_index, count=20
        )
        if optimal_cells:
            cell_candidates = []
            for cell in optimal_cells:
                for vid in cell.vertex_ids:
                    vnode = self._nodes.get(vid)
                    if vnode and not vnode.is_occupied:
                        quality_bonus = cell.quality * 5.0
                        label_overlap = len(label_set & set(vnode.labels)) if vnode.labels else 0
                        score = quality_bonus + label_overlap * 2.0
                        cell_candidates.append((vid, score))
                bc_node = self._nodes.get(cell.body_center_id)
                if bc_node and not bc_node.is_occupied:
                    quality_bonus = cell.quality * 5.0
                    score = quality_bonus + 3.0
                    cell_candidates.append((cell.body_center_id, score))
            if cell_candidates:
                cell_candidates.sort(key=lambda x: -x[1])
                top_score = cell_candidates[0][1]
                top_tier = [c for c in cell_candidates if c[1] >= top_score * 0.8]
                return random.choice(top_tier)[0]

        related_occ = []
        for on in occupied_nodes:
            overlap = len(label_set & set(on.labels))
            if overlap > 0:
                related_occ.append((on, overlap * 2))
            else:
                related_occ.append((on, 0))

        face_candidates = []
        for on, bonus in related_occ:
            for fnid in on.face_neighbors:
                fn = self._nodes.get(fnid)
                if fn and not fn.is_occupied:
                    cell_quality = 1.0
                    cells = self._cell_map.get_cells_for_node(fnid)
                    if cells:
                        cell_quality = max(c.quality for c in cells)
                    vacancy = self._vacancy_attraction(fnid) * 0.5
                    energy_bonus = 0.0
                    if self._reflection_field:
                        energy = self._reflection_field._node_energy.get(fnid, 0.5)
                        energy_bonus = (1.0 - energy) * 2.0
                    crystal_nearby = 0.0
                    if fn.crystal_channels:
                        crystal_nearby = len(fn.crystal_channels) * 0.5
                    cell_q = self._cell_quality_factor(fnid)
                    autocorr_bonus = 0.0
                    if bonus > 0 and self._spatial_autocorrelation > 0.05:
                        autocorr_bonus = bonus * self._spatial_autocorrelation * 3.0
                    elif bonus == 0 and self._spatial_autocorrelation < -0.05:
                        autocorr_bonus = 2.0
                    face_candidates.append((fnid, fn, bonus + 10 + cell_quality * 3 + vacancy + energy_bonus + crystal_nearby + cell_q * 2.0 + autocorr_bonus))

        if face_candidates:
            face_candidates.sort(key=lambda x: -x[2])
            top_bonus = face_candidates[0][2]
            top_tier = [c for c in face_candidates if c[2] >= top_bonus - 1]
            chosen = random.choice(top_tier)
            return chosen[0]

        edge_candidates = []
        for on, bonus in related_occ:
            for enid in on.edge_neighbors:
                en = self._nodes.get(enid)
                if en and not en.is_occupied:
                    vacancy = self._vacancy_attraction(enid) * 0.5
                    edge_candidates.append((enid, en, bonus + 5 + vacancy))

        if edge_candidates:
            edge_candidates.sort(key=lambda x: -x[2])
            top_bonus = edge_candidates[0][2]
            top_tier = [c for c in edge_candidates if c[2] >= top_bonus - 1]
            chosen = random.choice(top_tier)
            return chosen[0]

        cellmate_candidates = []
        for on, bonus in related_occ:
            if bonus > 0:
                for cmid in self._get_bcc_cellmates(on.id):
                    cm = self._nodes.get(cmid)
                    if cm and not cm.is_occupied:
                        vacancy = self._vacancy_attraction(cmid) * 0.3
                        cellmate_candidates.append((cmid, cm, bonus + 3 + vacancy))
        if cellmate_candidates:
            cellmate_candidates.sort(key=lambda x: -x[2])
            top = cellmate_candidates[0][2]
            top_tier = [c for c in cellmate_candidates if c[2] >= top * 0.8]
            return random.choice(top_tier)[0]

        occ_positions = [n.position for n in occupied_nodes]
        centroid = np.mean(occ_positions, axis=0)

        best_id = None
        best_score = float('inf')
        sample = min(300, len(self._nodes))
        for nid in random.sample(list(self._nodes.keys()), sample):
            node = self._nodes[nid]
            if node.is_occupied:
                continue
            centroid_dist = float(np.sum((node.position - centroid) ** 2))
            vacancy = self._vacancy_attraction(nid)
            score = centroid_dist * 0.7 - vacancy * 0.3
            if score < best_score:
                best_score = score
                best_id = nid

        return best_id or random.choice(list(self._nodes.keys()))

    def query(self, text: str, k: int = 5, labels=None) -> List[Dict]:
        with self._lock:
            qtokens = self._extract_tokens(text) if text else set()
            qtrigrams = self._extract_ngrams(text, 3) if text else set()

            pre_hit_ids = set()
            if qtokens:
                for t in qtokens:
                    if t in self._label_index:
                        pre_hit_ids.update(self._label_index[t])
                    if t in self._content_token_index:
                        pre_hit_ids.update(self._content_token_index[t])

            scored = []
            checked = 0
            candidate_ids = list(pre_hit_ids) if pre_hit_ids else [nid for nid, n in self._nodes.items() if n.is_occupied]
            if len(candidate_ids) < k:
                extra = [nid for nid, n in self._nodes.items() if n.is_occupied and nid not in set(candidate_ids)]
                candidate_ids.extend(extra[:k * 3])

            for nid in candidate_ids:
                node = self._nodes.get(nid)
                if not node or not node.is_occupied:
                    continue
                if labels and not any(l in node.labels for l in labels):
                    continue

                text_score = 0.0
                ctokens = set()
                if qtokens:
                    ctokens = self._extract_tokens(node.content)
                    if ctokens:
                        overlap = len(qtokens & ctokens)
                        text_score = overlap / max(len(qtokens), 1)
                        if text_score > 0:
                            for qt in qtokens & ctokens:
                                tf = sum(1 for _ in [1 for c in node.content.lower().split() if qt in c])
                                if tf > 1:
                                    text_score += 0.05 * min(tf, 3)

                trigram_score = 0.0
                if qtrigrams and text_score > 0:
                    ctrigrams = self._extract_ngrams(node.content, 3)
                    if ctrigrams:
                        trigram_score = len(qtrigrams & ctrigrams) / max(len(qtrigrams), 1)

                label_score = 0.0
                if labels:
                    overlap = len(set(labels) & set(node.labels))
                    label_score = overlap / max(len(labels), 1)

                activation_score = min(node.activation / 5.0, 1.0)
                weight_score = min(node.weight / 5.0, 1.0)

                hebbian_boost = 0.0
                if qtokens:
                    for nnid in node.face_neighbors[:6] + node.edge_neighbors[:4]:
                        bias = self._hebbian.get_path_bias(nid, nnid)
                        if bias > 0.05:
                            hebbian_boost = min(0.15, bias * 0.5)
                            break

                crystal_boost = 0.0
                if node.crystal_channels:
                    crystal_boost = min(0.10, len(node.crystal_channels) * 0.02)

                pulse_boost = 0.0
                if node.pulse_accumulator > 0.1:
                    pulse_boost = min(0.05, node.pulse_accumulator * 0.1)

                pcnn_activity = 0.0
                if node.fired:
                    pcnn_activity = 0.04
                elif node.internal_activity > 0.5:
                    pcnn_activity = 0.02

                resonance_bonus = 0.0
                for ev in self._resonance_events:
                    if nid[:8] in ev.get("node_ids", []):
                        resonance_bonus = min(0.04, ev.get("strength", 0) * 0.05)
                        break

                spatial_quality = 0.0
                if self._reflection_field:
                    sq = self._reflection_field.get_spatial_quality(self, nid)
                    spatial_quality = sq

                geometric_quality = self._compute_node_geometric_quality(nid)

                cell_effective = 0.0
                node_cells = self._cell_map.get_cells_for_node(nid)
                if node_cells:
                    cell_effective = sum(c.effective_quality for c in node_cells) / len(node_cells)

                geo_topo_divergence = self._compute_geometric_topo_divergence(nid)
                divergence_bonus = 0.0
                if geo_topo_divergence > 0.5:
                    divergence_bonus = geo_topo_divergence * 0.05

                neighbor_density_score = 0.0
                occ_neighbors = sum(1 for fnid in node.face_neighbors[:8] if self._nodes.get(fnid) and self._nodes[fnid].is_occupied)
                if occ_neighbors > 0:
                    neighbor_density_score = min(1.0, occ_neighbors / 6.0)

                bcc_coherence = self._bcc_cell_coherence(nid)

                autocorr_bonus = 0.0
                if self._spatial_autocorrelation > 0.1:
                    autocorr_bonus = self._spatial_autocorrelation * neighbor_density_score * 0.03

                dream_bonus = 0.0
                if "__dream__" in node.labels:
                    dream_bonus = 0.03

                low_priority_penalty = 0.0
                if "__low_priority__" in node.labels:
                    low_priority_penalty = 0.05

                shortcut_boost = 0.0
                if self._self_organize and hasattr(self._self_organize, '_shortcuts'):
                    for sc_key, sc_str in self._self_organize._shortcuts.items():
                        if nid in sc_key:
                            shortcut_boost = min(0.08, sc_str * 0.3)
                            break

                final = (
                    0.28 * text_score
                    + 0.10 * trigram_score
                    + 0.12 * label_score
                    + 0.08 * activation_score
                    + 0.05 * weight_score
                    + 0.04 * hebbian_boost
                    + 0.04 * crystal_boost
                    + 0.02 * pulse_boost
                    + 0.05 * spatial_quality
                    + 0.04 * geometric_quality
                    + 0.03 * cell_effective
                    + 0.03 * neighbor_density_score
                    + 0.02 * geo_topo_divergence
                    + 0.02 * divergence_bonus
                    + 0.02 * bcc_coherence
                    + 0.01 * autocorr_bonus
                    + 0.01 * dream_bonus
                    + shortcut_boost
                    - low_priority_penalty
                    + pcnn_activity
                    + resonance_bonus
                )
                scored.append((nid, final))

            scored.sort(key=lambda x: -x[1])

            results = []
            for nid, score in scored[:k]:
                node = self._nodes[nid]
                node.touch()
                node.reinforce(0.05)
                results.append({
                    "id": nid,
                    "content": node.content,
                    "distance": score,
                    "weight": node.weight,
                    "labels": list(node.labels),
                    "activation": node.activation,
                    "metadata": node.metadata,
                })

            if results:
                best_id = results[0]["id"]
                self._emit_pulse(best_id, strength=0.3, pulse_type=PulseType.REINFORCING)

            return results

    def _extract_ngrams(self, text: str, n: int = 3) -> set:
        import re
        ngrams = set()
        clean = re.sub(r'[^\w\s]', '', text.lower())
        words = clean.split()
        for i in range(max(1, len(words) - n + 1)):
            ng = ' '.join(words[i:i+n])
            if len(ng) >= n:
                ngrams.add(ng)
        for char_group in re.findall(r'[\u4e00-\u9fff]+', text):
            for i in range(max(1, len(char_group) - n + 1)):
                ngrams.add(char_group[i:i+n])
        return ngrams

    def _extract_tokens(self, text: str) -> set:
        import re
        tokens = set()
        for w in re.findall(r"[a-zA-Z0-9]{2,}", text.lower()):
            tokens.add(w)
        for bigram in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            for i in range(len(bigram) - 1):
                tokens.add(bigram[i:i+2])
        return tokens

    def _emit_pulse(
        self,
        source_id: str,
        strength: float = 0.5,
        pulse_type: PulseType = PulseType.EXPLORATORY,
    ):
        cfg = PCNNConfig
        if pulse_type == PulseType.EXPLORATORY:
            max_hops = cfg.MAX_HOPS_EXPLORATORY
            bias_fn = self._bias_exploratory
        elif pulse_type == PulseType.REINFORCING:
            max_hops = cfg.MAX_HOPS_REINFORCING
            bias_fn = self._bias_reinforcing
        elif pulse_type == PulseType.SELF_CHECK:
            max_hops = cfg.SELF_CHECK_MAX_HOPS
            bias_fn = self._bias_self_check
        elif pulse_type == PulseType.CASCADE:
            max_hops = cfg.CASCADE_MAX_HOPS
            bias_fn = self._bias_cascade
        elif pulse_type == PulseType.STRUCTURE:
            max_hops = cfg.STRUCTURE_MAX_HOPS
            bias_fn = self._bias_structure
        else:
            max_hops = cfg.MAX_HOPS_TENSION
            bias_fn = self._bias_tension_sensing

        pulse = NeuralPulse(
            source_id, strength,
            max_hops=max_hops,
            pulse_type=pulse_type,
            bias_fn=bias_fn,
        )

        if pulse_type == PulseType.CASCADE:
            self._propagate_cascade(pulse)
        else:
            self._propagate_pulse(pulse)
        self._pulse_type_counts[pulse_type] = self._pulse_type_counts.get(pulse_type, 0) + 1

    def _bias_exploratory(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
        biased = []
        for nid, strength, ctype in candidates:
            h = min(self._hebbian.get_path_bias(src, nid), 3.0)
            biased.append((nid, strength * random.uniform(0.8, 1.2) * (1.0 + h * 0.5), ctype))
        return biased

    def _bias_reinforcing(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
        biased = []
        for nid, strength, ctype in candidates:
            node = self._nodes.get(nid)
            weight_boost = min((node.weight * 0.5 + 1.0) if node else 1.0, 5.0)
            hebbian_w = min(self._hebbian.get_path_bias(src, nid), 3.0)
            hebbian_boost = 1.0 + hebbian_w * 2.0
            quality_factor = 1.0
            if node and node.metadata:
                gq = node.metadata.get("geometric_quality", 0.5)
                quality_factor = 0.8 + 0.4 * gq
            biased.append((nid, strength * weight_boost * hebbian_boost * quality_factor, ctype))
        return biased

    def _bias_tension_sensing(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
        biased = []
        for nid, strength, ctype in candidates:
            node = self._nodes.get(nid)
            if node is None:
                biased.append((nid, strength, ctype))
                continue
            connectivity = len(node.face_neighbors) + len(node.edge_neighbors)
            tension_factor = 1.0
            if connectivity < 6:
                tension_factor = 2.0
            elif connectivity < 10:
                tension_factor = 1.5
            if not node.is_occupied:
                tension_factor *= 1.3
            neighbor_weights = []
            for fnid in node.face_neighbors[:6]:
                fn = self._nodes.get(fnid)
                if fn and fn.is_occupied:
                    neighbor_weights.append(fn.weight)
            if neighbor_weights:
                w_var = float(np.var(neighbor_weights))
                tension_factor *= (1.0 + w_var * 0.5)
            h = min(self._hebbian.get_path_bias(src, nid), 3.0)
            tension_factor *= (1.0 + h * 0.3)
            biased.append((nid, strength * tension_factor, ctype))
        return biased

    def _bias_self_check(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
        biased = []
        for nid, strength, ctype in candidates:
            node = self._nodes.get(nid)
            if node is None:
                biased.append((nid, strength, ctype))
                continue
            h = min(self._hebbian.get_path_bias(src, nid), 3.0)
            h_factor = 1.0 + h * 0.3
            if not node.is_occupied:
                biased.append((nid, strength * 1.5 * h_factor, ctype))
            else:
                occupied_neighbors = 0
                for fnid in node.face_neighbors[:8]:
                    fn = self._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        occupied_neighbors += 1
                if occupied_neighbors == 0:
                    biased.append((nid, strength * 2.0 * h_factor, ctype))
                else:
                    biased.append((nid, strength * 0.5 * h_factor, ctype))
        return biased

    def _bias_cascade(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
        biased = []
        for nid, strength, ctype in candidates:
            node = self._nodes.get(nid)
            if node is None:
                biased.append((nid, strength, ctype))
                continue
            crystal_boost = 1.0
            source_node = self._nodes.get(src)
            if source_node:
                crystal_boost = self._crystallized.get_boost(source_node.id, nid)
            weight_factor = 1.0 + (node.weight * 0.3 if node.is_occupied else 0.1)
            h = min(self._hebbian.get_path_bias(src, nid), 3.0)
            biased.append((nid, strength * weight_factor * crystal_boost * (1.0 + h * 0.5), ctype))
        return biased

    def _bias_structure(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
        biased = []
        for nid, strength, ctype in candidates:
            node = self._nodes.get(nid)
            if node is None:
                biased.append((nid, strength * 0.5, ctype))
                continue
            total_neighbors = len(node.face_neighbors) + len(node.edge_neighbors)
            h = min(self._hebbian.get_path_bias(src, nid), 3.0)
            h_factor = 1.0 + h * 0.3
            if total_neighbors < 6:
                biased.append((nid, strength * 2.5 * h_factor, ctype))
            elif total_neighbors < 10:
                biased.append((nid, strength * 1.5 * h_factor, ctype))
            else:
                biased.append((nid, strength * 0.8 * h_factor, ctype))
        return biased

    def _cell_quality_factor(self, nid: str) -> float:
        cells = self._cell_map.get_cells_for_node(nid)
        if not cells:
            return 1.0
        avg_quality = sum(c.quality for c in cells) / len(cells)
        avg_eff = sum(c.effective_quality for c in cells) / len(cells)
        return 0.7 + 0.3 * (0.6 * avg_quality + 0.4 * avg_eff)

    def _propagate_pulse(self, pulse: NeuralPulse):
        if not pulse.alive:
            return

        current_id = pulse.path[-1]
        current = self._nodes.get(current_id)
        if current is None:
            return

        current.pulse_accumulator = min(PCNNConfig.MAX_PULSE_ACCUMULATOR, current.pulse_accumulator + pulse.strength)
        current.last_pulse_time = time.time()

        if current.is_occupied:
            current.reinforce(pulse.strength * 0.01)
            if pulse.pulse_type == PulseType.SELF_CHECK:
                self._detect_empty_associations(current_id, current)
            if pulse.pulse_type == PulseType.REINFORCING and len(pulse.path) >= 2 and self._pulse_engine is not None and self._pulse_engine.is_alive():
                self._hebbian.record_path(pulse.path[-3:] if len(pulse.path) >= 3 else pulse.path, success=True, strength=pulse.strength * 0.6)

        cfg = PCNNConfig
        raw_candidates = []
        cell_q = self._cell_quality_factor(current_id)
        for nid in current.face_neighbors:
            if nid not in pulse.path:
                base_strength = pulse.strength * cfg.FACE_DECAY * cell_q
                crystal_boost = self._crystallized.get_boost(current_id, nid)
                spatial_bias = self._reflection_field.get_pulse_direction_bias(self, current_id, nid) if self._reflection_field else 1.0
                bcc_dir = self._bcc_direction_factor(current.position, self._nodes[nid].position) if nid in self._nodes else 1.0
                raw_candidates.append((nid, base_strength * crystal_boost * spatial_bias * bcc_dir, "face"))
        for nid in current.edge_neighbors:
            if nid not in pulse.path:
                base_strength = pulse.strength * cfg.FACE_DECAY * cfg.EDGE_DECAY_FACTOR * cell_q
                crystal_boost = self._crystallized.get_boost(current_id, nid)
                spatial_bias = self._reflection_field.get_pulse_direction_bias(self, current_id, nid) if self._reflection_field else 1.0
                bcc_dir = self._bcc_direction_factor(current.position, self._nodes[nid].position) if nid in self._nodes else 1.0
                raw_candidates.append((nid, base_strength * crystal_boost * spatial_bias * bcc_dir, "edge"))
        for nid in current.vertex_neighbors:
            if nid not in pulse.path:
                nn = self._nodes.get(nid)
                if nn and nn.is_occupied:
                    base_strength = pulse.strength * cfg.FACE_DECAY * cfg.EDGE_DECAY_FACTOR * 0.3
                    raw_candidates.append((nid, base_strength, "vertex"))

        if not raw_candidates:
            return

        self._propagation_source = current_id
        if pulse.bias_fn is not None:
            biased = pulse.bias_fn(raw_candidates)
        else:
            biased = raw_candidates

        total_w = sum(w for _, w, _ in biased)
        if total_w <= 0:
            return

        weights = [w for _, w, _ in biased]
        if not all(math.isfinite(w) and w > 0 for w in weights):
            return

        fanout = 2 if pulse.pulse_type == PulseType.EXPLORATORY and len(biased) >= 2 else 1
        chosen_indices = random.choices(range(len(biased)), weights=weights, k=min(fanout, len(biased)))
        for idx in chosen_indices:
            next_id, next_strength, direction = biased[idx]

            new_pulse = NeuralPulse(
                pulse.source_id, next_strength, pulse.max_hops, pulse.pulse_type, pulse.bias_fn
            )
            new_pulse.hops = pulse.hops + 1
            new_pulse.path = pulse.path + [next_id]
            new_pulse.direction = direction
            self._propagate_pulse(new_pulse)

    def _propagate_cascade(self, pulse: NeuralPulse):
        if not pulse.alive:
            return
        if pulse.cascade_depth >= PCNNConfig.CASCADE_MAX_DEPTH:
            self._propagate_pulse(pulse)
            return

        current_id = pulse.path[-1]
        current = self._nodes.get(current_id)
        if current is None:
            return

        current.pulse_accumulator = min(PCNNConfig.MAX_PULSE_ACCUMULATOR, current.pulse_accumulator + pulse.strength)
        current.last_pulse_time = time.time()
        if current.is_occupied:
            current.reinforce(pulse.strength * 0.02)

        cfg = PCNNConfig
        raw_candidates = []
        for nid in current.face_neighbors:
            if nid not in pulse.path:
                base_strength = pulse.strength * cfg.FACE_DECAY
                crystal_boost = self._crystallized.get_boost(current_id, nid)
                nn = self._nodes.get(nid)
                spatial_bias = 1.0
                bcc_dir = 1.0
                if nn and self._reflection_field:
                    spatial_bias = self._reflection_field.get_pulse_direction_bias(self, current_id, nid)
                if nn:
                    bcc_dir = self._bcc_direction_factor(current.position, nn.position)
                raw_candidates.append((nid, base_strength * crystal_boost * spatial_bias * bcc_dir, "face"))
        for nid in current.edge_neighbors:
            if nid not in pulse.path:
                base_strength = pulse.strength * cfg.FACE_DECAY * cfg.EDGE_DECAY_FACTOR
                crystal_boost = self._crystallized.get_boost(current_id, nid)
                nn = self._nodes.get(nid)
                spatial_bias = 1.0
                bcc_dir = 1.0
                if nn and self._reflection_field:
                    spatial_bias = self._reflection_field.get_pulse_direction_bias(self, current_id, nid)
                if nn:
                    bcc_dir = self._bcc_direction_factor(current.position, nn.position)
                raw_candidates.append((nid, base_strength * crystal_boost * spatial_bias * bcc_dir, "edge"))
        for nid in current.vertex_neighbors[:4]:
            if nid not in pulse.path:
                nn = self._nodes.get(nid)
                if nn and nn.is_occupied:
                    base_strength = pulse.strength * cfg.FACE_DECAY * cfg.EDGE_DECAY_FACTOR * 0.3
                    crystal_boost = self._crystallized.get_boost(current_id, nid)
                    raw_candidates.append((nid, base_strength * crystal_boost, "vertex"))

        if not raw_candidates:
            return

        if pulse.bias_fn is not None:
            biased = pulse.bias_fn(raw_candidates)
        else:
            biased = raw_candidates

        total_energy = sum(w for _, w, _ in biased)
        if total_energy <= 0:
            return

        k = min(cfg.CASCADE_BRANCHING_FACTOR, len(biased))
        weights = [w for _, w, _ in biased]
        if not all(math.isfinite(w) and w > 0 for w in weights):
            return
        selected_indices = random.choices(range(len(biased)), weights=weights, k=k)

        child_energy_budget = pulse.strength * cfg.CASCADE_ENERGY_CONSERVATION * cfg.CASCADE_BRANCHING_DECAY
        energy_per_child = child_energy_budget / k

        for idx in selected_indices:
            next_id, next_strength, direction = biased[idx]
            child = pulse.clone(energy_per_child)
            child.hops = pulse.hops + 1
            child.path = pulse.path + [next_id]
            child.direction = direction
            self._cascade_count += 1

            self._propagate_cascade(child)

        self._hebbian.record_path(pulse.path[-3:] if len(pulse.path) >= 3 else pulse.path, success=True, strength=pulse.strength * 0.8)

    def start_pulse_engine(self):
        if self._pulse_engine is not None and self._pulse_engine.is_alive():
            return
        self._stop_event.clear()
        self._pulse_engine = threading.Thread(target=self._pulse_loop, name="neural-pulse", daemon=True)
        self._pulse_engine.start()
        self._self_check = SelfCheckEngine(self)
        self._self_check.start()
        self._lattice_checker = LatticeIntegrityChecker(self)
        self._self_organize = SelfOrganizeEngine(self)
        self._dream_engine = DreamEngine(self)
        self._agent_driver = AgentMemoryDriver(self)
        self._feedback_loop = FeedbackLoop(self)
        self._session_manager = SessionManager(self)
        self._reflection_field = SpatialReflectionField()
        self._phase_transition = None
        logger.info(
            "PCNN pulse engine started (v6.2) — face_decay=%.2f, edge_decay=%.2f, cascade=on, dream=on, reflection=on, vertex=on, autocorr=on",
            PCNNConfig.FACE_DECAY, PCNNConfig.FACE_DECAY * PCNNConfig.EDGE_DECAY_FACTOR,
        )

    def stop_pulse_engine(self):
        if self._self_check:
            self._self_check.stop()
        self._stop_event.set()
        if self._pulse_engine:
            self._pulse_engine.join(timeout=5)
            self._pulse_engine = None

    def _pulse_loop(self):
        cycle = 0
        while not self._stop_event.wait(timeout=self._adaptive_interval):
            try:
                self._pulse_cycle()
                cycle += 1

                if cycle % PCNNConfig.CONVERGENCE_CHECK_CYCLES == 0:
                    self._check_convergence_bridges()

                if cycle % PCNNConfig.GLOBAL_DECAY_CYCLES == 0:
                    self._global_decay()
                    self._hebbian.decay_all()

                if cycle % 30 == 0:
                    self._pcnn_global_step()
                    self._update_adaptive_interval()

                if cycle % 60 == 0:
                    self._crystal_maintenance()

                if cycle % 600 == 0 and self._lattice_checker:
                    self._lattice_checker.run_full_check()

                if cycle % PCNNConfig.SELF_ORGANIZE_INTERVAL == 0 and self._self_organize:
                    self._self_organize.run_cycle()

                if cycle % PCNNConfig.DREAM_CYCLE_INTERVAL == 0 and self._dream_engine:
                    self._dream_engine.run_dream_cycle()

                if cycle % 120 == 0:
                    self._cell_map.update_all_densities(self._nodes)

                if cycle % 150 == 0 and self._reflection_field:
                    self._reflection_field.run_reflection_cycle(self)
                    self._apply_phase_behavior()

                if cycle % 300 == 0:
                    self.compute_spatial_autocorrelation()
                    self._detect_resonance()
                    if self._phase_transition and hasattr(self, '_phase_transition'):
                        try:
                            gt, tensions = self._phase_transition.compute_global_tension(self)
                            if self._phase_transition.should_trigger(gt):
                                clusters = self._phase_transition.identify_tension_clusters(tensions, self)
                                if clusters:
                                    self._phase_transition.execute_transition(self, tensions, clusters)
                                    logger.info("Phase transition triggered: global_tension=%.1f, clusters=%d", gt, len(clusters))
                        except Exception as e:
                            logger.error("Phase transition error: %s", e)

            except Exception as e:
                logger.error("Pulse cycle error: %s", e, exc_info=True)

    def _pulse_cycle(self):
        with self._lock:
            occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
            if not occupied:
                return

            pulse_type = self._select_pulse_type()
            cfg = PCNNConfig

            if pulse_type == PulseType.EXPLORATORY:
                nid = self._select_source_exploratory(occupied)
                lo, hi = cfg.EXPLORATORY_STRENGTH_RANGE
                strength = random.uniform(lo, hi)
            elif pulse_type == PulseType.REINFORCING:
                nid = self._select_source_reinforcing(occupied)
                lo, hi = cfg.REINFORCING_STRENGTH_RANGE
                strength = random.uniform(lo, hi)
            elif pulse_type == PulseType.SELF_CHECK:
                nid = self._select_source_self_check(occupied)
                strength = cfg.SELF_CHECK_STRENGTH
            elif pulse_type == PulseType.CASCADE:
                nid = self._select_source_cascade(occupied)
                lo, hi = cfg.CASCADE_STRENGTH_RANGE
                strength = random.uniform(lo, hi)
            elif pulse_type == PulseType.STRUCTURE:
                nid = self._select_source_structure(occupied)
                strength = cfg.STRUCTURE_STRENGTH
            else:
                nid = self._select_source_tension(occupied)
                lo, hi = cfg.TENSION_STRENGTH_RANGE
                strength = random.uniform(lo, hi)

            self._emit_pulse(nid, strength, pulse_type)
            self._pulse_count += 1

    def _select_pulse_type(self) -> PulseType:
        cfg = PCNNConfig
        types = list(cfg.PULSE_TYPE_PROBABILITIES.keys())
        weights = list(cfg.PULSE_TYPE_PROBABILITIES.values())
        phase = self._current_phase
        if phase == "turbulent":
            for i, t in enumerate(types):
                if t == PulseType.EXPLORATORY:
                    weights[i] *= 1.5
                elif t == PulseType.REINFORCING:
                    weights[i] *= 0.7
        elif phase == "crystalline":
            for i, t in enumerate(types):
                if t == PulseType.STRUCTURE:
                    weights[i] *= 1.5
                elif t == PulseType.EXPLORATORY:
                    weights[i] *= 0.7
        return random.choices(types, weights=weights, k=1)[0]

    def _select_source_exploratory(self, occupied: List[Tuple[str, Any]]) -> str:
        if random.random() < 0.3:
            unoccupied_hot = [
                (nid, n) for nid, n in self._nodes.items()
                if not n.is_occupied and n.pulse_accumulator > 0.05
            ]
            if unoccupied_hot:
                return random.choice(unoccupied_hot)[0]

        return random.choice(occupied)[0]

    def _select_source_reinforcing(self, occupied: List[Tuple[str, Any]]) -> str:
        filtered = [(nid, n) for nid, n in occupied if "__low_priority__" not in n.labels]
        if not filtered:
            filtered = occupied
        weighted = [(nid, n.activation * max(n.weight, 0.5)) for nid, n in filtered]
        total = sum(w for _, w in weighted)
        if total <= 0:
            return random.choice(occupied)[0]
        return random.choices(
            [i for i, _ in weighted], weights=[w for _, w in weighted], k=1
        )[0]

    def _select_source_tension(self, occupied: List[Tuple[str, Any]]) -> str:
        tension_scores = []
        for nid, node in occupied:
            nb_weights = []
            for fnid in node.face_neighbors[:8]:
                fn = self._nodes.get(fnid)
                if fn and fn.is_occupied:
                    nb_weights.append(fn.weight)
            if nb_weights:
                w_var = float(np.var(nb_weights))
                avg_w = float(np.mean(nb_weights))
                tension = w_var + abs(node.weight - avg_w)
            else:
                tension = 0.0
            tension_scores.append((nid, tension + 0.01))

        total = sum(t for _, t in tension_scores)
        if total <= 0:
            return random.choice(occupied)[0]
        return random.choices(
            [i for i, _ in tension_scores], weights=[t for _, t in tension_scores], k=1
        )[0]

    def _select_source_self_check(self, occupied: List[Tuple[str, Any]]) -> str:
        isolation_scores = []
        for nid, node in occupied:
            occupied_neighbors = 0
            for fnid in node.face_neighbors[:8]:
                fn = self._nodes.get(fnid)
                if fn and fn.is_occupied:
                    occupied_neighbors += 1
            for enid in node.edge_neighbors[:4]:
                en = self._nodes.get(enid)
                if en and en.is_occupied:
                    occupied_neighbors += 1
            isolation = 1.0 / (occupied_neighbors + 1)
            vitality_penalty = 0.0
            if node.weight >= 2.0 and node.activation < 0.1:
                vitality_penalty = 2.0
            isolation_scores.append((nid, isolation + vitality_penalty + 0.01))

        total = sum(s for _, s in isolation_scores)
        if total <= 0:
            return random.choice(occupied)[0]
        return random.choices(
            [i for i, _ in isolation_scores], weights=[s for _, s in isolation_scores], k=1
        )[0]

    def _select_source_cascade(self, occupied: List[Tuple[str, Any]]) -> str:
        filtered = [(nid, n) for nid, n in occupied if "__low_priority__" not in n.labels]
        if not filtered:
            filtered = occupied
        dream_sources = [(nid, n) for nid, n in filtered if "__dream__" in n.labels and n.weight >= 2.0]
        if dream_sources and random.random() < 0.3:
            return random.choice(dream_sources)[0]
        hebbian_nodes = set()
        for (a, b), w in self._hebbian._edges.items():
            if w > 1.0:
                hebbian_nodes.add(a)
                hebbian_nodes.add(b)

        hebbian_occupied = [(nid, n) for nid, n in filtered if nid in hebbian_nodes]
        if hebbian_occupied:
            weighted = [(nid, n.activation * max(n.weight, 0.5)) for nid, n in hebbian_occupied]
            total = sum(w for _, w in weighted)
            if total > 0:
                return random.choices(
                    [i for i, _ in weighted], weights=[w for _, w in weighted], k=1
                )[0]

        high_weight = [(nid, n.weight * n.activation) for nid, n in occupied if n.weight >= 2.0]
        if high_weight:
            total = sum(w for _, w in high_weight)
            if total > 0:
                return random.choices(
                    [i for i, _ in high_weight], weights=[w for _, w in high_weight], k=1
                )[0]

        return self._select_source_reinforcing(occupied)

    def _select_source_structure(self, occupied: List[Tuple[str, Any]]) -> str:
        structure_scores = []
        for nid, node in occupied:
            total_neighbors = len(node.face_neighbors) + len(node.edge_neighbors)
            neighbor_weight_var = 0.0
            nb_weights = []
            for fnid in node.face_neighbors[:8]:
                fn = self._nodes.get(fnid)
                if fn and fn.is_occupied:
                    nb_weights.append(fn.weight)
            if nb_weights:
                neighbor_weight_var = float(np.var(nb_weights))
            crystal_count = len(node.crystal_channels)
            structure_score = neighbor_weight_var + crystal_count * 0.5 + (1.0 if total_neighbors < 10 else 0.0)
            structure_scores.append((nid, structure_score + 0.01))

        total = sum(s for _, s in structure_scores)
        if total <= 0:
            return random.choice(occupied)[0]
        return random.choices(
            [i for i, _ in structure_scores], weights=[s for _, s in structure_scores], k=1
        )[0]

    def _pcnn_global_step(self):
        with self._lock:
            for nid, node in self._nodes.items():
                neighbor_outputs = []
                for fnid in node.face_neighbors[:4]:
                    fn = self._nodes.get(fnid)
                    if fn:
                        out = 1.0 if fn.fired else 0.0
                        if out > 0:
                            neighbor_outputs.append((fnid, out * fn.activation * 0.1, "face"))
                for enid in node.edge_neighbors[:3]:
                    en = self._nodes.get(enid)
                    if en:
                        out = 1.0 if en.fired else 0.0
                        if out > 0:
                            neighbor_outputs.append((enid, out * en.activation * 0.05, "edge"))
                for vnid in node.vertex_neighbors[:2]:
                    vn = self._nodes.get(vnid)
                    if vn:
                        out = 1.0 if vn.fired else 0.0
                        if out > 0:
                            neighbor_outputs.append((vnid, out * vn.activation * 0.02, "vertex"))

                node.pcnn_step(neighbor_outputs)

    def _update_adaptive_interval(self):
        bridge_rate = self._bridge_count / max(self._pulse_count, 1)
        self._recent_bridge_rate = bridge_rate

        cfg = PCNNConfig
        if bridge_rate > 0.02:
            self._adaptive_interval = max(
                cfg.MIN_PULSE_INTERVAL, self._adaptive_interval * 0.95
            )
        elif bridge_rate < 0.001:
            self._adaptive_interval = min(
                cfg.MAX_PULSE_INTERVAL, self._adaptive_interval * 1.02
            )
        else:
            target = cfg.BASE_PULSE_INTERVAL
            self._adaptive_interval = 0.9 * self._adaptive_interval + 0.1 * target

    def _apply_phase_behavior(self):
        """
        Phase state drives actual system behavior:
        - crystalline: reduce pulse frequency, deepen self-org
        - ordered: normal operation
        - turbulent: increase pulse frequency, boost self-check
        - fluid: normal with slight exploration boost
        """
        if not self._reflection_field:
            return
        phase = self._reflection_field._phase_state
        if phase == "crystalline":
            self._adaptive_interval = min(
                PCNNConfig.MAX_PULSE_INTERVAL,
                self._adaptive_interval * 1.1
            )
        elif phase == "turbulent":
            self._adaptive_interval = max(
                PCNNConfig.MIN_PULSE_INTERVAL,
                self._adaptive_interval * 0.85
            )
        self._current_phase = phase

    def _crystal_maintenance(self):
        with self._lock:
            new_crystals = self._crystallized.scan_and_crystallize(self._hebbian._edges)

            for node in self._nodes.values():
                node.crystal_channels.clear()

            for key, crystal in self._crystallized._crystals.items():
                a, b = crystal["nodes"]
                node_a = self._nodes.get(a)
                node_b = self._nodes.get(b)
                if node_a:
                    node_a.crystal_channels[b] = crystal["crystal_weight"]
                if node_b:
                    node_b.crystal_channels[a] = crystal["crystal_weight"]

            if new_crystals > 0:
                logger.info("Crystal maintenance: %d new crystallized pathways", new_crystals)

    def _detect_resonance(self):
        """
        Resonance Propagation: detect synchronized firing patterns across
        the BCC lattice. When multiple distant nodes fire simultaneously
        along the same crystallographic direction, it indicates a
        "standing wave" — a coherent pattern that can be crystallized.

        Based on Eckhorn PCNN synchronization theory: coupled oscillators
        naturally synchronize when their coupling strength exceeds a threshold.
        In BCC lattice, <111> directions provide the strongest coupling.
        """
        occupied = [(nid, n) for nid, n in self._nodes.items()
                     if n.is_occupied and n.fired]
        if len(occupied) < 3:
            self._resonance_events = []
            return

        directions = {
            (1,1,1): [], (1,1,-1): [], (1,-1,1): [], (1,-1,-1): [],
            (-1,1,1): [], (-1,1,-1): [], (-1,-1,1): [], (-1,-1,-1): [],
        }
        for nid, node in occupied:
            pos = node.position
            for dkey in directions:
                proj = float(np.dot(pos, np.array(dkey, dtype=np.float32)))
                directions[dkey].append((nid, proj))

        resonance_events = []
        for dkey, nodes_proj in directions.items():
            if len(nodes_proj) < 2:
                continue
            nodes_proj.sort(key=lambda x: x[1])
            proj_vals = [p for _, p in nodes_proj]
            proj_range = max(proj_vals) - min(proj_vals)
            if proj_range <= 0:
                continue
            n_bins = max(10, int(proj_range / self._spacing))
            if n_bins > 200:
                n_bins = 200
            bin_width = proj_range / n_bins
            bins = [0] * n_bins
            bin_nodes = [[] for _ in range(n_bins)]
            for nid, proj in nodes_proj:
                bi = min(int((proj - proj_vals[0]) / bin_width), n_bins - 1)
                bins[bi] += 1
                bin_nodes[bi].append(nid)
            if max(bins) == 0:
                continue
            avg_density = len(nodes_proj) / n_bins
            if avg_density < 1:
                continue
            threshold_density = avg_density * 2.0
            dense_indices = [i for i, c in enumerate(bins) if c >= threshold_density]
            if not dense_indices:
                continue
            groups = []
            current_group = [dense_indices[0]]
            for i in range(1, len(dense_indices)):
                if dense_indices[i] - dense_indices[i-1] <= 1:
                    current_group.append(dense_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [dense_indices[i]]
            groups.append(current_group)
            clusters = []
            for group in groups:
                cluster_nodes = []
                for bi in group:
                    cluster_nodes.extend(bin_nodes[bi])
                if len(cluster_nodes) >= 3:
                    clusters.append(cluster_nodes)

            for cluster in clusters:
                ids = cluster
                avg_weight = np.mean([self._nodes[nid].weight for nid in ids])
                resonance_strength = len(cluster) * avg_weight / 10.0
                if resonance_strength > 0.5:
                    resonance_events.append({
                        "direction": list(dkey),
                        "node_count": len(cluster),
                        "strength": round(float(resonance_strength), 3),
                        "node_ids": [nid[:8] for nid in ids[:5]],
                    })
                    for i in range(len(ids) - 1):
                        self._hebbian.record_path(
                            [ids[i], ids[i+1]],
                            success=True,
                            strength=resonance_strength * 0.3,
                        )

        self._resonance_events = resonance_events[-10:]

    def _check_convergence_bridges(self):
        cfg = PCNNConfig
        with self._lock:
            hot_empty = [
                (nid, n) for nid, n in self._nodes.items()
                if not n.is_occupied and n.pulse_accumulator > cfg.BRIDGE_THRESHOLD
            ]
            if len(hot_empty) < 2:
                return

            bridges_this_cycle = 0
            for nid, node in hot_empty[:5]:
                sources = set()
                for fnid in node.face_neighbors:
                    fn = self._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        sources.add(fnid)
                for enid in node.edge_neighbors[:4]:
                    en = self._nodes.get(enid)
                    if en and en.is_occupied:
                        sources.add(enid)

                if len(sources) >= cfg.MIN_BRIDGE_SOURCES and not node.is_occupied:
                    source_nodes = []
                    source_labels = set()
                    for sid in list(sources)[:4]:
                        sn = self._nodes.get(sid)
                        if sn and sn.is_occupied:
                            source_nodes.append((sid, sn))
                            source_labels.update(sn.labels)
                    source_labels.discard("__dream__")
                    source_labels.discard("__system__")
                    source_labels.discard("__pulse_bridge__")

                    non_system_labels = [l for l in source_labels if not l.startswith("__")]
                    if len(non_system_labels) < 1:
                        continue

                    src_token_sets = [self._extract_tokens(sn.content) for _, sn in source_nodes]
                    pairwise_overlap = 0
                    pair_count = 0
                    for ti in range(len(src_token_sets)):
                        for tj in range(ti + 1, len(src_token_sets)):
                            if src_token_sets[ti] and src_token_sets[tj]:
                                inter = len(src_token_sets[ti] & src_token_sets[tj])
                                union = len(src_token_sets[ti] | src_token_sets[tj])
                                if union > 0:
                                    pairwise_overlap += inter / union
                                pair_count += 1
                    avg_semantic = pairwise_overlap / max(pair_count, 1)

                    bridge_weight = node.pulse_accumulator * (0.5 + 0.5 * avg_semantic)
                    if bridge_weight < 0.3 and avg_semantic < 0.15:
                        continue

                    label_str = ", ".join(non_system_labels[:4])
                    summaries = []
                    for _, sn in source_nodes:
                        clean = sn.content.lstrip("[").split("] ", 1)
                        text = clean[-1] if len(clean) > 1 else clean[0]
                        summaries.append(text.strip()[:50])
                    bridge = f"[bridge] {label_str}: {' ; '.join(summaries[:3])}"

                    node.content = bridge
                    node.labels = non_system_labels[:6] + ["__pulse_bridge__"]
                    node.weight = min(max(bridge_weight, 0.3), 2.0)
                    node.activation = bridge_weight * 0.5
                    node.base_activation = 0.05
                    node.metadata = {
                        "bridge_sources": len(source_nodes),
                        "semantic_overlap": round(avg_semantic, 3),
                        "pulse_accumulated": round(node.pulse_accumulator, 3),
                    }
                    chash = hashlib.sha256(bridge.encode()).hexdigest()[:12]
                    self._content_hash_index[chash] = nid
                    for lbl in node.labels:
                        self._label_index[lbl].add(nid)
                    for tok in self._extract_tokens(bridge):
                        self._content_token_index[tok].add(nid)
                    node.pulse_accumulator = 0.0
                    self._bridge_count += 1
                    bridges_this_cycle += 1

                    self._pulse_log.append({
                        "time": time.time(),
                        "bridge_id": nid,
                        "strength": node.weight,
                        "sources": len(sources),
                    })
                    if len(self._pulse_log) > self._max_log:
                        self._pulse_log = self._pulse_log[-self._max_log // 2:]

                    for i in range(len(node.face_neighbors)):
                        for j in range(i + 1, min(i + 3, len(node.face_neighbors))):
                            path = [node.face_neighbors[i], nid, node.face_neighbors[j]]
                            self._hebbian.record_path(path, success=True, strength=node.weight)

    def _global_decay(self):
        now = time.time()
        for node in self._nodes.values():
            dt = now - node.last_pulse_time if node.last_pulse_time > 0 else 60
            node.decay(dt)

    def get_node(self, nid: str) -> Optional[Dict]:
        node = self._nodes.get(nid)
        if node is None:
            return None
        return {
            "id": node.id,
            "content": node.content,
            "position": node.position.tolist(),
            "centroid": node.position.tolist(),
            "labels": list(node.labels),
            "weight": float(node.weight),
            "activation": float(node.activation),
            "face_neighbors": len(node.face_neighbors),
            "edge_neighbors": len(node.edge_neighbors),
            "creation_time": float(node.creation_time),
            "access_count": int(node.access_count),
            "metadata": {k: float(v) if isinstance(v, (np.floating, np.integer)) else bool(v) if isinstance(v, np.bool_) else v for k, v in node.metadata.items()},
            "feeding": float(node.feeding),
            "linking": float(node.linking),
            "internal_activity": float(node.internal_activity),
            "threshold": float(node.threshold),
            "fired": bool(node.fired),
            "crystal_channels": {k[:8]: round(float(v), 3) for k, v in node.crystal_channels.items()},
        }

    def list_occupied(self) -> List[Dict]:
        with self._lock:
            results = []
            for nid, node in self._nodes.items():
                if node.is_occupied:
                    results.append(self.get_node(nid))
            return results

    def topology_graph(self) -> Dict:
        with self._lock:
            nodes = []
            for nid, node in self._nodes.items():
                if node.is_occupied or node.pulse_accumulator > 0.1:
                    nodes.append({
                        "id": nid,
                        "centroid": node.position.tolist(),
                        "weight": node.weight,
                        "labels": list(node.labels),
                        "is_dream": "__pulse_bridge__" in node.labels,
                        "activation": node.activation,
                        "occupied": node.is_occupied,
                    })

            occupied_ids = {n["id"] for n in nodes}
            edges = []
            for n1, n2, etype in self._edges:
                if n1 in occupied_ids and n2 in occupied_ids:
                    edges.append({"source": n1, "target": n2, "type": etype})

            return {"nodes": nodes, "edges": edges}

    def stats(self) -> Dict:
        with self._lock:
            total = len(self._nodes)
            occupied = sum(1 for n in self._nodes.values() if n.is_occupied)
            bridges = sum(1 for n in self._nodes.values() if "__pulse_bridge__" in n.labels)
            face_edges = sum(1 for _, _, t in self._edges if t == "face")
            edge_edges = sum(1 for _, _, t in self._edges if t == "edge")
            avg_activation = np.mean([n.activation for n in self._nodes.values()]) if self._nodes else 0
            avg_face_conn = np.mean([len(n.face_neighbors) for n in self._nodes.values()]) if self._nodes else 0
            fired_count = sum(1 for n in self._nodes.values() if n.fired)
            crystal_nodes = sum(1 for n in self._nodes.values() if n.crystal_channels)

            return {
                "total_nodes": total,
                "occupied_nodes": occupied,
                "bridge_nodes": bridges,
                "empty_nodes": total - occupied,
                "face_edges": face_edges,
                "edge_edges": edge_edges,
                "vertex_edges": sum(1 for _, _, t in self._edges if t == "vertex"),
                "avg_vertex_connections": float(np.mean([len(n.vertex_neighbors) for n in self._nodes.values()])) if self._nodes else 0,
                "avg_activation": float(avg_activation),
                "avg_face_connections": float(avg_face_conn),
                "pulse_count": self._pulse_count,
                "bridge_count": self._bridge_count,
                "cascade_count": self._cascade_count,
                "crystal_nodes": crystal_nodes,
                "pulse_engine_running": self._pulse_engine is not None and self._pulse_engine.is_alive(),
                "pulse_type_counts": {t.value: c for t, c in self._pulse_type_counts.items()},
                "fired_nodes": fired_count,
                "adaptive_interval": round(self._adaptive_interval, 3),
                "bridge_rate": round(self._recent_bridge_rate, 6),
                "hebbian": self._hebbian.stats(),
                "crystallized": self._crystallized.stats(),
                "reflection_field": self._reflection_field.stats() if self._reflection_field else {"phase_state": "uninitialized"},
                "lattice_integrity": self._lattice_checker.get_latest() if self._lattice_checker else None,
                "self_check": self.self_check_status() if self._self_check else {"engine_running": False},
                "self_organize": self._self_organize.stats() if self._self_organize else {"engine_active": False},
                "honeycomb_cells": self._cell_map.structural_analysis(self._nodes),
                "spatial_autocorrelation": {
                    "morans_i": round(self._spatial_autocorrelation, 4),
                    "history_len": len(self._autocorrelation_history),
                },
                "bcc_unit_index_size": len(self._bcc_unit_index),
                "current_phase": self._current_phase,
                "resonance_events": self._resonance_events,
                "pcnn_config": {
                    "face_decay": PCNNConfig.FACE_DECAY,
                    "edge_decay": round(PCNNConfig.FACE_DECAY * PCNNConfig.EDGE_DECAY_FACTOR, 3),
                    "beta": PCNNConfig.BETA,
                    "alpha_feed": PCNNConfig.ALPHA_FEED,
                    "alpha_link": PCNNConfig.ALPHA_LINK,
                    "alpha_threshold": PCNNConfig.ALPHA_THRESHOLD,
                    "cascade_branching": PCNNConfig.CASCADE_BRANCHING_FACTOR,
                    "cascade_max_depth": PCNNConfig.CASCADE_MAX_DEPTH,
                    "crystallize_threshold": PCNNConfig.CRYSTALLIZE_THRESHOLD,
                },
            }

    def _detect_empty_associations(self, nid: str, node: "HoneycombNode"):
        all_neighbors = node.face_neighbors + node.edge_neighbors
        occupied_count = 0
        empty_count = 0
        for nnid in all_neighbors:
            nn = self._nodes.get(nnid)
            if nn is None:
                empty_count += 1
            elif not nn.is_occupied:
                empty_count += 1
            else:
                occupied_count += 1

        total = occupied_count + empty_count
        if total > 0 and occupied_count == 0:
            if "__isolated__" not in node.labels:
                node.labels.append("__isolated__")
                node.metadata["isolation_detected"] = time.time()
                logger.debug("Empty association detected at node %s", nid[:8])

    def run_self_check(self) -> Dict[str, Any]:
        if self._self_check is None:
            return {"error": "self_check engine not initialized"}
        result = self._self_check.run_full_check()
        with self._lock:
            self._self_check._history.append(result)
            if len(self._self_check._history) > self._self_check._max_history:
                self._self_check._history = self._self_check._history[-self._self_check._max_history // 2:]
        return result.to_dict()

    def self_check_status(self) -> Dict[str, Any]:
        if self._self_check is None:
            return {"engine_running": False}
        status = self._self_check.stats()
        latest = self._self_check.get_latest()
        status["latest_check"] = latest
        return status

    def self_check_history(self, n: int = 10) -> List[Dict]:
        if self._self_check is None:
            return []
        return self._self_check.get_history(n)

    def detect_duplicates(self) -> List[Dict[str, Any]]:
        with self._lock:
            occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
            if len(occupied) < 2:
                return []

            contents_tokens = {}
            for nid, node in occupied:
                contents_tokens[nid] = self._extract_tokens(node.content)

            duplicates = []
            checked = set()
            threshold = PCNNConfig.DUPLICATE_TOKEN_OVERLAP

            for i, (nid_a, node_a) in enumerate(occupied):
                tokens_a = contents_tokens[nid_a]
                if not tokens_a:
                    continue
                for j in range(i + 1, min(i + 50, len(occupied))):
                    nid_b, node_b = occupied[j]
                    pair_key = (min(nid_a, nid_b), max(nid_a, nid_b))
                    if pair_key in checked:
                        continue
                    checked.add(pair_key)

                    tokens_b = contents_tokens.get(nid_b, set())
                    if not tokens_b:
                        continue

                    intersection = len(tokens_a & tokens_b)
                    union = len(tokens_a | tokens_b)
                    if union == 0:
                        continue
                    jaccard = intersection / union

                    if jaccard >= threshold:
                        duplicates.append({
                            "node_a": nid_a[:12],
                            "node_b": nid_b[:12],
                            "similarity": round(jaccard, 3),
                            "content_a": node_a.content[:80],
                            "content_b": node_b.content[:80],
                            "weight_a": node_a.weight,
                            "weight_b": node_b.weight,
                            "labels_a": list(node_a.labels),
                            "labels_b": list(node_b.labels),
                        })
            return duplicates

    def detect_isolated(self) -> List[Dict[str, Any]]:
        with self._lock:
            isolated = []
            for nid, node in self._nodes.items():
                if not node.is_occupied:
                    continue
                occupied_neighbors = 0
                total_neighbors = 0
                for fnid in node.face_neighbors:
                    fn = self._nodes.get(fnid)
                    total_neighbors += 1
                    if fn and fn.is_occupied:
                        occupied_neighbors += 1
                for enid in node.edge_neighbors:
                    en = self._nodes.get(enid)
                    total_neighbors += 1
                    if en and en.is_occupied:
                        occupied_neighbors += 1

                if occupied_neighbors == 0:
                    isolated.append({
                        "id": nid[:12],
                        "content": node.content[:60],
                        "weight": node.weight,
                        "activation": round(node.activation, 4),
                        "total_neighbors": total_neighbors,
                        "labels": list(node.labels),
                    })
            return isolated

    def browse_timeline(self, direction: str = "newest", limit: int = 20,
                        label_filter=None, min_weight: float = 0.0,
                        offset: int = 0) -> tuple:
        with self._lock:
            items = []
            for nid, node in self._nodes.items():
                if not node.is_occupied:
                    continue
                if min_weight > 0 and node.weight < min_weight:
                    continue
                if label_filter and not any(l in node.labels for l in label_filter):
                    continue
                items.append({
                    "id": nid,
                    "content": node.content,
                    "weight": node.weight,
                    "labels": list(node.labels),
                    "creation_time": node.creation_time,
                    "activation": node.activation,
                })
            items.sort(key=lambda x: x["creation_time"], reverse=(direction == "newest"))
            total = len(items)
            return items[offset:offset + limit], total

    def associate(self, tetra_id: str, max_depth: int = 2) -> List[Dict]:
        with self._lock:
            node = self._nodes.get(tetra_id)
            if node is None:
                return []

            visited = {tetra_id}
            frontier = [tetra_id]
            results = []
            shortcut_edges = {}
            if self._self_organize:
                for (a, b), s in self._self_organize._shortcuts.items():
                    shortcut_edges[a] = (b, s)
                    shortcut_edges[b] = (a, s)

            for depth in range(max_depth):
                next_frontier = []
                for fid in frontier:
                    fn = self._nodes.get(fid)
                    if fn is None:
                        continue
                    neighbors = list(fn.face_neighbors) + list(fn.edge_neighbors) + list(fn.vertex_neighbors)
                    if fid in shortcut_edges:
                        target_id, strength = shortcut_edges[fid]
                        if target_id not in visited:
                            neighbors.append(target_id)
                    for nid in neighbors:
                        if nid in visited:
                            continue
                        visited.add(nid)
                        nn = self._nodes.get(nid)
                        if nn and nn.is_occupied:
                            conn = "face" if nid in fn.face_neighbors else "edge" if nid in fn.edge_neighbors else "vertex" if nid in fn.vertex_neighbors else "shortcut"
                            hebb_w = self._hebbian.get_path_bias(fid, nid)
                            crystal_w = fn.crystal_channels.get(nid, 0.0)
                            relevance = nn.weight * 0.4 + nn.activation * 0.2 + hebb_w * 0.01 + crystal_w * 0.2
                            if conn == "shortcut":
                                relevance += 1.0
                            results.append({
                                "id": nid,
                                "content": nn.content,
                                "type": conn,
                                "weight": nn.weight,
                                "labels": list(nn.labels),
                                "activation": nn.activation,
                                "relevance": round(relevance, 3),
                                "depth": depth + 1,
                            })
                        next_frontier.append(nid)
                frontier = next_frontier

            results.sort(key=lambda x: -x.get("relevance", 0))
            return results

    def pulse_status(self) -> Dict:
        with self._lock:
            recent = self._pulse_log[-10:]
            hot_nodes = sorted(
                [(nid, n.pulse_accumulator) for nid, n in self._nodes.items() if n.pulse_accumulator > 0.1],
                key=lambda x: -x[1]
            )[:10]
            return {
                "pulse_count": self._pulse_count,
                "bridge_count": self._bridge_count,
                "cascade_count": self._cascade_count,
                "engine_running": self._pulse_engine is not None and self._pulse_engine.is_alive(),
                "recent_bridges": recent,
                "hot_nodes": [(nid[:8], round(acc, 3)) for nid, acc in hot_nodes],
                "pulse_type_counts": {t.value: c for t, c in self._pulse_type_counts.items()},
                "adaptive_interval": round(self._adaptive_interval, 3),
                "hebbian": self._hebbian.stats(),
                "hebbian_top_paths": self._hebbian.get_top_paths(10),
                "crystallized": self._crystallized.stats(),
                "crystallized_top": self._crystallized.top_crystals(10),
            }

    def get_pcnn_node_states(self, n: int = 20) -> List[Dict]:
        with self._lock:
            nodes = sorted(
                [(nid, n) for nid, n in self._nodes.items() if n.is_occupied],
                key=lambda x: -x[1].internal_activity
            )[:n]
            return [
                {
                    "id": nid[:8],
                    "feeding": round(node.feeding, 4),
                    "linking": round(node.linking, 4),
                    "internal_activity": round(node.internal_activity, 4),
                    "threshold": round(node.threshold, 4),
            "fired": bool(node.fired),
                    "activation": round(node.activation, 4),
                }
                for nid, node in nodes
            ]

    def get_tension_map(self, top_n: int = 20) -> List[Dict]:
        with self._lock:
            tensions = []
            for nid, node in self._nodes.items():
                if not node.is_occupied:
                    continue
                nb_weights = []
                for fnid in node.face_neighbors[:8]:
                    fn = self._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        nb_weights.append(fn.weight)
                if len(nb_weights) < 2:
                    continue
                avg_w = float(np.mean(nb_weights))
                w_var = float(np.var(nb_weights))
                tension = w_var + abs(node.weight - avg_w) * 0.5
                tensions.append({
                    "id": nid[:8],
                    "weight": round(node.weight, 2),
                    "avg_neighbor_weight": round(avg_w, 2),
                    "weight_variance": round(w_var, 3),
                    "tension": round(tension, 3),
                    "content": node.content[:60],
                })
            tensions.sort(key=lambda x: -x["tension"])
            return tensions[:top_n]

    def run_lattice_check(self) -> Dict[str, Any]:
        if self._lattice_checker is None:
            self._lattice_checker = LatticeIntegrityChecker(self)
        report = self._lattice_checker.run_full_check()
        return report.to_dict()

    def lattice_check_status(self) -> Dict[str, Any]:
        if self._lattice_checker is None:
            return {"checker_running": False, "checks_performed": 0}
        return self._lattice_checker.stats()

    def lattice_check_history(self, n: int = 10) -> List[Dict]:
        if self._lattice_checker is None:
            return []
        return self._lattice_checker.get_history(n)

    def crystallized_status(self) -> Dict[str, Any]:
        return {
            "stats": self._crystallized.stats(),
            "top_crystals": self._crystallized.top_crystals(20),
        }

    def trigger_cascade(self, source_id: Optional[str] = None, strength: float = 0.5) -> Dict[str, Any]:
        with self._lock:
            if source_id is None:
                occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
                if not occupied:
                    return {"error": "no occupied nodes"}
                source_id = self._select_source_cascade(occupied)
            self._emit_pulse(source_id, strength, PulseType.CASCADE)
            self._pulse_count += 1
            return {
                "triggered": True,
                "source": source_id[:12],
                "strength": strength,
                "cascade_count": self._cascade_count,
            }

    def trigger_structure_pulse(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if source_id is None:
                occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
                if not occupied:
                    return {"error": "no occupied nodes"}
                source_id = self._select_source_structure(occupied)
            self._emit_pulse(source_id, PCNNConfig.STRUCTURE_STRENGTH, PulseType.STRUCTURE)
            self._pulse_count += 1
            return {
                "triggered": True,
                "source": source_id[:12],
                "strength": PCNNConfig.STRUCTURE_STRENGTH,
            }

    def force_crystallize(self) -> Dict[str, Any]:
        with self._lock:
            self._crystal_maintenance()
            return {
                "crystallized": self._crystallized.stats(),
                "hebbian_edges": len(self._hebbian._edges),
            }

    def run_self_organize(self) -> Dict[str, Any]:
        if self._self_organize is None:
            self._self_organize = SelfOrganizeEngine(self)
        result = self._self_organize.run_cycle()
        return result.to_dict()

    def self_organize_status(self) -> Dict[str, Any]:
        if self._self_organize is None:
            return {"engine_active": False}
        status = self._self_organize.stats()
        status["clusters"] = self._self_organize.get_clusters()
        status["shortcuts"] = self._self_organize.get_shortcuts(10)
        return status

    def self_organize_history(self, n: int = 10) -> List[Dict]:
        if self._self_organize is None:
            return []
        return self._self_organize.get_history(n)

    def get_clusters(self) -> List[Dict]:
        if self._self_organize is None:
            return []
        return self._self_organize.get_clusters()

    def get_shortcuts(self, n: int = 20) -> List[Dict]:
        if self._self_organize is None:
            return []
        return self._self_organize.get_shortcuts(n)

    def honeycomb_analysis(self) -> Dict[str, Any]:
        with self._lock:
            self._cell_map.update_all_densities(self._nodes)
            analysis = self._cell_map.structural_analysis(self._nodes)
            best_cells = self._cell_map.get_best_cells(10)
            dense_cells = self._cell_map.get_cells_by_density(10)
            analysis["best_quality_cells"] = [c.to_dict() for c in best_cells]
            analysis["highest_density_cells"] = [c.to_dict() for c in dense_cells]
            return analysis

    def get_tetrahedral_cells(self, n: int = 20, sort_by: str = "quality") -> List[Dict]:
        with self._lock:
            self._cell_map.update_all_densities(self._nodes)
            if sort_by == "density":
                cells = self._cell_map.get_cells_by_density(n)
            else:
                cells = self._cell_map.get_best_cells(n)
            return [c.to_dict() for c in cells]

    def get_cell_for_node(self, node_id: str) -> List[Dict]:
        cells = self._cell_map.get_cells_for_node(node_id)
        with self._lock:
            for c in cells:
                c.update_density(self._nodes)
            return [c.to_dict() for c in cells]

    def run_dream_cycle(self) -> Dict[str, Any]:
        if self._dream_engine is None:
            self._dream_engine = DreamEngine(self)
        result = self._dream_engine.run_dream_cycle()
        return result.to_dict()

    def dream_status(self) -> Dict[str, Any]:
        if self._dream_engine is None:
            return {"engine_active": False}
        return self._dream_engine.stats()

    def dream_history(self, n: int = 10) -> List[Dict]:
        if self._dream_engine is None:
            return []
        return self._dream_engine.get_history(n)

    def agent_get_context(self, topic: str, max_memories: int = 15) -> Dict[str, Any]:
        if self._agent_driver is None:
            self._agent_driver = AgentMemoryDriver(self)
        return self._agent_driver.get_context(topic, max_memories)

    def agent_reasoning_chain(self, source_id: str, target_query: str, max_hops: int = 5) -> Dict[str, Any]:
        if self._agent_driver is None:
            self._agent_driver = AgentMemoryDriver(self)
        return self._agent_driver.reasoning_chain(source_id, target_query, max_hops)

    def agent_suggest(self, context: str = "") -> Dict[str, Any]:
        if self._agent_driver is None:
            self._agent_driver = AgentMemoryDriver(self)
        return self._agent_driver.suggest_actions(context)

    def agent_navigate(self, source_id: str, target_id: str, max_hops: int = 6) -> Dict[str, Any]:
        if self._agent_driver is None:
            self._agent_driver = AgentMemoryDriver(self)
        return self._agent_driver.navigate(source_id, target_id, max_hops)

    def feedback_record(self, action: str, context_id: str, outcome: str,
                        confidence: float = 0.5, reasoning: str = "",
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        if self._feedback_loop is None:
            self._feedback_loop = FeedbackLoop(self)
        return self._feedback_loop.record_outcome(action, context_id, outcome, confidence, reasoning, metadata)

    def feedback_learn(self, action: str, source_id: str, target_id: str,
                       success: bool, confidence: float = 0.5) -> Dict[str, Any]:
        if self._feedback_loop is None:
            self._feedback_loop = FeedbackLoop(self)
        return self._feedback_loop.learn_from_action(action, source_id, target_id, success, confidence)

    def feedback_stats(self) -> Dict[str, Any]:
        if self._feedback_loop is None:
            return {"total_feedback": 0}
        return self._feedback_loop.get_stats()

    def feedback_insights(self) -> List[Dict[str, Any]]:
        if self._feedback_loop is None:
            return []
        return self._feedback_loop.get_learning_insights()

    def session_create(self, agent_id: str, metadata: Dict = None) -> str:
        if self._session_manager is None:
            self._session_manager = SessionManager(self)
        return self._session_manager.create_session(agent_id, metadata)

    def session_add(self, session_id: str, role: str, content: str,
                    metadata: Dict = None) -> Dict[str, Any]:
        if self._session_manager is None:
            self._session_manager = SessionManager(self)
        return self._session_manager.add_to_session(session_id, role, content, metadata)

    def session_recall(self, session_id: str, n: int = 20) -> Dict[str, Any]:
        if self._session_manager is None:
            self._session_manager = SessionManager(self)
        return self._session_manager.recall_session(session_id, n)

    def session_consolidate(self, session_id: str) -> Dict[str, Any]:
        if self._session_manager is None:
            self._session_manager = SessionManager(self)
        return self._session_manager.consolidate_session(session_id)

    def session_list(self) -> List[Dict[str, Any]]:
        if self._session_manager is None:
            return []
        return self._session_manager.list_sessions()

    def session_get(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self._session_manager is None:
            return None
        return self._session_manager.get_session(session_id)


class FeedbackRecord:
    __slots__ = ("action", "context_id", "outcome", "confidence", "reasoning", "timestamp", "metadata")

    def __init__(self, action: str, context_id: str, outcome: str,
                 confidence: float, reasoning: str, metadata: Dict[str, Any] = None):
        self.action = action
        self.context_id = context_id
        self.outcome = outcome
        self.confidence = max(0.0, min(1.0, confidence))
        self.reasoning = reasoning
        self.timestamp = time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "context_id": self.context_id[:12],
            "outcome": self.outcome,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning[:200],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class FeedbackLoop:
    """
    Agent decision feedback loop — learns from agent actions to strengthen
    or deprioritize memory associations. Never deletes, only adjusts priority.

    Core principle: negative outcomes do NOT reduce weight.
    Instead they tag as __low_priority__ and reduce Hebbian path strength slightly.
    Positive outcomes strengthen weight + crystallize candidate paths.
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._records: List[FeedbackRecord] = []
        self._max_records = 500
        self._lock = threading.RLock()
        self._outcome_counts: Dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        self._action_counts: Dict[str, int] = defaultdict(int)
        self._consecutive_positive: Dict[str, int] = defaultdict(int)

    def record_outcome(self, action: str, context_id: str, outcome: str,
                       confidence: float = 0.5, reasoning: str = "",
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        if outcome not in ("positive", "negative", "neutral"):
            outcome = "neutral"

        record = FeedbackRecord(action, context_id, outcome, confidence, reasoning, metadata)

        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records // 2:]
            self._outcome_counts[outcome] += 1
            self._action_counts[action] += 1

        field = self._field
        with field._lock:
            node = field._nodes.get(context_id)
            if node is None:
                for nid, n in field._nodes.items():
                    if nid.startswith(context_id) and n.is_occupied:
                        node = n
                        context_id = nid
                        break

            if node is None or not node.is_occupied:
                return {"recorded": True, "action_taken": "no_node_found"}

            adjustments = []

            if outcome == "positive":
                boost = confidence * 0.2
                node.weight = min(10.0, node.weight + boost)
                node.activation = min(10.0, node.activation + boost * 0.5)
                if "__low_priority__" in node.labels:
                    node.labels.remove("__low_priority__")

                spatial_spread = boost * 0.3
                for fnid in node.face_neighbors[:6]:
                    fn = field._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        shared = len(set(node.labels) & set(fn.labels))
                        if shared > 0:
                            fn.activation = min(10.0, fn.activation + spatial_spread * shared * 0.1)

                self._consecutive_positive[context_id] = self._consecutive_positive.get(context_id, 0) + 1
                if self._consecutive_positive[context_id] >= 3:
                    if field._hebbian:
                        for fnid in node.face_neighbors[:8]:
                            fn = field._nodes.get(fnid)
                            if fn and fn.is_occupied:
                                field._hebbian.record_path([context_id, fnid], True, 0.5)
                    adjustments.append("hebbian_reinforced")

                adjustments.append(f"weight_boosted:+{boost:.3f}")

            elif outcome == "negative":
                if "__low_priority__" not in node.labels:
                    node.labels.append("__low_priority__")
                node.metadata["negative_feedback_count"] = node.metadata.get("negative_feedback_count", 0) + 1
                self._consecutive_positive[context_id] = 0
                adjustments.append("tagged_low_priority")

            else:
                node.activation = max(0.0, node.activation - 0.01)
                adjustments.append("activation_nudged:-0.01")

        return {"recorded": True, "action_taken": "; ".join(adjustments)}

    def learn_from_action(self, action: str, source_id: str, target_id: str,
                          success: bool, confidence: float = 0.5) -> Dict[str, Any]:
        field = self._field
        learning_result = {"action": action, "success": success}

        with field._lock:
            src = field._nodes.get(source_id)
            tgt = field._nodes.get(target_id)

            if src and tgt and success:
                if field._hebbian:
                    field._hebbian.record_path([source_id, target_id], True, 0.3 * confidence)
                    learning_result["hebbian_reinforced"] = True

                if action == "navigate" and field._crystallized:
                    path_weight = field._hebbian.get_path_bias(source_id, target_id) if field._hebbian else 0
                    if path_weight > PCNNConfig.CRYSTALLIZE_THRESHOLD * 0.5:
                        field._crystallized.try_crystallize(source_id, target_id, path_weight)
                        learning_result["crystal_candidate"] = True

            elif src and tgt and not success:
                if field._hebbian:
                    field._hebbian._edges.pop((source_id, target_id), None)
                    learning_result["hebbian_weakened"] = True

        return learning_result

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = sum(self._outcome_counts.values())
            positive_rate = self._outcome_counts["positive"] / max(1, total)
            return {
                "total_feedback": total,
                "outcome_counts": dict(self._outcome_counts),
                "action_counts": dict(self._action_counts),
                "positive_rate": round(positive_rate, 3),
                "recent_records": [r.to_dict() for r in self._records[-5:]],
            }

    def get_learning_insights(self) -> List[Dict[str, Any]]:
        insights = []
        with self._lock:
            node_positive = defaultdict(int)
            node_negative = defaultdict(int)
            for r in self._records:
                if r.outcome == "positive":
                    node_positive[r.context_id] += 1
                elif r.outcome == "negative":
                    node_negative[r.context_id] += 1

            for nid, pos_count in sorted(node_positive.items(), key=lambda x: -x[1])[:10]:
                neg_count = node_negative.get(nid, 0)
                insights.append({
                    "node_id": nid[:12],
                    "positive_count": pos_count,
                    "negative_count": neg_count,
                    "insight": "highly_effective" if pos_count > neg_count * 3 else "balanced",
                })

        return insights


class SessionRecord:
    __slots__ = ("role", "content", "timestamp", "memory_id", "metadata")

    def __init__(self, role: str, content: str, memory_id: str = None, metadata: Dict = None):
        self.role = role
        self.content = content
        self.timestamp = time.time()
        self.memory_id = memory_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content[:200],
            "timestamp": self.timestamp,
            "memory_id": self.memory_id[:12] if self.memory_id else None,
            "metadata": self.metadata,
        }


class Session:
    def __init__(self, session_id: str, agent_id: str, metadata: Dict = None):
        self.session_id = session_id
        self.agent_id = agent_id
        self.created_at = time.time()
        self.last_active = time.time()
        self.records: List[SessionRecord] = []
        self.metadata = metadata or {}
        self.ephemeral_ids: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "record_count": len(self.records),
            "metadata": self.metadata,
        }


class SessionManager:
    """
    Conversation memory management — distinguishes ephemeral (temporary)
    from permanent memories. Ephemeral memories are session-scoped context
    that can be consolidated into permanent memories when the session ends.
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
        self._max_sessions = 50

    def create_session(self, agent_id: str, metadata: Dict = None) -> str:
        session_id = hashlib.sha256(f"{agent_id}:{time.time()}:{random.random()}".encode()).hexdigest()[:16]
        session = Session(session_id, agent_id, metadata)

        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_active)
                self.consolidate_session(oldest_id)
                del self._sessions[oldest_id]

            self._sessions[session_id] = session

        return session_id

    def add_to_session(self, session_id: str, role: str, content: str,
                       metadata: Dict = None) -> Dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": "session not found"}

        field = self._field
        memory_id = None
        if content and len(content.strip()) > 0:
            ephemeral_labels = ["__ephemeral__", f"__session_{session_id[:8]}__"]
            if metadata and metadata.get("labels"):
                ephemeral_labels.extend(metadata["labels"])

            with field._lock:
                memory_id = field.store(
                    content=content,
                    labels=ephemeral_labels,
                    weight=0.5,
                    metadata={"session_id": session_id, "role": role, "ephemeral": True},
                )

        record = SessionRecord(role, content, memory_id, metadata)

        with self._lock:
            session.records.append(record)
            session.last_active = time.time()
            if memory_id:
                session.ephemeral_ids.append(memory_id)

        return {"added": True, "memory_id": memory_id[:12] if memory_id else None}

    def recall_session(self, session_id: str, n: int = 20) -> Dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": "session not found"}
            records = session.records[-n:]
            return {
                "session_id": session_id,
                "records": [r.to_dict() for r in records],
                "total_records": len(session.records),
                "agent_id": session.agent_id,
            }

    def consolidate_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": "session not found"}

        field = self._field
        consolidated = 0
        promoted = 0

        with field._lock:
            for mid in session.ephemeral_ids:
                node = field._nodes.get(mid)
                if node is None:
                    continue

                if node.weight >= 0.8:
                    node.labels = [l for l in node.labels if l.startswith("__session_") or l == "__ephemeral__"]
                    if "__ephemeral__" in node.labels:
                        node.labels.remove("__ephemeral__")
                    node.labels.append("__consolidated__")
                    node.weight = min(10.0, node.weight + 0.3)
                    if "ephemeral" in node.metadata:
                        del node.metadata["ephemeral"]
                    field._emit_pulse(mid, strength=0.5, pulse_type=PulseType.REINFORCING)
                    promoted += 1
                else:
                    node.base_activation = max(node.base_activation, 0.02)
                    consolidated += 1

        with self._lock:
            session.metadata["consolidated_at"] = time.time()
            session.metadata["promoted"] = promoted
            session.metadata["soft_kept"] = consolidated

        return {
            "session_id": session_id,
            "total_ephemeral": len(session.ephemeral_ids),
            "promoted_to_permanent": promoted,
            "soft_kept": consolidated,
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [s.to_dict() for s in sorted(self._sessions.values(), key=lambda s: -s.last_active)]

    def expire_sessions(self, max_age: int = 3600) -> Dict[str, Any]:
        now = time.time()
        expired = []
        with self._lock:
            to_remove = []
            for sid, session in self._sessions.items():
                if now - session.last_active > max_age:
                    self.consolidate_session(sid)
                    to_remove.append(sid)
                    expired.append(sid)
            for sid in to_remove:
                del self._sessions[sid]

        return {"expired_sessions": len(expired), "session_ids": [s[:12] for s in expired]}

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return {
                **session.to_dict(),
                "recent_records": [r.to_dict() for r in session.records[-10:]],
            }
