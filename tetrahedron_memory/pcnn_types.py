from __future__ import annotations

import enum
import time
from typing import Optional


class PulseType(enum.Enum):
    EXPLORATORY = "exploratory"
    REINFORCING = "reinforcing"
    TENSION_SENSING = "tension_sensing"
    SELF_CHECK = "self_check"
    CASCADE = "cascade"
    STRUCTURE = "structure"
    DEEP_REINFORCEMENT = "deep_reinforcement"
    DREAM_PULSE = "dream_pulse"
    FEEDBACK_PULSE = "feedback_pulse"


class PCNNConfig:
    __slots__ = ()

    def __setattr__(self, name, value):
        raise TypeError(f"PCNNConfig is immutable: cannot set {name}")
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
    ALPHA_THRESHOLD = 0.12

    BETA = 0.35

    V_F = 1.0
    V_L = 0.5
    V_THETA = 4.5

    FACE_DECAY = 0.72
    EDGE_DECAY_FACTOR = 0.50
    VERTEX_DECAY_FACTOR = 0.25

    MAX_HOPS_EXPLORATORY = 12
    MAX_HOPS_REINFORCING = 8
    MAX_HOPS_TENSION = 5

    BRIDGE_THRESHOLD = 0.30
    MIN_BRIDGE_SOURCES = 2

    BASE_PULSE_INTERVAL = 0.50
    MIN_PULSE_INTERVAL = 0.20
    MAX_PULSE_INTERVAL = 2.00

    EXPLORATORY_STRENGTH_RANGE = (0.08, 0.20)
    REINFORCING_STRENGTH_RANGE = (0.25, 0.50)
    TENSION_STRENGTH_RANGE = (0.40, 0.70)

    PULSE_TYPE_PROBABILITIES = {
        PulseType.EXPLORATORY: 0.22,
        PulseType.REINFORCING: 0.22,
        PulseType.CASCADE: 0.18,
        PulseType.TENSION_SENSING: 0.10,
        PulseType.SELF_CHECK: 0.07,
        PulseType.STRUCTURE: 0.06,
        PulseType.DEEP_REINFORCEMENT: 0.15,
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

    CASCADE_BRANCHING_FACTOR = 5
    CASCADE_BRANCHING_DECAY = 0.65
    CASCADE_MAX_DEPTH = 8
    CASCADE_ENERGY_CONSERVATION = 0.95
    CASCADE_STRENGTH_RANGE = (0.30, 0.60)
    CASCADE_MAX_HOPS = 8

    STRUCTURE_MAX_HOPS = 3
    STRUCTURE_STRENGTH = 0.40
    STRUCTURE_INTEGRITY_INTERVAL = 300

    CRYSTALLIZE_THRESHOLD = 1.8
    CRYSTALLIZE_MAX_PATHS = 5000
    CRYSTAL_PULSE_BOOST = 1.8
    CRYSTAL_WEIGHT_FLOOR = 2.0
    SUPER_CRYSTAL_THRESHOLD = 4.0
    SUPER_CRYSTAL_BOOST = 2.5

    DEEP_REINFORCEMENT_STRENGTH_RANGE = (0.50, 0.80)
    DEEP_REINFORCEMENT_MAX_HOPS = 10
    DEEP_REINFORCEMENT_DECAY = 0.0

    CONVERGENCE_CHECK_CYCLES = 60
    GLOBAL_DECAY_CYCLES = 120

    SELF_ORGANIZE_INTERVAL = 60
    CLUSTER_MIN_SIZE = 4
    CLUSTER_LABEL_OVERLAP = 0.4
    CLUSTER_MAX_LABELS = 8
    CLUSTER_QUALITY_PROMOTION_THRESHOLD = 0.8
    ENTROPY_IDEAL_RATIO = 0.15
    ENTROPY_BOOST_FACTOR = 0.2
    ENTROPY_SUPPRESS_FACTOR = 0.1
    ENTROPY_HIGH_THRESHOLD = 0.7
    CONSOLIDATION_MIN_SIMILARITY = 0.75
    CONSOLIDATION_MAX_PER_CYCLE = 5
    CONSOLIDATION_WEIGHT_TRANSFER = 0.6
    SHORTCUT_MAX_DISTANCE = 6
    SHORTCUT_MIN_LABEL_OVERLAP = 2
    SHORTCUT_MAX_PER_CYCLE = 5
    SHORTCUT_VIRTUAL_STRENGTH = 0.8

    TETRA_QUALITY_THRESHOLD = 0.3
    TETRA_IDEAL_VOLUME_FACTOR = 0.1178
    TETRA_MAX_CELLS_PER_ANALYSIS = 500
    TETRA_DENSITY_PENALTY = 0.5

    DREAM_CYCLE_INTERVAL = 180
    DREAM_MAX_RECOMBINATIONS = 20
    DREAM_MIN_SOURCE_WEIGHT = 1.0
    DREAM_INSIGHT_WEIGHT = 1.5
    DREAM_CROSS_DOMAIN_BONUS = 2.0
    DREAM_QUALITY_THRESHOLD = 0.65
    DREAM_MIN_DOMAIN_DEPTH = 0.3
    DREAM_SOURCE_MIN_WEIGHT = 1.5
    DREAM_SOURCE_MIN_ACTIVATION = 0.5
    DREAM_INSIGHT_MIN_CREATIVITY = 0.45

    AGENT_CONTEXT_MAX_MEMORIES = 15
    AGENT_REASONING_MAX_HOPS = 5
    AGENT_SUGGESTION_TOP_N = 5

    MAX_PULSE_ACCUMULATOR = 5.0
    MAX_INTERNAL_ACTIVITY = 50.0
    MAX_FEEDING = 20.0
    MAX_LINKING = 10.0

    ADAPTIVE_MIN_INTERVAL = 0.1
    ADAPTIVE_MAX_INTERVAL = 3.0
    ADAPTIVE_LOAD_THRESHOLD = 0.8

    MINIMUM_VIABLE_PULSE = {
        "exploratory": 0.15,
        "reinforcing": 0.20,
        "deep_reinforcement": 0.10,
        "cascade": 0.15,
        "self_check": 0.10,
        "tension_sensing": 0.05,
        "structure": 0.05,
        "dream_pulse": 0.10,
        "feedback_pulse": 0.10,
    }

    PULSE_BUDGET_PER_CYCLE = 50
    CASCADE_ENERGY_BUDGET = 0.3

    QUERY_WEIGHTS = {
        "text": 0.11,
        "trigram": 0.10,
        "label": 0.12,
        "activation": 0.08,
        "weight": 0.05,
        "hebbian": 0.06,
        "crystal": 0.06,
        "pulse": 0.02,
        "spatial_quality": 0.05,
        "geometric_quality": 0.04,
        "cell_effective": 0.03,
        "neighbor_density": 0.03,
        "geo_topo_divergence": 0.02,
        "bcc_coherence": 0.02,
        "autocorr": 0.01,
        "dream": 0.01,
        "recency": 0.06,
        "temporal": 0.04,
        "spatial_proximity": 0.05,
        "plane_priority": 0.04,
    }

    ATTENTION_MULTIPLIER = 0.5
    ATTENTION_ADDITIVE = 0.15

    PULSE_CYCLE_INTERVALS = {
        "convergence_check": 60,
        "global_decay": 120,
        "pcnn_step": 30,
        "crystal_maintenance": 60,
        "lattice_check": 600,
        "self_organize": 60,
        "dream": 300,
        "cell_density": 120,
        "reflection": 150,
        "attention": 50,
        "resonance": 300,
        "emergence": 500,
        "feedback_evolve": 200,
        "dark_plane_flow": 60,
    }

    DARK_PLANE_CONFIG = {
        "surface": {"pulse_weight": 3.0, "query_priority": 1.0, "dream_focus": 0.3},
        "shallow": {"pulse_weight": 2.0, "query_priority": 0.8, "dream_focus": 1.0},
        "deep": {"pulse_weight": 0.8, "query_priority": 0.5, "dream_focus": 0.6},
        "abyss": {"pulse_weight": 0.2, "query_priority": 0.2, "dream_focus": 0.1},
    }

    LIFECYCLE_THRESHOLDS = {
        "fresh_seconds": 3600,
        "consolidating_seconds": 86400,
        "crystallized_reinforcement": 5,
        "ancient_days": 7,
    }

    ATTENTION_CONFIG = {
        "decay_rate": 0.03,
        "diffusion_rate": 0.3,
        "max_foci": 3,
    }

    LATTICE_EXPANSION = {
        "occupancy_threshold": 0.85,
        "density_check_threshold": 0.60,
        "dense_ratio_threshold": 0.3,
        "dense_neighbor_count": 5,
        "sample_size": 200,
    }


class NeuralPulse:
    __slots__ = (
        "source_id", "strength", "hops", "path", "path_set",
        "direction", "birth_time", "max_hops", "pulse_type",
        "bias_fn", "cascade_depth", "cascade_parent_id",
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
        self.path_set = {source_id}
        self.direction = "face"
        self.birth_time = time.time()
        self.max_hops = max_hops
        self.pulse_type = pulse_type
        self.bias_fn = bias_fn
        self.cascade_depth = cascade_depth
        self.cascade_parent_id = cascade_parent_id

    def propagate(self, decay: float = 0.7) -> float:
        self.hops += 1
        if self.pulse_type != PulseType.DEEP_REINFORCEMENT:
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
        child.path = self.path[-8:]
        child.path_set = set(child.path)
        child.direction = self.direction
        return child
