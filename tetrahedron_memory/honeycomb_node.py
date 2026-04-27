from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from .pcnn_types import PCNNConfig


class HoneycombNode:
    __slots__ = (
        "id", "position", "face_neighbors", "edge_neighbors", "vertex_neighbors",
        "content", "labels", "weight", "activation", "base_activation",
        "last_pulse_time", "pulse_accumulator", "creation_time",
        "metadata", "access_count", "decay_rate",
        "feeding", "linking", "internal_activity", "threshold", "fired",
        "crystal_channels",
        "reinforcement_count", "last_dream_cycle", "domain_affinity",
        "pulse_resonance", "domain_signature", "bridge_score",
        "hibernated", "last_pulse_direction",
        "_frozen_metadata",
        "_cached_tokens", "_cached_trigrams", "_label_set",
    )

    def __init__(self, id: str, position: np.ndarray):
        self.id = id
        self.position = position
        self.face_neighbors: List[str] = []
        self.edge_neighbors: List[str] = []
        self.vertex_neighbors: List[str] = []
        self.content: str | None = None
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

        self.reinforcement_count: int = 0
        self.last_dream_cycle: float = 0.0
        self.domain_affinity: Dict[str, float] = {}

        self.pulse_resonance: float = 0.0
        self.domain_signature: int = 0
        self.bridge_score: float = 0.0
        self.hibernated: bool = False
        self.last_pulse_direction: str = ""
        self._frozen_metadata: Dict[str, Any] | None = None
        self._cached_tokens: set | None = None
        self._cached_trigrams: set | None = None
        self._label_set: set | None = None

    @property
    def is_occupied(self) -> bool:
        return self.content is not None

    @property
    def is_hibernated(self) -> bool:
        return self.hibernated

    @property
    def topological_signature(self) -> int:
        sig = 0
        for i, nid in enumerate(self.face_neighbors[:8]):
            sig = (sig * 31) ^ hash(nid)
        for i, nid in enumerate(self.edge_neighbors[:6]):
            sig = (sig * 17) ^ hash(nid)
        return sig

    def touch(self):
        self.last_pulse_time = time.time()
        self.access_count += 1

    def decay(self, dt: float):
        if self.hibernated:
            return
        if self.is_occupied:
            rate = self.decay_rate / max(self.weight, 0.5)
            if self.weight < 0.3:
                rate *= 2.0
            if self.crystal_channels:
                rate *= 0.5
            rate *= self.lifecycle_decay_multiplier()
            self.activation = max(self.base_activation, self.activation - rate * dt)
        else:
            self.pulse_accumulator *= 0.95

    def reinforce(self, amount: float):
        boost = amount * max(self.weight, 0.5) * 0.3
        self.activation = min(10.0, self.activation + amount + boost)
        self.reinforcement_count += 1

    def update_domain_affinity(self):
        if not self.labels:
            return
        for label in self.labels:
            if label.startswith("__"):
                continue
            self.domain_affinity[label] = self.domain_affinity.get(label, 0.0) + self.weight * 0.1
        if len(self.domain_affinity) > 20:
            sorted_domains = sorted(self.domain_affinity.items(), key=lambda x: -x[1])
            self.domain_affinity = dict(sorted_domains[:15])

    def pcnn_step(self, neighbor_outputs: List[Tuple[str, float, str]]):
        """
        One PCNN timestep for this node with pulse resonance optimization.

        Parameters
        ----------
        neighbor_outputs : list of (neighbor_id, output_strength, connection_type)
            "face", "edge", or "vertex"
        """
        cfg = PCNNConfig

        self.pulse_resonance *= 0.95

        s_input = self.activation if self.is_occupied else self.pulse_accumulator
        s_input = max(0.0, min(cfg.MAX_INTERNAL_ACTIVITY, s_input))

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

        if not math.isfinite(self.internal_activity):
            self.internal_activity = 0.0
            self.feeding = 0.0
            self.linking = 0.0

        self.pulse_resonance += self.internal_activity * 0.1

        adaptive_threshold = cfg.V_THETA * (1.0 + 0.1 * self.reinforcement_count / max(self.reinforcement_count, 5))

        self.fired = self.internal_activity > adaptive_threshold and self.pulse_resonance > 0.3

        if self.fired:
            self.threshold = cfg.ALPHA_THRESHOLD * self.threshold + cfg.V_THETA
            self.pulse_accumulator = min(cfg.MAX_PULSE_ACCUMULATOR, self.pulse_accumulator + self.internal_activity * 0.02)
        else:
            self.threshold = cfg.ALPHA_THRESHOLD * self.threshold

    def compute_domain_signature(self) -> int:
        sorted_labels = sorted(self.labels)
        self.domain_signature = hash(tuple(sorted_labels))
        return self.domain_signature

    def compute_bridge_score(self, neighbors: List["HoneycombNode"]) -> float:
        if not self.is_occupied or not neighbors:
            self.bridge_score = 0.0
            return 0.0

        face_set = set(self.face_neighbors)
        edge_set = set(self.edge_neighbors)

        unique_sigs: set[int] = set()
        connection_strength = 0.0
        for nb in neighbors:
            if not nb.is_occupied:
                continue
            unique_sigs.add(nb.domain_signature)
            if nb.id in face_set:
                connection_strength += 1.0
            elif nb.id in edge_set:
                connection_strength += 0.5
            else:
                connection_strength += 0.25

        if not unique_sigs:
            self.bridge_score = 0.0
            return 0.0

        score = len(unique_sigs) * connection_strength
        self.bridge_score = round(min(1.0, max(0.0, score)), 4)
        return self.bridge_score

    def hibernate(self):
        if self.hibernated:
            return
        self._frozen_metadata = {
            "content": self.content,
            "labels": list(self.labels),
            "weight": self.weight,
            "activation": self.activation,
            "metadata": dict(self.metadata),
            "domain_affinity": dict(self.domain_affinity),
            "reinforcement_count": self.reinforcement_count,
            "crystal_channels": dict(self.crystal_channels),
            "pulse_resonance": self.pulse_resonance,
            "domain_signature": self.domain_signature,
            "bridge_score": self.bridge_score,
        }
        self.content = None
        self.labels = []
        self.hibernated = True

    def wake_from_hibernation(self, content: str, labels: List[str], weight: float):
        if not self.hibernated or self._frozen_metadata is None:
            return
        self.content = content
        self.labels = labels
        self.weight = weight
        self.activation = self._frozen_metadata.get("activation", 0.0)
        self.metadata = self._frozen_metadata.get("metadata", {})
        self.domain_affinity = self._frozen_metadata.get("domain_affinity", {})
        self.reinforcement_count = self._frozen_metadata.get("reinforcement_count", 0)
        self.crystal_channels = self._frozen_metadata.get("crystal_channels", {})
        self.pulse_resonance = self._frozen_metadata.get("pulse_resonance", 0.0)
        self.domain_signature = self._frozen_metadata.get("domain_signature", 0)
        self.bridge_score = self._frozen_metadata.get("bridge_score", 0.0)
        self._frozen_metadata = None
        self.hibernated = False

    def get_efficiency_score(self) -> float:
        if not self.is_occupied:
            return 0.0
        weight_factor = min(1.0, self.weight / 5.0)
        activation_factor = min(1.0, self.activation / 5.0)
        access_efficiency = 1.0 / (1.0 + self.access_count * 0.01)
        efficiency = weight_factor * 0.4 + activation_factor * 0.3 + access_efficiency * 0.3
        return round(min(1.0, max(0.0, efficiency)), 4)

    def lifecycle_stage(self) -> str:
        if not self.is_occupied:
            return "empty"
        age = time.time() - self.creation_time
        if age < 3600:
            return "fresh"
        if self.reinforcement_count >= 5:
            return "crystallized"
        if age < 86400:
            return "consolidating"
        if age > 86400 * 7 and self.reinforcement_count < 2:
            return "ancient"
        return "consolidating"

    def lifecycle_decay_multiplier(self) -> float:
        stage = self.lifecycle_stage()
        if stage == "fresh":
            return 0.5
        if stage == "consolidating":
            return 1.0
        if stage == "crystallized":
            return 0.3
        if stage == "ancient":
            return 2.0
        return 1.0

    def dark_plane(self) -> str:
        if not self.is_occupied:
            return "void"
        if self._frozen_metadata and "dark_plane" in self._frozen_metadata:
            return self._frozen_metadata["dark_plane"]
        stage = self.lifecycle_stage()
        if stage == "fresh":
            return "surface"
        if stage == "consolidating":
            return "shallow"
        if stage == "crystallized":
            return "deep"
        return "abyss"

    def plane_energy_cost(self) -> float:
        plane = self.dark_plane()
        if plane == "surface":
            return 1.0
        if plane == "shallow":
            return 0.6
        if plane == "deep":
            return 0.2
        return 0.1

    def plane_pulse_receptivity(self) -> float:
        plane = self.dark_plane()
        if plane == "surface":
            return 1.5
        if plane == "shallow":
            return 1.0
        if plane == "deep":
            return 0.6
        return 0.2
