from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

from .pcnn_types import PCNNConfig


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

    def __init__(self, max_crystals: int = 200, weight_floor: float = 2.0, auto_expand: bool = True, expand_factor: float = 1.5):
        self._crystals: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._max_crystals = max_crystals
        self._weight_floor = weight_floor
        self._creation_count = 0
        self._pulse_transmissions = 0
        self._auto_expand = auto_expand
        self._expand_factor = expand_factor

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
            if self._auto_expand:
                self._expand_capacity()
            else:
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
        cw = crystal.get("crystal_weight", 2.0)
        hw = crystal.get("hebbian_weight", 2.0)
        if hw >= PCNNConfig.SUPER_CRYSTAL_THRESHOLD:
            return 1.0 + 1.5 * min(cw / 10.0, 1.0)
        return 1.0 + 0.8 * min(cw / 10.0, 1.0)

    def get_crystal_path(self, node_a: str, node_b: str) -> float:
        return self._crystals.get(self._make_key(node_a, node_b), {}).get("crystal_weight", 0.0)

    def _evict_weakest(self):
        if not self._crystals:
            return
        weakest_key = min(self._crystals, key=lambda k: self._crystals[k]["crystal_weight"])
        del self._crystals[weakest_key]

    def _expand_capacity(self):
        """Auto-expand crystal capacity by expand_factor when full."""
        old_cap = self._max_crystals
        self._max_crystals = int(self._max_crystals * self._expand_factor)
        print(f"[CrystallizedPathway] Auto-expanded: {old_cap} → {self._max_crystals}")

    def _make_key(self, a: str, b: str) -> Tuple[str, str]:
        return (min(a, b), max(a, b))

    def scan_and_crystallize(self, hebbian_edges: Dict[Tuple[str, str], float]):
        new_count = 0
        for (a, b), weight in hebbian_edges.items():
            if weight >= PCNNConfig.CRYSTALLIZE_THRESHOLD:
                if self.try_crystallize(a, b, weight):
                    new_count += 1
        return new_count

    def health_check(self, occupied_set: set) -> int:
        dead_keys = []
        for key, crystal in self._crystals.items():
            a, b = crystal["nodes"]
            if a not in occupied_set and b not in occupied_set:
                dead_keys.append(key)
        for key in dead_keys:
            del self._crystals[key]
        return len(dead_keys)

    def stats(self) -> Dict[str, Any]:
        if not self._crystals:
            return {
                "total_crystals": 0,
                "super_crystals": 0,
                "total_created": self._creation_count,
                "total_transmissions": self._pulse_transmissions,
                "avg_crystal_weight": 0.0,
                "max_crystal_weight": 0.0,
            }
        weights = [c["crystal_weight"] for c in self._crystals.values()]
        super_count = sum(1 for c in self._crystals.values() if c.get("hebbian_weight", 0) >= PCNNConfig.SUPER_CRYSTAL_THRESHOLD)
        return {
            "total_crystals": len(self._crystals),
            "super_crystals": super_count,
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
