"""
Void Channel -- Topological Handle Attachment for Cross-Domain Associations

When cross-domain associations exceed a threshold, void channels are created
as topological handles in the dark plane substrate. Channels have three
dimension levels:

  dim=1: Simple bridge -- bidirectional energy coupling
  dim=2: Higher handle -- multi-body activation (triggered by H4 phase transition)
  dim=3: Control channel -- full Hebbian enhancement + regulation signal propagation

Channel lifecycle:
  Creation: triggered by DreamEngine cross-domain synthesis, SelfOrganizeEngine
            cross-cluster connections, or phase transitions (H4/H5/H6)
  Effects:  applied every flow cycle -- energy coupling, activation boost
  Decay:    inactive channels weaken over time
  Upgrade:  phase transitions can increase channel dimension
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .dark_plane_substrate import DarkPlaneSubstrate
    from .honeycomb_neural_field import HoneycombNeuralField


@dataclass
class VoidChannelRecord:
    node_a: str
    node_b: str
    strength: float
    creation_time: float
    association_score: float
    label_distance: float
    dimension: int
    energy_coupling: float
    is_active: bool = True
    created_by_phase: int = 0


class VoidChannel:
    __slots__ = (
        "_substrate",
        "_field",
        "_channels",
        "_max_channels",
        "_association_threshold",
    )

    def __init__(self, substrate: DarkPlaneSubstrate, field: HoneycombNeuralField):
        self._substrate = substrate
        self._field = field
        self._channels: List[VoidChannelRecord] = []
        self._max_channels: int = 200
        self._association_threshold: float = 0.65

    def try_create_channel(
        self,
        node_a: str,
        node_b: str,
        association_score: float,
        phase_level: int = 0,
    ) -> Optional[VoidChannelRecord]:
        if association_score < self._association_threshold:
            return None

        if node_a not in self._field._occupied_ids or node_b not in self._field._occupied_ids:
            return None

        node_obj_a = self._field._nodes.get(node_a)
        node_obj_b = self._field._nodes.get(node_b)
        if node_obj_a is None or node_obj_b is None:
            return None
        if not node_obj_a.is_occupied or not node_obj_b.is_occupied:
            return None

        labels_a = set(node_obj_a.labels) - {"__duplicate_of__"}
        labels_b = set(node_obj_b.labels) - {"__duplicate_of__"}
        if not labels_a or not labels_b:
            return None

        intersection = labels_a & labels_b
        union = labels_a | labels_b
        label_distance = 1.0 - len(intersection) / max(len(union), 1)
        if label_distance < 0.5:
            return None

        active_count = sum(1 for ch in self._channels if ch.is_active)
        if active_count >= self._max_channels:
            weakest = None
            weakest_idx = -1
            for idx, ch in enumerate(self._channels):
                if ch.is_active and (weakest is None or ch.strength < weakest.strength):
                    weakest = ch
                    weakest_idx = idx
            if weakest is not None:
                weakest.is_active = False

        dimension = 1
        strength_mult = 1.0
        if phase_level >= 6:
            dimension = 3
            strength_mult = 2.0
        elif phase_level >= 5:
            dimension = 3
            strength_mult = 1.5
        elif phase_level >= 4:
            dimension = 2
            strength_mult = 1.5

        channel = VoidChannelRecord(
            node_a=node_a,
            node_b=node_b,
            strength=association_score * strength_mult,
            creation_time=time.time(),
            association_score=association_score,
            label_distance=label_distance,
            dimension=dimension,
            energy_coupling=0.0,
            is_active=True,
            created_by_phase=phase_level,
        )
        self._channels.append(channel)
        return channel

    def apply_channel_effects(self):
        e_void = self._substrate._state.void_energy
        nodes = self._field._nodes

        for channel in self._channels:
            if not channel.is_active:
                continue

            node_a = nodes.get(channel.node_a)
            node_b = nodes.get(channel.node_b)
            if node_a is None or node_b is None:
                continue
            if not node_a.is_occupied or not node_b.is_occupied:
                continue

            coupling = 0.15 * e_void * channel.strength
            channel.energy_coupling = coupling

            boost = coupling * 0.5
            node_a.activation = min(node_a.activation + boost, node_a.weight * 2.0)
            node_b.activation = min(node_b.activation + boost, node_b.weight * 2.0)

            if channel.dimension >= 2:
                neighbor_boost = boost * 0.3
                for nnid in node_a.face_neighbors[:8]:
                    nn = nodes.get(nnid)
                    if nn and nn.is_occupied:
                        nn.activation = min(nn.activation + neighbor_boost, nn.weight * 2.0)
                for nnid in node_b.face_neighbors[:8]:
                    nn = nodes.get(nnid)
                    if nn and nn.is_occupied:
                        nn.activation = min(nn.activation + neighbor_boost, nn.weight * 2.0)

            if channel.dimension >= 3:
                hebbian = self._field._hebbian
                if hebbian:
                    key = (min(channel.node_a, channel.node_b), max(channel.node_a, channel.node_b))
                    current_w = hebbian._edges.get(key, 0.0)
                    hebbian._edges[key] = current_w + coupling * 0.1

    def cascade_upgrade(self):
        for channel in self._channels:
            if channel.is_active:
                channel.dimension = min(channel.dimension + 1, 3)
                channel.strength = min(channel.strength * 2.0, 2.0)

    def decay_inactive(self, pulse_log=None):
        recent_active = set()
        occupied = self._field._occupied_ids

        if pulse_log:
            for entry in pulse_log[-200:]:
                for nid_key in ("source", "target"):
                    nid = entry.get(nid_key)
                    if nid:
                        recent_active.add(nid)

        for channel in self._channels:
            if not channel.is_active:
                continue

            a_hit = channel.node_a in recent_active
            b_hit = channel.node_b in recent_active

            node_a = self._field._nodes.get(channel.node_a)
            node_b = self._field._nodes.get(channel.node_b)
            if node_a and node_a.access_count > 0:
                a_hit = True
            if node_b and node_b.access_count > 0:
                b_hit = True

            if not a_hit and not b_hit:
                channel.strength *= 0.95
                if channel.strength < 0.1:
                    channel.is_active = False

    def get_channels_for_node(self, node_id: str) -> List[VoidChannelRecord]:
        return [
            ch for ch in self._channels
            if ch.is_active and (ch.node_a == node_id or ch.node_b == node_id)
        ]

    def get_stats(self) -> dict:
        active = [ch for ch in self._channels if ch.is_active]
        by_dim = {1: 0, 2: 0, 3: 0}
        total_strength = 0.0
        total_coupling = 0.0
        for ch in active:
            by_dim[ch.dimension] = by_dim.get(ch.dimension, 0) + 1
            total_strength += ch.strength
            total_coupling += ch.energy_coupling

        return {
            "total_channels": len(self._channels),
            "active_channels": len(active),
            "by_dimension": by_dim,
            "avg_strength": total_strength / max(len(active), 1),
            "total_energy_coupling": total_coupling,
            "max_channels": self._max_channels,
            "association_threshold": self._association_threshold,
        }

    def get_state(self) -> list:
        return [
            {
                "node_a": ch.node_a,
                "node_b": ch.node_b,
                "strength": ch.strength,
                "creation_time": ch.creation_time,
                "association_score": ch.association_score,
                "label_distance": ch.label_distance,
                "dimension": ch.dimension,
                "energy_coupling": ch.energy_coupling,
                "is_active": ch.is_active,
                "created_by_phase": ch.created_by_phase,
            }
            for ch in self._channels
            if ch.is_active
        ]

    def set_state(self, channels_data: list):
        self._channels.clear()
        for cd in channels_data:
            self._channels.append(
                VoidChannelRecord(
                    node_a=cd.get("node_a", ""),
                    node_b=cd.get("node_b", ""),
                    strength=float(cd.get("strength", 0.5)),
                    creation_time=float(cd.get("creation_time", 0)),
                    association_score=float(cd.get("association_score", 0.5)),
                    label_distance=float(cd.get("label_distance", 0.5)),
                    dimension=int(cd.get("dimension", 1)),
                    energy_coupling=float(cd.get("energy_coupling", 0)),
                    is_active=bool(cd.get("is_active", True)),
                    created_by_phase=int(cd.get("created_by_phase", 0)),
                )
            )
