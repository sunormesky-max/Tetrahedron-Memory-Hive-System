from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("tetramem.self_regulation")


class SelfRegulationEngine:
    """
    Six-layer self-regulation system inspired by human physiology.

    1. Homeostasis — keep key metrics in healthy ranges by auto-adjusting parameters
    2. Circadian Rhythm — alternating work/consolidation periods
    3. Autonomic Nervous — sympathetic (active) / parasympathetic (rest) mode
    4. Immune — detect anomalies, repair damage, clean dead weight
    5. Endocrine — query success feedback → parameter tuning
    6. Stress Response — overload protection with emergency throttle
    """

    HOMEOSTASIS_TARGETS = {
        "emergence_score": {"low": 0.25, "optimal_low": 0.40, "optimal_high": 0.70, "high": 0.85},
        "bridge_rate": {"low": 0.005, "optimal_low": 0.01, "optimal_high": 0.05, "high": 0.10},
        "crystal_ratio": {"low": 0.1, "optimal_low": 0.3, "optimal_high": 0.7, "high": 0.9},
        "field_entropy": {"low": 0.2, "optimal_low": 0.4, "optimal_high": 0.7, "high": 0.9},
        "activation_mean": {"low": 0.5, "optimal_low": 2.0, "optimal_high": 15.0, "high": 25.0},
    }

    def __init__(self, field):
        self._field = field
        self._last_emergence_score = 0.0
        self._regulation_count = 0
        self._start_time = time.time()

        self._params = {
            "pulse_interval_multiplier": 1.0,
            "attention_decay_rate": 0.05,
            "attention_diffusion_rate": 0.3,
            "hebbian_reinforce_factor": 1.15,
            "pulse_strength_multiplier": 1.0,
            "bridge_threshold_offset": 0.0,
            "cascade_depth_bonus": 0,
            "dream_frequency_multiplier": 1.0,
        }
        self._param_history: List[Dict[str, Any]] = []

        self._circadian_period = 3600
        self._circadian_phase = "work"
        self._circadian_work_ratio = 0.7
        self._circadian_last_switch = time.time()

        self._autonomic_mode = "balanced"
        self._sympathetic_score = 0.5
        self._parasympathetic_score = 0.5

        self._immune_log: List[Dict[str, Any]] = []
        self._immune_total_repairs = 0
        self._immune_scan_interval = 600
        self._immune_last_scan = 0.0

        self._endocrine_hormones = {
            "dopamine": 0.5,
            "cortisol": 0.0,
            "serotonin": 0.5,
            "acetylcholine": 0.5,
        }
        self._query_success_history: List[float] = []
        self._endocrine_max_history = 100

        self._stress_level = 0.0
        self._stress_threshold_warning = 0.6
        self._stress_threshold_critical = 0.85
        self._stress_emergency_mode = False

        self._regulation_log: List[Dict[str, Any]] = []
        self._max_log = 200

    def regulate(self) -> Dict[str, Any]:
        now = time.time()
        actions = []

        self._update_circadian(now)
        circadian_action = self._circadian_phase
        actions.append(f"circadian:{circadian_action}")

        emergence = self._get_emergence_data()
        self._update_endocrine(emergence)
        endocrine_actions = self._apply_endocrine()
        actions.extend(endocrine_actions)

        self._update_stress(emergence)
        stress_actions = self._apply_stress_response()
        actions.extend(stress_actions)

        self._update_autonomic(emergence)
        autonomic_actions = self._apply_autonomic()
        actions.extend(autonomic_actions)

        homeostasis_actions = self._apply_homeostasis(emergence)
        actions.extend(homeostasis_actions)

        if now - self._immune_last_scan > self._immune_scan_interval:
            immune_actions = self._run_immune()
            actions.extend(immune_actions)
            self._immune_last_scan = now

        self._regulation_count += 1
        record = {
            "time": now,
            "emergence_score": emergence.get("overall_score", 0),
            "circadian": self._circadian_phase,
            "autonomic": self._autonomic_mode,
            "stress": round(self._stress_level, 4),
            "hormones": {k: round(v, 4) for k, v in self._endocrine_hormones.items()},
            "params": dict(self._params),
            "actions": actions,
        }
        self._regulation_log.append(record)
        if len(self._regulation_log) > self._max_log:
            self._regulation_log = self._regulation_log[-self._max_log // 2:]

        self._apply_params_to_field()

        return record

    def notify_query_result(self, score: float, k: int, returned: int):
        hit_rate = returned / max(k, 1)
        combined = score * 0.4 + hit_rate * 0.6
        self._query_success_history.append(combined)
        if len(self._query_success_history) > self._endocrine_max_history:
            self._query_success_history = self._query_success_history[-self._endocrine_max_history // 2:]

    def status(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time
        return {
            "regulation_count": self._regulation_count,
            "uptime_seconds": round(uptime, 1),
            "circadian": {
                "phase": self._circadian_phase,
                "period": self._circadian_period,
                "work_ratio": self._circadian_work_ratio,
            },
            "autonomic": {
                "mode": self._autonomic_mode,
                "sympathetic": round(self._sympathetic_score, 4),
                "parasympathetic": round(self._parasympathetic_score, 4),
            },
            "hormones": {k: round(v, 4) for k, v in self._endocrine_hormones.items()},
            "stress": {
                "level": round(self._stress_level, 4),
                "emergency_mode": self._stress_emergency_mode,
                "warning_threshold": self._stress_threshold_warning,
                "critical_threshold": self._stress_threshold_critical,
            },
            "params": {k: round(v, 6) if isinstance(v, float) else v for k, v in self._params.items()},
            "immune": {
                "total_repairs": self._immune_total_repairs,
                "last_scan": round(self._immune_last_scan, 1),
                "scan_interval": self._immune_scan_interval,
            },
            "query_success_avg": round(
                float(np.mean(self._query_success_history[-20:])) if self._query_success_history else 0, 4
            ),
        }

    def get_history(self, n: int = 20) -> List[Dict]:
        return self._regulation_log[-n:]

    def force_mode(self, mode: str):
        if mode in ("sympathetic", "parasympathetic", "balanced"):
            self._autonomic_mode = mode
        if mode in ("work", "consolidation"):
            self._circadian_phase = mode
            self._circadian_last_switch = time.time()

    # ── Layer 1: Homeostasis ──────────────────────────────────

    def _apply_homeostasis(self, emergence: Dict) -> List[str]:
        actions = []
        score = emergence.get("overall_score", 0)
        self._last_emergence_score = score

        bridges = emergence.get("bridges", {})
        crystal = emergence.get("crystal", {})
        phase = emergence.get("phase", {})

        bridge_rate = bridges.get("creation_rate", 0)
        crystal_ratio = crystal.get("crystal_ratio", 0)
        entropy = phase.get("field_entropy", 1.0)
        activation = crystal.get("avg_activation", 0)

        if bridge_rate < 0.005:
            self._adjust("bridge_threshold_offset", -0.05, min_val=-0.2)
            self._adjust("pulse_strength_multiplier", 0.05, min_val=0.5)
            actions.append("homeostasis:boost_bridges")
        elif bridge_rate > 0.10:
            self._adjust("bridge_threshold_offset", 0.03, max_val=0.3)
            actions.append("homeostasis:throttle_bridges")

        if crystal_ratio < 0.1:
            self._adjust("hebbian_reinforce_factor", 0.02, max_val=1.5)
            actions.append("homeostasis:boost_crystallization")
        elif crystal_ratio > 0.9:
            self._adjust("hebbian_reinforce_factor", -0.02, min_val=1.0)
            actions.append("homeostasis:slow_crystallization")

        if entropy < 0.2:
            self._adjust("pulse_interval_multiplier", -0.05, min_val=0.3)
            self._adjust("attention_decay_rate", -0.005, min_val=0.01)
            actions.append("homeostasis:reduce_entropy")
        elif entropy > 0.9:
            self._adjust("pulse_interval_multiplier", 0.05, max_val=2.0)
            self._adjust("attention_decay_rate", 0.005, max_val=0.2)
            actions.append("homeostasis:increase_order")

        if activation > 25.0:
            self._adjust("pulse_strength_multiplier", -0.05, min_val=0.3)
            actions.append("homeostasis:reduce_activation")
        elif activation < 0.5:
            self._adjust("pulse_strength_multiplier", 0.05, max_val=2.0)
            actions.append("homeostasis:boost_activation")

        if score < 0.25:
            self._adjust("dream_frequency_multiplier", 0.1, max_val=3.0)
            self._adjust("attention_diffusion_rate", 0.02, max_val=0.6)
            actions.append("homeostasis:emergency_activate")
        elif score > 0.85:
            self._adjust("pulse_interval_multiplier", 0.02, max_val=2.0)
            actions.append("homeostasis:stabilize_high")

        return actions

    # ── Layer 2: Circadian Rhythm ─────────────────────────────

    def _update_circadian(self, now: float):
        elapsed_in_period = (now - self._circadian_last_switch) % self._circadian_period
        work_duration = self._circadian_period * self._circadian_work_ratio

        new_phase = "work" if elapsed_in_period < work_duration else "consolidation"
        if new_phase != self._circadian_phase:
            self._circadian_phase = new_phase
            self._circadian_last_switch = now
            logger.info("Circadian phase switch → %s", new_phase)

    # ── Layer 3: Autonomic Nervous ────────────────────────────

    def _update_autonomic(self, emergence: Dict):
        score = emergence.get("overall_score", 0)
        stress = self._stress_level
        query_success = float(np.mean(self._query_success_history[-20:])) if self._query_success_history else 0.5

        if stress > 0.6 or score < 0.2:
            self._sympathetic_score = min(1.0, self._sympathetic_score + 0.1)
            self._parasympathetic_score = max(0.0, self._parasympathetic_score - 0.1)
            self._autonomic_mode = "sympathetic"
        elif stress < 0.3 and query_success > 0.6 and score > 0.4:
            self._parasympathetic_score = min(1.0, self._parasympathetic_score + 0.1)
            self._sympathetic_score = max(0.0, self._sympathetic_score - 0.1)
            self._autonomic_mode = "parasympathetic"
        else:
            self._sympathetic_score = 0.5 + (self._sympathetic_score - 0.5) * 0.9
            self._parasympathetic_score = 0.5 + (self._parasympathetic_score - 0.5) * 0.9
            self._autonomic_mode = "balanced"

    def _apply_autonomic(self) -> List[str]:
        actions = []
        if self._autonomic_mode == "sympathetic":
            self._adjust("pulse_interval_multiplier", -0.03, min_val=0.3)
            self._adjust("attention_diffusion_rate", 0.02, max_val=0.6)
            actions.append("autonomic:sympathetic_activate")
        elif self._autonomic_mode == "parasympathetic":
            self._adjust("pulse_interval_multiplier", 0.03, max_val=2.0)
            self._adjust("attention_decay_rate", 0.003, max_val=0.15)
            actions.append("autonomic:parasympathetic_rest")
        return actions

    # ── Layer 4: Immune ───────────────────────────────────────

    def _run_immune(self) -> List[str]:
        actions = []
        field = self._field
        repairs = 0

        try:
            with field._lock:
                dead_edges = 0
                for nid in list(field._hebbian._edges.keys()):
                    a, b = nid
                    if a not in field._occupied_ids or b not in field._occupied_ids:
                        dead_edges += 1
                if dead_edges > 0:
                    field._hebbian._edges = {
                        k: v for k, v in field._hebbian._edges.items()
                        if k[0] in field._occupied_ids and k[1] in field._occupied_ids
                    }
                    repairs += dead_edges
                    actions.append(f"immune:cleaned_{dead_edges}_dead_hebbian_edges")

                orphan_temporal = 0
                for nid in list(field._temporal_edges.keys()):
                    if nid not in field._occupied_ids:
                        orphan_temporal += 1
                        del field._temporal_edges[nid]
                    else:
                        field._temporal_edges[nid] = [
                            (eid, p) for eid, p in field._temporal_edges[nid]
                            if eid in field._occupied_ids
                        ]
                if orphan_temporal > 0:
                    repairs += orphan_temporal
                    actions.append(f"immune:cleaned_{orphan_temporal}_orphan_temporal")

                if field._crystallized and hasattr(field._crystallized, '_crystals'):
                    dead_crystals = 0
                    for key in list(field._crystallized._crystals.keys()):
                        if hasattr(key, '__iter__'):
                            parts = list(key) if not isinstance(key, str) else [key]
                            alive = all(p in field._occupied_ids for p in parts)
                        else:
                            alive = key in field._occupied_ids
                        if not alive:
                            dead_crystals += 1
                            del field._crystallized._crystals[key]
                    if dead_crystals > 0:
                        repairs += dead_crystals
                        actions.append(f"immune:pruned_{dead_crystals}_dead_crystals")

                for nid in list(field._attention_mask.keys()):
                    if nid not in field._occupied_ids:
                        del field._attention_mask[nid]
                        repairs += 1
                if repairs > 0:
                    actions.append(f"immune:cleaned_attention_mask")

                total_nodes = len(field._nodes)
                if total_nodes > 10000:
                    empty_non_frontier = [
                        nid for nid, n in field._nodes.items()
                        if not n.is_occupied and nid not in field._frontier_empty
                    ]
                    if len(empty_non_frontier) > total_nodes * 0.3:
                        n_free = min(len(empty_non_frontier) - int(total_nodes * 0.1), 500)
                        for nid in empty_non_frontier[:n_free]:
                            del field._nodes[nid]
                            repairs += 1
                        actions.append(f"immune:freed_{n_free}_empty_nodes")

        except Exception as e:
            logger.error("Immune scan error: %s", e)

        if repairs > 0:
            self._immune_total_repairs += repairs
            self._immune_log.append({
                "time": time.time(),
                "repairs": repairs,
                "actions": actions,
            })
            if len(self._immune_log) > 50:
                self._immune_log = self._immune_log[-25:]

        return actions

    # ── Layer 5: Endocrine ────────────────────────────────────

    def _update_endocrine(self, emergence: Dict):
        score = emergence.get("overall_score", 0)
        recent_success = (
            float(np.mean(self._query_success_history[-20:]))
            if self._query_success_history else 0.5
        )

        h = self._endocrine_hormones

        h["dopamine"] = 0.7 * h["dopamine"] + 0.3 * recent_success

        cortisol_input = 0.0
        if score < 0.2:
            cortisol_input = 0.8
        elif score < 0.4:
            cortisol_input = 0.4
        elif self._stress_level > 0.6:
            cortisol_input = 0.6
        h["cortisol"] = 0.8 * h["cortisol"] + 0.2 * cortisol_input

        serotonin_input = 0.5
        if score > 0.5:
            serotonin_input = 0.7
        if score > 0.7:
            serotonin_input = 0.9
        h["serotonin"] = 0.85 * h["serotonin"] + 0.15 * serotonin_input

        ach_input = 0.5
        if self._circadian_phase == "work":
            ach_input = 0.7
        elif self._circadian_phase == "consolidation":
            ach_input = 0.3
        h["acetylcholine"] = 0.8 * h["acetylcholine"] + 0.2 * ach_input

        for key in h:
            h[key] = max(0.0, min(1.0, h[key]))

    def _apply_endocrine(self) -> List[str]:
        actions = []
        h = self._endocrine_hormones

        if h["dopamine"] > 0.7:
            self._adjust("hebbian_reinforce_factor", 0.01, max_val=1.5)
            actions.append("endocrine:dopamine_boost_reinforcement")
        elif h["dopamine"] < 0.3:
            self._adjust("hebbian_reinforce_factor", -0.01, min_val=1.0)
            actions.append("endocrine:dopamine_reduce_reinforcement")

        if h["cortisol"] > 0.6:
            self._adjust("pulse_interval_multiplier", 0.05, max_val=2.5)
            self._adjust("pulse_strength_multiplier", -0.03, min_val=0.3)
            actions.append("endocrine:cortisol_throttle")
        elif h["cortisol"] < 0.2:
            self._adjust("pulse_strength_multiplier", 0.02, max_val=2.0)
            actions.append("endocrine:cortisol_low_normalize")

        if h["serotonin"] > 0.7:
            self._adjust("dream_frequency_multiplier", -0.05, min_val=0.3)
            actions.append("endocrine:serotonin_stable")
        elif h["serotonin"] < 0.3:
            self._adjust("dream_frequency_multiplier", 0.1, max_val=3.0)
            self._adjust("attention_diffusion_rate", 0.02, max_val=0.6)
            actions.append("endocrine:serotonin_boost_exploration")

        if h["acetylcholine"] > 0.6:
            self._adjust("attention_decay_rate", -0.003, min_val=0.01)
            actions.append("endocrine:acetylcholine_focus")
        elif h["acetylcholine"] < 0.3:
            self._adjust("attention_decay_rate", 0.003, max_val=0.15)
            actions.append("endocrine:acetylcholine_relax")

        return actions

    # ── Layer 6: Stress Response ──────────────────────────────

    def _update_stress(self, emergence: Dict):
        score = emergence.get("overall_score", 0)
        stressors = []

        field = self._field
        occupied = len(field._occupied_ids)
        total = len(field._nodes)
        if total > 0:
            density = occupied / total
            if density > 0.8:
                stressors.append(0.5 * (density - 0.8) / 0.2)

        if hasattr(field, '_pulse_log'):
            recent = [e for e in field._pulse_log if time.time() - e.get("time", 0) < 60]
            pulse_rate = len(recent) / 60.0
            if pulse_rate > 30:
                stressors.append(0.3 * min((pulse_rate - 30) / 30, 1.0))

        if score < 0.15:
            stressors.append(0.4 * (0.15 - score) / 0.15)

        cortisol = self._endocrine_hormones.get("cortisol", 0)
        if cortisol > 0.6:
            stressors.append(0.3 * cortisol)

        target_stress = min(sum(stressors), 1.0)
        self._stress_level = 0.85 * self._stress_level + 0.15 * target_stress

        if self._stress_level > self._stress_threshold_critical:
            self._stress_emergency_mode = True
        elif self._stress_level < self._stress_threshold_warning * 0.8:
            self._stress_emergency_mode = False

    def _apply_stress_response(self) -> List[str]:
        actions = []
        if self._stress_emergency_mode:
            self._params["pulse_interval_multiplier"] = min(
                self._params["pulse_interval_multiplier"] + 0.1, 3.0
            )
            self._params["pulse_strength_multiplier"] = max(
                self._params["pulse_strength_multiplier"] - 0.05, 0.2
            )
            actions.append("stress:emergency_throttle")
        elif self._stress_level > self._stress_threshold_warning:
            self._adjust("pulse_interval_multiplier", 0.03, max_val=2.5)
            actions.append("stress:warning_slowdown")
        return actions

    # ── Helpers ───────────────────────────────────────────────

    def _get_emergence_data(self) -> Dict:
        if self._field._emergence_history:
            return self._field._emergence_history[-1]
        return {"overall_score": 0.0, "bridges": {}, "crystal": {}, "phase": {}}

    def _adjust(self, param: str, delta: float, min_val: float = None, max_val: float = None):
        current = self._params.get(param, 0)
        new_val = current + delta
        if min_val is not None:
            new_val = max(min_val, new_val)
        if max_val is not None:
            new_val = min(max_val, new_val)
        self._params[param] = new_val

    def _apply_params_to_field(self):
        field = self._field

        field._attention_decay_rate = self._params["attention_decay_rate"]
        field._attention_diffusion_rate = self._params["attention_diffusion_rate"]

        multiplier = self._params["pulse_interval_multiplier"]
        base_interval = 0.5
        field._adaptive_interval = max(0.2, min(2.0, base_interval * multiplier))

        if hasattr(field._hebbian, '_reinforce'):
            field._hebbian._reinforce = self._params["hebbian_reinforce_factor"]

        if self._circadian_phase == "consolidation":
            field._adaptive_interval *= 1.3
        elif self._circadian_phase == "work":
            field._adaptive_interval *= 0.9

        if self._stress_emergency_mode:
            field._adaptive_interval = max(field._adaptive_interval, 1.5)
