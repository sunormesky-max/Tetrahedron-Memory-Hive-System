"""
Self-Regulation Engine -- Six-Layer Physiological Control System

Six interacting layers modeled after human physiology:

1. Homeostasis (PID Controller)
   Each regulated metric has a PID controller:
     - Proportional: responds to current error
     - Integral: accumulates past error (eliminates steady-state drift)
     - Derivative: responds to rate of change (damps oscillation)
   With anti-windup to prevent integral saturation.

2. Circadian Rhythm
   Alternates work/consolidation phases with configurable period.
   During consolidation: slower pulses, stronger Hebbian reinforcement,
   more frequent dream cycles. During work: faster responses, wider attention.

3. Autonomic Nervous System
   Continuous spectrum from sympathetic (active/alert) to parasympathetic
   (rest/repair). Mode determined by:
     - Stress level (sympathetic activation)
     - Query success rate (parasympathetic reward)
     - Emergence quality (system health indicator)
   Hysteresis bands prevent mode thrashing.

4. Immune System
   Periodic scan for structural anomalies:
     - Dead Hebbian edges (nodes deleted but edges remain)
     - Orphan temporal edges
     - Dead crystal references
     - Stale attention mask entries
     - Excess empty node memory
     - Inconsistent _occupied_ids vs actual node state
     - Lattice connectivity gaps (missing neighbors)

5. Endocrine System (Hormone Modulators)
   Four hormones with distinct effects:
     - Dopamine: reward signal, increases Hebbian reinforcement
     - Cortisol: stress signal, throttles pulse engine
     - Serotonin: satisfaction signal, stabilizes dream frequency
     - Acetylcholine: attention signal, modulates focus depth
   Hormones decay with configurable half-lives and have nonlinear effects.

6. Stress Response
   Multi-source stress aggregation:
     - Occupancy density (field getting full)
     - Pulse rate excess (overactive system)
     - Low emergence quality (system degrading)
     - Cortisol feedback (endocrine reinforcement)
   Emergency throttle with hysteresis to prevent rapid on/off cycling.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .pcnn_types import PCNNConfig

logger = logging.getLogger("tetramem.self_regulation")


class PIDController:
    __slots__ = (
        "_kp", "_ki", "_kd",
        "_integral", "_prev_error", "_integral_limit",
        "_output_min", "_output_max",
    )

    def __init__(
        self,
        kp: float = 0.1,
        ki: float = 0.01,
        kd: float = 0.02,
        integral_limit: float = 2.0,
        output_min: float = -1.0,
        output_max: float = 1.0,
    ):
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral = 0.0
        self._prev_error = 0.0
        self._integral_limit = integral_limit
        self._output_min = output_min
        self._output_max = output_max

    def update(self, setpoint: float, measurement: float, dt: float = 1.0) -> float:
        error = setpoint - measurement
        self._integral += error * dt
        self._integral = max(-self._integral_limit, min(self._integral_limit, self._integral))
        derivative = (error - self._prev_error) / max(dt, 0.001)
        self._prev_error = error
        output = self._kp * error + self._ki * self._integral + self._kd * derivative
        return max(self._output_min, min(self._output_max, output))

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


class SelfRegulationEngine:

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
        self._last_regulate_time = time.time()

        self._pid_controllers = {
            "bridge_rate": PIDController(kp=0.15, ki=0.02, kd=0.03, integral_limit=1.0),
            "crystal_ratio": PIDController(kp=0.1, ki=0.01, kd=0.02, integral_limit=0.8),
            "field_entropy": PIDController(kp=0.08, ki=0.01, kd=0.02, integral_limit=0.5),
            "activation_mean": PIDController(kp=0.05, ki=0.005, kd=0.01, integral_limit=1.0),
            "emergence_score": PIDController(kp=0.12, ki=0.015, kd=0.025, integral_limit=1.5),
        }

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
        self._autonomic_hysteresis_high = 0.65
        self._autonomic_hysteresis_low = 0.35
        self._autonomic_raw_score = 0.5

        self._immune_log: List[Dict[str, Any]] = []
        self._immune_total_repairs = 0
        self._immune_scan_interval = 600
        self._immune_last_scan = 0.0
        self._immune_anomaly_count = 0

        self._endocrine_hormones = {
            "dopamine": 0.5,
            "cortisol": 0.0,
            "serotonin": 0.5,
            "acetylcholine": 0.5,
        }
        self._endocrine_halflife = {
            "dopamine": 0.85,
            "cortisol": 0.90,
            "serotonin": 0.88,
            "acetylcholine": 0.82,
        }
        self._query_success_history: deque = deque(maxlen=100)

        self._stress_level = 0.0
        self._stress_threshold_warning = 0.6
        self._stress_threshold_critical = 0.85
        self._stress_emergency_mode = False
        self._stress_hysteresis_enter = 0.85
        self._stress_hysteresis_exit = 0.50

        self._regulation_log: List[Dict[str, Any]] = []
        self._max_log = 200

        self._dark_plane_feedback_score = 0.0
        self._substrate = None

    def set_substrate(self, substrate):
        self._substrate = substrate

    def regulate(self) -> Dict[str, Any]:
        now = time.time()
        dt = max(now - self._last_regulate_time, 0.01)
        self._last_regulate_time = now
        actions = []

        self._update_circadian(now)
        actions.append(f"circadian:{self._circadian_phase}")

        emergence = self._get_emergence_data()
        self._last_emergence_score = emergence.get("overall_score", 0)

        self._update_endocrine(emergence, dt)
        endocrine_actions = self._apply_endocrine()
        actions.extend(endocrine_actions)

        self._update_stress(emergence, dt)
        stress_actions = self._apply_stress_response()
        actions.extend(stress_actions)

        self._update_autonomic(emergence, dt)
        autonomic_actions = self._apply_autonomic()
        actions.extend(autonomic_actions)

        homeostasis_actions = self._apply_homeostasis(emergence, dt)
        actions.extend(homeostasis_actions)

        if now - self._immune_last_scan > self._immune_scan_interval:
            immune_actions = self._run_immune()
            actions.extend(immune_actions)
            self._immune_last_scan = now

        self._apply_params_to_field()

        if self._substrate is not None:
            sig = self._substrate.get_regulation_signals()
            pe = sig.get("persistent_entropy", 0)
            coherence = sig.get("coherence", 0)
            h5_reg = sig.get("h5_regulation", 0)
            h6_cascade = sig.get("h6_cascade_active", False)
            psi = sig.get("psi_field", 0)

            self._endocrine_hormones["dopamine"] = min(
                1.0, self._endocrine_hormones["dopamine"] + 0.05 * pe
            )
            self._endocrine_hormones["serotonin"] = min(
                1.0, self._endocrine_hormones["serotonin"] + 0.03 * pe
            )
            self._endocrine_hormones["acetylcholine"] = min(
                1.0, self._endocrine_hormones["acetylcholine"] + 0.04 * coherence
            )

            if h5_reg > 0:
                self._autonomic_raw_score = max(0, self._autonomic_raw_score - 0.05 * h5_reg)
            elif h5_reg < 0:
                self._autonomic_raw_score = min(1, self._autonomic_raw_score + 0.05 * abs(h5_reg))

            if h6_cascade:
                self._stress_level = min(1.0, self._stress_level + 0.5)
                self._endocrine_hormones["cortisol"] = min(
                    1.0, self._endocrine_hormones["cortisol"] + 0.4
                )

            self._endocrine_hormones["dopamine"] = min(
                1.0, self._endocrine_hormones["dopamine"] + 0.1 * psi
            )

        self._regulation_count += 1
        record = {
            "time": now,
            "emergence_score": emergence.get("overall_score", 0),
            "circadian": self._circadian_phase,
            "autonomic": self._autonomic_mode,
            "stress": round(self._stress_level, 4),
            "hormones": {k: round(v, 4) for k, v in self._endocrine_hormones.items()},
            "params": {k: round(v, 6) if isinstance(v, float) else v for k, v in self._params.items()},
            "actions": actions,
            "dt": round(dt, 4),
        }
        self._regulation_log.append(record)
        if len(self._regulation_log) > self._max_log:
            self._regulation_log = self._regulation_log[-self._max_log // 2:]

        return record

    def notify_query_result(self, score: float, k: int, returned: int):
        hit_rate = returned / max(k, 1)
        combined = score * 0.4 + hit_rate * 0.6
        self._query_success_history.append(combined)

    def notify_dark_plane_transitions(self, transitions: int, reawakenings: int, total_occupied: int):
        if total_occupied > 0:
            transition_rate = transitions / total_occupied
            reawakening_rate = reawakenings / total_occupied

            if transition_rate > 0.1:
                self._dark_plane_feedback_score = min(1.0, self._dark_plane_feedback_score + 0.2)
            else:
                self._dark_plane_feedback_score = max(0.0, self._dark_plane_feedback_score - 0.05)

            if reawakening_rate > 0.02:
                self._endocrine_hormones["dopamine"] = min(
                    1.0, self._endocrine_hormones["dopamine"] + reawakening_rate * 0.5
                )

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
                "raw_score": round(self._autonomic_raw_score, 4),
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
                "anomaly_count": self._immune_anomaly_count,
                "last_scan": round(self._immune_last_scan, 1),
                "scan_interval": self._immune_scan_interval,
            },
            "query_success_avg": round(
                float(np.mean(list(self._query_success_history)[-20:])) if self._query_success_history else 0, 4
            ),
            "dark_plane_feedback": round(self._dark_plane_feedback_score, 4),
        }

    def get_history(self, n: int = 20) -> List[Dict]:
        return self._regulation_log[-n:]

    def force_mode(self, mode: str):
        if mode in ("sympathetic", "parasympathetic", "balanced"):
            self._autonomic_mode = mode
        if mode in ("work", "consolidation"):
            self._circadian_phase = mode
            self._circadian_last_switch = time.time()

    def _get_emergence_data(self) -> Dict:
        if self._field._emergence_history:
            return self._field._emergence_history[-1]
        return {"overall_score": 0.0, "bridges": {}, "crystal": {}, "phase": {}}

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))

    # ── Layer 1: Homeostasis (PID) ─────────────────────────────

    def _apply_homeostasis(self, emergence: Dict, dt: float) -> List[str]:
        actions = []
        score = emergence.get("overall_score", 0)

        bridges = emergence.get("bridges", {})
        crystal = emergence.get("crystal", {})
        phase = emergence.get("phase", {})

        bridge_rate = bridges.get("creation_rate", 0)
        crystal_ratio = crystal.get("crystal_ratio", 0)
        entropy = phase.get("field_entropy", 1.0)
        activation = crystal.get("avg_activation", 0)

        pid_bridge = self._pid_controllers["bridge_rate"]
        pid_bridge_output = pid_bridge.update(0.03, bridge_rate, dt)
        if abs(pid_bridge_output) > 0.001:
            self._params["bridge_threshold_offset"] = self._clamp(
                self._params["bridge_threshold_offset"] - pid_bridge_output * 0.3,
                -0.3, 0.3
            )
            self._params["pulse_strength_multiplier"] = self._clamp(
                self._params["pulse_strength_multiplier"] + pid_bridge_output * 0.1,
                0.3, 2.5
            )
            if pid_bridge_output > 0:
                actions.append("homeostasis:boost_bridges")
            else:
                actions.append("homeostasis:throttle_bridges")

        pid_crystal = self._pid_controllers["crystal_ratio"]
        pid_crystal_output = pid_crystal.update(0.5, crystal_ratio, dt)
        if abs(pid_crystal_output) > 0.001:
            self._params["hebbian_reinforce_factor"] = self._clamp(
                self._params["hebbian_reinforce_factor"] + pid_crystal_output * 0.05,
                1.0, 1.8
            )
            actions.append("homeostasis:crystal_pid")

        pid_entropy = self._pid_controllers["field_entropy"]
        pid_entropy_output = pid_entropy.update(0.55, entropy, dt)
        if abs(pid_entropy_output) > 0.001:
            self._params["pulse_interval_multiplier"] = self._clamp(
                self._params["pulse_interval_multiplier"] - pid_entropy_output * 0.1,
                0.3, 2.5
            )
            self._params["attention_decay_rate"] = self._clamp(
                self._params["attention_decay_rate"] + pid_entropy_output * 0.01,
                0.01, 0.2
            )
            actions.append("homeostasis:entropy_pid")

        pid_activation = self._pid_controllers["activation_mean"]
        pid_activation_output = pid_activation.update(8.0, activation, dt)
        if abs(pid_activation_output) > 0.001:
            self._params["pulse_strength_multiplier"] = self._clamp(
                self._params["pulse_strength_multiplier"] + pid_activation_output * 0.05,
                0.3, 2.5
            )
            actions.append("homeostasis:activation_pid")

        pid_emergence = self._pid_controllers["emergence_score"]
        pid_emergence_output = pid_emergence.update(0.55, score, dt)
        if abs(pid_emergence_output) > 0.01:
            self._params["dream_frequency_multiplier"] = self._clamp(
                self._params["dream_frequency_multiplier"] - pid_emergence_output * 0.2,
                0.3, 3.0
            )
            self._params["attention_diffusion_rate"] = self._clamp(
                self._params["attention_diffusion_rate"] + pid_emergence_output * 0.05,
                0.1, 0.6
            )
            if score < 0.25:
                self._params["cascade_depth_bonus"] = max(
                    self._params["cascade_depth_bonus"], 1
                )
            else:
                self._params["cascade_depth_bonus"] = max(
                    0, self._params["cascade_depth_bonus"] - 1
                )
            actions.append("homeostasis:emergence_pid")

        return actions

    # ── Layer 2: Circadian Rhythm ──────────────────────────────

    def _update_circadian(self, now: float):
        elapsed_in_period = (now - self._circadian_last_switch) % self._circadian_period
        work_duration = self._circadian_period * self._circadian_work_ratio

        new_phase = "work" if elapsed_in_period < work_duration else "consolidation"
        if new_phase != self._circadian_phase:
            self._circadian_phase = new_phase
            self._circadian_last_switch = now
            logger.info("Circadian phase switch -> %s", new_phase)

    # ── Layer 3: Autonomic Nervous ─────────────────────────────

    def _update_autonomic(self, emergence: Dict, dt: float):
        score = emergence.get("overall_score", 0)
        stress = self._stress_level
        qh = list(self._query_success_history)[-20:]
        query_success = float(np.mean(qh)) if qh else 0.5

        raw = 0.5 + (stress - 0.3) * 0.4 - (query_success - 0.5) * 0.3 - (score - 0.5) * 0.3
        raw = max(0.0, min(1.0, raw))

        decay = 0.85
        self._autonomic_raw_score = decay * self._autonomic_raw_score + (1 - decay) * raw

        s = self._autonomic_raw_score
        if s > self._autonomic_hysteresis_high:
            self._autonomic_mode = "sympathetic"
        elif s < self._autonomic_hysteresis_low:
            self._autonomic_mode = "parasympathetic"
        else:
            if self._autonomic_mode == "sympathetic" and s > self._autonomic_hysteresis_low:
                pass
            elif self._autonomic_mode == "parasympathetic" and s < self._autonomic_hysteresis_high:
                pass
            else:
                self._autonomic_mode = "balanced"

        target_symp = s
        target_para = 1.0 - s
        rate = 0.1
        self._sympathetic_score += rate * (target_symp - self._sympathetic_score)
        self._parasympathetic_score += rate * (target_para - self._parasympathetic_score)

    def _apply_autonomic(self) -> List[str]:
        actions = []
        if self._autonomic_mode == "sympathetic":
            self._params["pulse_interval_multiplier"] = self._clamp(
                self._params["pulse_interval_multiplier"] - 0.03, 0.3, 2.5
            )
            self._params["attention_diffusion_rate"] = self._clamp(
                self._params["attention_diffusion_rate"] + 0.02, 0.1, 0.6
            )
            actions.append("autonomic:sympathetic")
        elif self._autonomic_mode == "parasympathetic":
            self._params["pulse_interval_multiplier"] = self._clamp(
                self._params["pulse_interval_multiplier"] + 0.03, 0.3, 2.5
            )
            self._params["attention_decay_rate"] = self._clamp(
                self._params["attention_decay_rate"] + 0.003, 0.01, 0.15
            )
            self._params["hebbian_reinforce_factor"] = self._clamp(
                self._params["hebbian_reinforce_factor"] + 0.005, 1.0, 1.8
            )
            actions.append("autonomic:parasympathetic")
        return actions

    # ── Layer 4: Immune ────────────────────────────────────────

    def _run_immune(self) -> List[str]:
        actions = []
        field = self._field
        repairs = 0
        anomalies = 0

        try:
            with field._lock:
                dead_edges = 0
                for nid in list(field._hebbian._edges.keys()):
                    a, b = nid
                    if a not in field._occupied_ids or b not in field._occupied_ids:
                        dead_edges += 1
                        anomalies += 1
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
                        anomalies += 1
                    else:
                        before = len(field._temporal_edges[nid])
                        field._temporal_edges[nid] = [
                            (eid, p) for eid, p in field._temporal_edges[nid]
                            if eid in field._occupied_ids
                        ]
                        removed = before - len(field._temporal_edges[nid])
                        if removed > 0:
                            orphan_temporal += removed
                            anomalies += 1
                if orphan_temporal > 0:
                    repairs += orphan_temporal
                    actions.append(f"immune:cleaned_{orphan_temporal}_orphan_temporal")

                if field._crystallized and hasattr(field._crystallized, '_crystals'):
                    dead_crystals = 0
                    for key in list(field._crystallized._crystals.keys()):
                        if hasattr(key, '__iter__') and not isinstance(key, str):
                            parts = list(key)
                            alive = all(p in field._occupied_ids for p in parts)
                        else:
                            alive = key in field._occupied_ids
                        if not alive:
                            dead_crystals += 1
                            del field._crystallized._crystals[key]
                            anomalies += 1
                    if dead_crystals > 0:
                        repairs += dead_crystals
                        actions.append(f"immune:pruned_{dead_crystals}_dead_crystals")

                mask_repairs = 0
                for nid in list(field._attention_mask.keys()):
                    if nid not in field._occupied_ids:
                        del field._attention_mask[nid]
                        mask_repairs += 1
                        anomalies += 1
                if mask_repairs > 0:
                    repairs += mask_repairs
                    actions.append(f"immune:cleaned_{mask_repairs}_attention_mask")

                occupied_actual = sum(
                    1 for n in field._nodes.values() if n.is_occupied
                )
                if occupied_actual != len(field._occupied_ids):
                    anomalies += 1
                    actual_ids = {nid for nid, n in field._nodes.items() if n.is_occupied}
                    missing = actual_ids - field._occupied_ids
                    extra = field._occupied_ids - actual_ids
                    if missing:
                        field._occupied_ids.update(missing)
                        field._occupied_count = len(field._occupied_ids)
                        repairs += len(missing)
                        actions.append(f"immune:fixed_{len(missing)}_missing_occupied")
                    if extra:
                        field._occupied_ids -= extra
                        field._occupied_count = len(field._occupied_ids)
                        repairs += len(extra)
                        actions.append(f"immune:fixed_{len(extra)}_extra_occupied")

                for nid in list(field._frontier_empty.keys()):
                    node = field._nodes.get(nid)
                    if node is None:
                        del field._frontier_empty[nid]
                        anomalies += 1

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
            self._immune_anomaly_count += anomalies
            self._immune_log.append({
                "time": time.time(),
                "repairs": repairs,
                "anomalies": anomalies,
                "actions": actions,
            })
            if len(self._immune_log) > 50:
                self._immune_log = self._immune_log[-25:]

        return actions

    # ── Layer 5: Endocrine ─────────────────────────────────────

    def _update_endocrine(self, emergence: Dict, dt: float):
        score = emergence.get("overall_score", 0)
        qh = list(self._query_success_history)[-20:]
        recent_success = float(np.mean(qh)) if qh else 0.5

        h = self._endocrine_hormones
        hl = self._endocrine_halflife

        dopamine_decay = hl["dopamine"]
        h["dopamine"] = dopamine_decay * h["dopamine"] + (1 - dopamine_decay) * recent_success

        cortisol_input = 0.0
        if score < 0.2:
            cortisol_input = 0.8
        elif score < 0.4:
            cortisol_input = 0.4
        elif self._stress_level > 0.6:
            cortisol_input = 0.5
        if self._dark_plane_feedback_score > 0.5:
            cortisol_input = min(1.0, cortisol_input + self._dark_plane_feedback_score * 0.3)
        cortisol_decay = hl["cortisol"]
        h["cortisol"] = cortisol_decay * h["cortisol"] + (1 - cortisol_decay) * cortisol_input

        serotonin_input = 0.5
        if score > 0.5:
            serotonin_input = 0.7
        if score > 0.7:
            serotonin_input = 0.9
        serotonin_decay = hl["serotonin"]
        h["serotonin"] = serotonin_decay * h["serotonin"] + (1 - serotonin_decay) * serotonin_input

        ach_input = 0.5
        if self._circadian_phase == "work":
            ach_input = 0.7
        elif self._circadian_phase == "consolidation":
            ach_input = 0.3
        ach_decay = hl["acetylcholine"]
        h["acetylcholine"] = ach_decay * h["acetylcholine"] + (1 - ach_decay) * ach_input

        for key in h:
            h[key] = max(0.0, min(1.0, h[key]))

    def _apply_endocrine(self) -> List[str]:
        actions = []
        h = self._endocrine_hormones

        dopamine = h["dopamine"]
        if dopamine > 0.7:
            boost = (dopamine - 0.7) * 0.04
            self._params["hebbian_reinforce_factor"] = self._clamp(
                self._params["hebbian_reinforce_factor"] + boost, 1.0, 1.8
            )
            actions.append("endocrine:dopamine_boost")
        elif dopamine < 0.3:
            reduction = (0.3 - dopamine) * 0.03
            self._params["hebbian_reinforce_factor"] = self._clamp(
                self._params["hebbian_reinforce_factor"] - reduction, 1.0, 1.8
            )
            actions.append("endocrine:dopamine_reduce")

        cortisol = h["cortisol"]
        if cortisol > 0.6:
            throttle = (cortisol - 0.6) * 0.15
            self._params["pulse_interval_multiplier"] = self._clamp(
                self._params["pulse_interval_multiplier"] + throttle, 0.3, 3.0
            )
            self._params["pulse_strength_multiplier"] = self._clamp(
                self._params["pulse_strength_multiplier"] - throttle * 0.2, 0.3, 2.5
            )
            actions.append("endocrine:cortisol_throttle")
        elif cortisol < 0.2:
            self._params["pulse_strength_multiplier"] = self._clamp(
                self._params["pulse_strength_multiplier"] + 0.02, 0.3, 2.5
            )
            actions.append("endocrine:cortisol_normalize")

        serotonin = h["serotonin"]
        if serotonin > 0.7:
            self._params["dream_frequency_multiplier"] = self._clamp(
                self._params["dream_frequency_multiplier"] - 0.03, 0.3, 3.0
            )
            actions.append("endocrine:serotonin_stable")
        elif serotonin < 0.3:
            self._params["dream_frequency_multiplier"] = self._clamp(
                self._params["dream_frequency_multiplier"] + 0.05, 0.3, 3.0
            )
            self._params["attention_diffusion_rate"] = self._clamp(
                self._params["attention_diffusion_rate"] + 0.02, 0.1, 0.6
            )
            actions.append("endocrine:serotonin_explore")

        ach = h["acetylcholine"]
        if ach > 0.6:
            self._params["attention_decay_rate"] = self._clamp(
                self._params["attention_decay_rate"] - 0.003, 0.01, 0.15
            )
            actions.append("endocrine:ach_focus")
        elif ach < 0.3:
            self._params["attention_decay_rate"] = self._clamp(
                self._params["attention_decay_rate"] + 0.003, 0.01, 0.15
            )
            actions.append("endocrine:ach_relax")

        return actions

    # ── Layer 6: Stress Response ───────────────────────────────

    def _update_stress(self, emergence: Dict, dt: float):
        score = emergence.get("overall_score", 0)
        stressors = []

        field = self._field
        occupied = len(field._occupied_ids)
        total = len(field._nodes)
        if total > 0:
            density = occupied / total
            if density > 0.7:
                stressors.append(0.4 * ((density - 0.7) / 0.3) ** 1.5)

        if hasattr(field, '_pulse_log'):
            recent = [e for e in field._pulse_log if time.time() - e.get("time", 0) < 60]
            pulse_rate = len(recent) / 60.0
            if pulse_rate > 20:
                stressors.append(0.25 * min((pulse_rate - 20) / 30, 1.0))

        if score < 0.15:
            stressors.append(0.5 * ((0.15 - score) / 0.15) ** 1.2)

        cortisol = self._endocrine_hormones.get("cortisol", 0)
        if cortisol > 0.5:
            stressors.append(0.2 * (cortisol - 0.5) * 2)

        if self._dark_plane_feedback_score > 0.6:
            stressors.append(0.15 * self._dark_plane_feedback_score)

        target_stress = min(sum(stressors), 1.0)
        smoothing = math.exp(-dt * 0.5)
        self._stress_level = smoothing * self._stress_level + (1 - smoothing) * target_stress
        self._stress_level = max(0.0, min(1.0, self._stress_level))

        if self._stress_emergency_mode:
            if self._stress_level < self._stress_hysteresis_exit:
                self._stress_emergency_mode = False
        else:
            if self._stress_level > self._stress_hysteresis_enter:
                self._stress_emergency_mode = True

    def _apply_stress_response(self) -> List[str]:
        actions = []
        if self._stress_emergency_mode:
            self._params["pulse_interval_multiplier"] = min(
                self._params["pulse_interval_multiplier"] + 0.15, 3.0
            )
            self._params["pulse_strength_multiplier"] = max(
                self._params["pulse_strength_multiplier"] - 0.08, 0.2
            )
            self._params["cascade_depth_bonus"] = 0
            actions.append("stress:emergency_throttle")
        elif self._stress_level > self._stress_threshold_warning:
            self._params["pulse_interval_multiplier"] = self._clamp(
                self._params["pulse_interval_multiplier"] + 0.03, 0.3, 2.5
            )
            actions.append("stress:warning_slowdown")
        return actions

    # ── Apply all params to field ──────────────────────────────

    def _apply_params_to_field(self):
        field = self._field

        field._attention_decay_rate = self._params["attention_decay_rate"]
        field._attention_diffusion_rate = self._params["attention_diffusion_rate"]

        multiplier = self._params["pulse_interval_multiplier"]
        base_interval = PCNNConfig.BASE_PULSE_INTERVAL if hasattr(field, '_resolution') else 0.5
        field._adaptive_interval = max(0.2, min(3.0, base_interval * multiplier))

        if hasattr(field._hebbian, '_reinforce'):
            field._hebbian._reinforce = self._params["hebbian_reinforce_factor"]

        if self._circadian_phase == "consolidation":
            field._adaptive_interval *= 1.3
        elif self._circadian_phase == "work":
            field._adaptive_interval *= 0.9

        if hasattr(field, '_dream_engine') and field._dream_engine:
            dream_interval = PCNNConfig.DREAM_CYCLE_INTERVAL
            freq_mult = self._params["dream_frequency_multiplier"]
            field._dream_engine._cycle_interval = max(
                50, int(dream_interval / max(freq_mult, 0.3))
            )

        if hasattr(field, '_bridge_threshold'):
            field._bridge_threshold = max(
                0.1, field._bridge_threshold + self._params["bridge_threshold_offset"] * 0.01
            )

        if hasattr(field, '_pulse_strength_base'):
            field._pulse_strength_base = self._params["pulse_strength_multiplier"]

        if self._stress_emergency_mode:
            field._adaptive_interval = max(field._adaptive_interval, 1.5)

    @property
    def cascade_depth_bonus(self) -> int:
        return self._params["cascade_depth_bonus"]

    @property
    def pulse_strength_multiplier(self) -> float:
        return self._params["pulse_strength_multiplier"]

    @property
    def dream_frequency_multiplier(self) -> float:
        return self._params["dream_frequency_multiplier"]
