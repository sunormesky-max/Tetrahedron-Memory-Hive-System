from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .pcnn_types import PCNNConfig, PulseType

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField

logger = logging.getLogger("tetramem.honeycomb")


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
