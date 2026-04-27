from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List

from .pcnn_types import PCNNConfig, PulseType

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField

logger = logging.getLogger("tetramem.honeycomb")


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
        self._consecutive_negative: Dict[str, int] = defaultdict(int)

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
                    field._hebbian._edges.pop((target_id, source_id), None)
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

    def record_and_learn(self, action: str, context_id: str, outcome: str,
                         confidence: float = 0.5, reasoning: str = "",
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced feedback that modifies the neural topology based on outcomes.
        """
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
        adjustments = []

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

            if outcome == "positive":
                boost = confidence * 0.2
                node.weight = min(10.0, node.weight + boost)
                node.activation = min(10.0, node.activation + boost * 0.5)
                if "__low_priority__" in node.labels:
                    node.labels.remove("__low_priority__")

                field._emit_pulse(
                    context_id, strength=boost * 2.0,
                    pulse_type=PulseType.REINFORCING,
                )

                if field._hebbian:
                    for fnid in node.face_neighbors[:8]:
                        fn = field._nodes.get(fnid)
                        if fn and fn.is_occupied:
                            shared = len(set(node.labels) & set(fn.labels))
                            if shared > 0:
                                field._hebbian.record_path(
                                    [context_id, fnid], True,
                                    0.3 * confidence * shared,
                                )
                adjustments.append(f"reinforcing_pulse:+{boost:.3f}")

                self._consecutive_positive[context_id] = self._consecutive_positive.get(context_id, 0) + 1
                if self._consecutive_positive[context_id] >= 3:
                    if field._hebbian:
                        for fnid in node.face_neighbors[:8]:
                            fn = field._nodes.get(fnid)
                            if fn and fn.is_occupied:
                                field._hebbian.record_path([context_id, fnid], True, 0.5)

                    if field._crystallized and self._consecutive_positive[context_id] >= 5:
                        for fnid in node.face_neighbors[:6]:
                            path_weight = field._hebbian.get_path_bias(context_id, fnid)
                            if path_weight > PCNNConfig.CRYSTALLIZE_THRESHOLD * 0.7:
                                field._crystallized.try_crystallize(context_id, fnid, path_weight)
                                adjustments.append("crystal_candidate")

                    adjustments.append("hebbian_reinforced_deep")

                node_labels = [l for l in node.labels if not l.startswith("__")]
                if node_labels:
                    pattern_key = f"success_{action}_{'_'.join(sorted(node_labels[:3]))}"
                    field.store(
                        content=f"[meta-learning] 成功模式: {action} in {','.join(node_labels[:3])} | {reasoning[:60]}",
                        labels=node_labels[:2] + ["__meta_learning__", "__success_pattern__"],
                        weight=0.8 * confidence,
                        metadata={
                            "pattern_type": "success",
                            "action": action,
                            "confidence": round(confidence, 3),
                        },
                    )
                    adjustments.append("success_pattern_stored")

            elif outcome == "negative":
                node.metadata["negative_feedback_count"] = node.metadata.get("negative_feedback_count", 0) + 1
                self._consecutive_positive[context_id] = 0

                field._emit_pulse(
                    context_id, strength=0.1,
                    pulse_type=PulseType.TENSION_SENSING,
                )
                adjustments.append("tension_pulse_emitted")

                node_labels = [l for l in node.labels if not l.startswith("__")]
                if node_labels:
                    field.store(
                        content=f"[meta-learning] 谨慎模式: {action} in {','.join(node_labels[:3])} | {reasoning[:60]}",
                        labels=node_labels[:2] + ["__meta_learning__", "__caution__"],
                        weight=0.5,
                        metadata={
                            "pattern_type": "caution",
                            "action": action,
                            "confidence": round(confidence, 3),
                        },
                    )
                    adjustments.append("caution_pattern_stored")

            else:
                node.access_count += 1
                adjustments.append("access_count_incremented")

        return {"recorded": True, "action_taken": "; ".join(adjustments)}

    def extract_learning_patterns(self) -> List[Dict[str, Any]]:
        """
        Extract learned patterns from feedback history.
        """
        patterns: List[Dict[str, Any]] = []
        with self._lock:
            action_domain: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0}))
            for r in self._records:
                domain = "__unknown__"
                n = self._field._nodes.get(r.context_id)
                if n and n.is_occupied:
                    labels = [l for l in n.labels if not l.startswith("__")]
                    domain = labels[0] if labels else "__unlabeled__"
                action_domain[r.action][domain][r.outcome] += 1

            for action, domains in action_domain.items():
                for domain, counts in domains.items():
                    total = sum(counts.values())
                    if total < 2:
                        continue
                    success_rate = counts["positive"] / total
                    patterns.append({
                        "pattern_type": "action_domain_success",
                        "action": action,
                        "domain": domain,
                        "total_trials": total,
                        "success_rate": round(success_rate, 3),
                        "insight": "highly_effective" if success_rate > 0.7 else "moderate" if success_rate > 0.4 else "low_effectiveness",
                    })

            crystal_success: Dict[str, int] = defaultdict(int)
            crystal_total: Dict[str, int] = defaultdict(int)
            for r in self._records:
                n = self._field._nodes.get(r.context_id)
                if n and n.is_occupied and n.crystal_channels:
                    key = "crystal_path"
                    crystal_total[key] += 1
                    if r.outcome == "positive":
                        crystal_success[key] += 1
            for key in crystal_total:
                if crystal_total[key] >= 2:
                    rate = crystal_success[key] / crystal_total[key]
                    patterns.append({
                        "pattern_type": "crystal_trust",
                        "crystal_success_rate": round(rate, 3),
                        "crystal_trials": crystal_total[key],
                        "insight": "trusted" if rate > 0.6 else "unreliable",
                    })

            temporal: Dict[str, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
            for r in self._records:
                hour_bucket = str(int(r.timestamp // 3600) % 24)
                temporal[hour_bucket][r.outcome] += 1
            for hour, counts in temporal.items():
                total = sum(counts.values())
                if total >= 3:
                    rate = counts["positive"] / total
                    patterns.append({
                        "pattern_type": "temporal",
                        "hour_bucket": hour,
                        "total": total,
                        "success_rate": round(rate, 3),
                        "insight": "peak_performance" if rate > 0.7 else "off_peak" if rate < 0.3 else "normal",
                    })

        patterns.sort(key=lambda x: -x.get("total_trials", x.get("total", 0)))
        return patterns[:20]

    def get_adaptive_weights(self, label_domain: str) -> Dict[str, float]:
        """
        Return learned weight adjustments for a specific domain.
        """
        with self._lock:
            domain_records = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
            for r in self._records:
                n = self._field._nodes.get(r.context_id)
                if n and n.is_occupied:
                    labels = [l for l in n.labels if not l.startswith("__")]
                    if label_domain in labels:
                        domain_records[r.outcome] += 1
                        domain_records["total"] += 1

            total = domain_records["total"]
            if total < 3:
                return {
                    "query_expansion_factor": 1.0,
                    "reinforcement_rate": 1.0,
                    "dream_inclusion_threshold": 0.6,
                    "cluster_cohesion_target": 0.5,
                }

            success_rate = domain_records["positive"] / total
            fail_rate = domain_records["negative"] / total

            if success_rate > 0.6:
                qe = 0.8
                rr = 1.3
                dit = 0.5
                cct = 0.6
            elif fail_rate > 0.4:
                qe = 1.4
                rr = 0.7
                dit = 0.8
                cct = 0.4
            else:
                qe = 1.0
                rr = 1.0
                dit = 0.6
                cct = 0.5

            return {
                "query_expansion_factor": round(qe, 2),
                "reinforcement_rate": round(rr, 2),
                "dream_inclusion_threshold": round(dit, 2),
                "cluster_cohesion_target": round(cct, 2),
            }

    def pulse_driven_reinforce(self, node_id: str, pulse_strength: float) -> Dict[str, Any]:
        field = self._field
        actions = []

        with field._lock:
            node = field._nodes.get(node_id)
            if node is None:
                for nid, n in field._nodes.items():
                    if nid.startswith(node_id) and n.is_occupied:
                        node = n
                        node_id = nid
                        break

            if node is None or not node.is_occupied:
                return {"reinforced": False, "reason": "node_not_found"}

            effective_strength = max(0.1, min(1.0, pulse_strength))

            field._emit_pulse(
                node_id, strength=effective_strength,
                pulse_type=PulseType.REINFORCING,
            )
            actions.append(f"reinforcing_pulse:{effective_strength:.3f}")

            node.weight = min(10.0, node.weight + effective_strength * 0.1)
            node.activation = min(10.0, node.activation + effective_strength * 0.05)
            actions.append(f"weight_boosted:+{effective_strength * 0.1:.3f}")

            if field._hebbian:
                for fnid in node.face_neighbors[:8]:
                    fn = field._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        shared_labels = len(set(node.labels) & set(fn.labels))
                        path_strength = effective_strength * 0.2 * (1.0 + shared_labels * 0.1)
                        field._hebbian.record_path(
                            [node_id, fnid], True, min(1.0, path_strength)
                        )
                actions.append("hebbian_paths_updated")

            crystallized = False
            if field._crystallized and field._hebbian:
                for fnid in node.face_neighbors[:6]:
                    path_weight = field._hebbian.get_path_bias(node_id, fnid)
                    if path_weight > PCNNConfig.CRYSTALLIZE_THRESHOLD:
                        field._crystallized.try_crystallize(node_id, fnid, path_weight)
                        crystallized = True
            if crystallized:
                actions.append("crystallization_attempted")

            with self._lock:
                self._consecutive_positive[node_id] = self._consecutive_positive.get(node_id, 0) + 1
                self._consecutive_negative[node_id] = 0

        return {
            "reinforced": True,
            "node_id": node_id[:12],
            "pulse_strength": round(effective_strength, 3),
            "actions": actions,
            "crystallized": crystallized,
        }

    def apply_negative_feedback(self, node_id: str, reason: str) -> Dict[str, Any]:
        field = self._field
        actions = []

        with field._lock:
            node = field._nodes.get(node_id)
            if node is None:
                for nid, n in field._nodes.items():
                    if nid.startswith(node_id) and n.is_occupied:
                        node = n
                        node_id = nid
                        break

            if node is None or not node.is_occupied:
                return {"applied": False, "reason": "node_not_found"}

            field._emit_pulse(
                node_id, strength=0.15,
                pulse_type=PulseType.TENSION_SENSING,
            )
            actions.append("tension_pulse_emitted")

            with self._lock:
                self._consecutive_negative[node_id] = self._consecutive_negative.get(node_id, 0) + 1
                self._consecutive_positive[node_id] = 0

                consecutive_neg = self._consecutive_negative[node_id]

            if consecutive_neg > 3:
                if "__low_priority__" not in node.labels:
                    node.labels.append("__low_priority__")
                    actions.append("marked_low_priority")
                else:
                    actions.append("already_low_priority")

            node.metadata["negative_feedback_count"] = node.metadata.get("negative_feedback_count", 0) + 1
            node.metadata["last_negative_reason"] = reason[:200]

            if field._hebbian:
                for fnid in node.face_neighbors[:8]:
                    fn = field._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        current_bias = field._hebbian.get_path_bias(node_id, fnid)
                        if current_bias > 0:
                            new_bias = max(
                                PCNNConfig.HEBBIAN_MIN_WEIGHT,
                                current_bias * 0.85,
                            )
                            field._hebbian._edges[(node_id, fnid)] = new_bias
                            field._hebbian._edges[(fnid, node_id)] = new_bias
                actions.append("hebbian_paths_reduced")

        return {
            "applied": True,
            "node_id": node_id[:12],
            "consecutive_negative": consecutive_neg,
            "actions": actions,
            "reason": reason[:100],
        }

    def evolve_weights(self) -> List[Dict[str, Any]]:
        proposals = []
        field = self._field

        with self._lock:
            node_feedback: Dict[str, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0, "total": 0})
            for r in self._records:
                node_feedback[r.context_id]["total"] += 1
                if r.outcome == "positive":
                    node_feedback[r.context_id]["positive"] += 1
                elif r.outcome == "negative":
                    node_feedback[r.context_id]["negative"] += 1

        with field._lock:
            for nid, counts in node_feedback.items():
                if counts["total"] < 3:
                    continue

                node = field._nodes.get(nid)
                if node is None:
                    for fnid, fn in field._nodes.items():
                        if fnid.startswith(nid) and fn.is_occupied:
                            node = fn
                            nid = fnid
                            break

                if node is None or not node.is_occupied:
                    continue

                positive_ratio = counts["positive"] / counts["total"]
                negative_ratio = counts["negative"] / counts["total"]

                current_weight = node.weight

                if positive_ratio > 0.8:
                    proposed_increase = min(2.0, current_weight * 0.15)
                    new_weight = min(10.0, current_weight + proposed_increase)
                    if new_weight > current_weight + 0.01:
                        proposals.append({
                            "node_id": nid[:12],
                            "action": "increase",
                            "current_weight": round(current_weight, 3),
                            "proposed_weight": round(new_weight, 3),
                            "delta": round(proposed_increase, 3),
                            "positive_ratio": round(positive_ratio, 3),
                            "feedback_total": counts["total"],
                            "reasoning": f"positive_ratio={positive_ratio:.0%}>{0.8:.0%}",
                        })

                elif negative_ratio > 0.6:
                    proposed_decrease = min(1.5, current_weight * 0.10)
                    new_weight = max(0.1, current_weight - proposed_decrease)
                    if new_weight < current_weight - 0.01:
                        proposals.append({
                            "node_id": nid[:12],
                            "action": "decrease",
                            "current_weight": round(current_weight, 3),
                            "proposed_weight": round(new_weight, 3),
                            "delta": round(-proposed_decrease, 3),
                            "negative_ratio": round(negative_ratio, 3),
                            "feedback_total": counts["total"],
                            "reasoning": f"negative_ratio={negative_ratio:.0%}>{0.6:.0%}",
                        })

        proposals.sort(key=lambda x: -abs(x["delta"]))
        top_proposals = proposals[:50]

        applied = 0
        for p in top_proposals:
            full_nid = None
            for nid, n in field._nodes.items():
                if nid.startswith(p["node_id"]) and n.is_occupied:
                    full_nid = nid
                    break
            if full_nid:
                node = field._nodes[full_nid]
                if p["action"] == "increase":
                    node.weight = min(10.0, node.weight + abs(p["delta"]) * 0.5)
                    node.activation = min(10.0, node.activation + abs(p["delta"]) * 0.2)
                    applied += 1
                elif p["action"] == "decrease":
                    node.weight = max(0.1, node.weight - abs(p["delta"]) * 0.3)
                    applied += 1

        if applied > 0:
            logger.info("FeedbackLoop: applied %d weight adjustments from %d proposals", applied, len(top_proposals))

        return top_proposals
