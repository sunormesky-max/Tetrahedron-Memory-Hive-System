"""
Agent Memory-Driven Closed Loop — the self-evolution engine.
Pure Python. No external engines.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField


class AgentMemoryLoop:
    """
    The complete agent self-evolution loop:
    1. Agent encounters situation -> stores experience
    2. Memory system auto-organizes -> clusters form
    3. Dream engine runs -> discovers cross-domain insights
    4. Insights surface to agent via proactive notifications
    5. Agent uses insights -> takes action -> records outcome
    6. Feedback loop learns from outcome -> adjusts memory topology
    7. Repeat - system gets smarter each cycle

    This is the CLOSED LOOP that makes the memory system ACTIVE,
    not just a passive store.
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._loop_history: List[Dict] = []
        self._max_history = 200
        self._evolution_metrics: Dict[str, Any] = {
            "total_cycles": 0,
            "insights_generated": 0,
            "insights_actually_used": 0,
            "positive_feedback_rate": 0.0,
            "avg_decision_quality": 0.0,
        }
        self._last_cycle_time: float = 0.0
        self._cycle_count_since_report: int = 0
        self._quality_samples: List[float] = []

    def run_evolution_cycle(self) -> Dict[str, Any]:
        field = self._field
        cycle_start = time.time()
        report: Dict[str, Any] = {
            "cycle": self._evolution_metrics["total_cycles"] + 1,
            "start_time": cycle_start,
            "phases": {},
        }

        with field._lock:
            metrics_before = self._snapshot_metrics()
            report["metrics_before"] = metrics_before

            sense = self._phase_sense(field)
            report["phases"]["SENSE"] = sense

            analyze = self._phase_analyze(field, sense)
            report["phases"]["ANALYZE"] = analyze

            plan = self._phase_plan(field, analyze)
            report["phases"]["PLAN"] = plan

            act = self._phase_act(field, plan)
            report["phases"]["ACT"] = act

            learn = self._phase_learn(field, metrics_before)
            report["phases"]["LEARN"] = learn

        cycle_end = time.time()
        report["duration_seconds"] = round(cycle_end - cycle_start, 3)
        report["status"] = "ok"

        self._evolution_metrics["total_cycles"] += 1
        self._last_cycle_time = cycle_end
        self._evolution_metrics["insights_generated"] += len(
            plan.get("suggestions", [])
        )
        self._evolution_metrics["avg_decision_quality"] = learn.get(
            "quality_score", 0.0
        )

        self._loop_history.append(report)
        if len(self._loop_history) > self._max_history:
            self._loop_history = self._loop_history[-self._max_history // 2 :]

        return report

    def get_evolution_report(self) -> Dict[str, Any]:
        field = self._field
        with field._lock:
            stats = field.stats()
            fb_stats = {}
            if hasattr(field, "_feedback") and field._feedback is not None:
                fb_stats = field._feedback.get_stats()
            so_stats = {}
            if field._self_organize is not None:
                so_stats = field._self_organize.stats()

        return {
            "metrics": dict(self._evolution_metrics),
            "last_cycle_time": self._last_cycle_time,
            "total_history_entries": len(self._loop_history),
            "field_stats_summary": {
                "occupied": stats.get("occupied_nodes", 0),
                "total": stats.get("total_nodes", 0),
                "avg_activation": round(stats.get("avg_activation", 0), 3),
                "bridge_nodes": stats.get("bridge_nodes", 0),
            },
            "feedback_summary": fb_stats,
            "self_organize_summary": {
                "active_clusters": so_stats.get("active_clusters", 0),
                "active_shortcuts": so_stats.get("active_shortcuts", 0),
            }
            if so_stats
            else {},
            "recent_cycles": [
                {
                    "cycle": c.get("cycle"),
                    "duration": c.get("duration_seconds"),
                    "quality": c.get("phases", {})
                    .get("LEARN", {})
                    .get("quality_score"),
                }
                for c in self._loop_history[-5:]
            ],
        }

    def get_proactive_suggestions(
        self, agent_context: str = ""
    ) -> List[Dict[str, Any]]:
        field = self._field
        suggestions: List[Dict[str, Any]] = []

        with field._lock:
            isolated = field.detect_isolated()
            high_weight_isolated = []
            for nid in isolated:
                node = field._nodes.get(nid)
                if node and node.is_occupied and node.weight >= 2.0:
                    high_weight_isolated.append(
                        {
                            "id": nid[:12],
                            "weight": round(node.weight, 2),
                            "labels": [
                                l
                                for l in node.labels
                                if not l.startswith("__")
                            ],
                        }
                    )

            if high_weight_isolated:
                suggestions.append(
                    {
                        "type": "bridge_concept",
                        "priority": "high",
                        "description": f"{len(high_weight_isolated)} high-weight memories are isolated. They need connections to benefit from association pathways.",
                        "affected_ids": [h["id"] for h in high_weight_isolated[:5]],
                    }
                )

            domain_weights: Dict[str, List[float]] = defaultdict(list)
            domain_negative: Dict[str, int] = defaultdict(int)
            if hasattr(field, "_feedback") and field._feedback is not None:
                fb_stats = field._feedback.get_stats()
                insights = field._feedback.get_learning_insights()
                for ins in insights:
                    if ins.get("insight") == "highly_effective":
                        domain_weights["effective"].append(
                            float(ins.get("positive_count", 0))
                        )
                    neg = ins.get("negative_count", 0)
                    if neg > ins.get("positive_count", 0):
                        suggestions.append(
                            {
                                "type": "avoid_pattern",
                                "priority": "medium",
                                "description": f"Node {ins.get('node_id')} has {neg} negative outcomes vs {ins.get('positive_count', 0)} positive. Pattern associated with this memory consistently fails.",
                                "node_id": ins["node_id"],
                                "negative_count": neg,
                            }
                        )

            if field._self_organize is not None:
                so_stats = field._self_organize.stats()
                clusters = so_stats.get("active_clusters", 0)
                if clusters > 0:
                    weak_clusters = []
                    for cl in field._self_organize._clusters.values() if hasattr(field._self_organize, '_clusters') else []:
                        if hasattr(cl, 'quality_score') and cl.quality_score < 0.3:
                            weak_clusters.append(
                                {
                                    "id": cl.cluster_id[:12],
                                    "quality": round(cl.quality_score, 3),
                                    "node_count": len(cl.node_ids),
                                }
                            )
                    if weak_clusters:
                        domains_str = ", ".join(
                            c["id"] for c in weak_clusters[:3]
                        )
                        suggestions.append(
                            {
                                "type": "explore_domain",
                                "priority": "medium",
                                "description": f"{len(weak_clusters)} clusters have low cohesion (< 0.3). Adding more memories in these domains would strengthen them: {domains_str}",
                                "weak_clusters": weak_clusters[:5],
                            }
                        )

            dreams = []
            if hasattr(field, "dream_history"):
                dreams = field.dream_history(5)
            if dreams:
                cross_domain_dreams = [
                    d
                    for d in dreams
                    if d.get("cross_domain", 0) > 0
                    or d.get("depth_levels", {}).get("L1_cross_domain", 0) > 0
                ]
                if cross_domain_dreams:
                    suggestions.append(
                        {
                            "type": "use_insight",
                            "priority": "high",
                            "description": f"{len(cross_domain_dreams)} recent dream cycles produced cross-domain insights. These represent novel connections the agent can leverage.",
                            "dream_count": len(cross_domain_dreams),
                        }
                    )

            occupied = [
                (nid, n)
                for nid, n in field._nodes.items()
                if n.is_occupied
            ]
            if len(occupied) >= 2:
                high_success_pairs = []
                if field._hebbian is not None:
                    top_paths = []
                    if hasattr(field._hebbian, "_edges"):
                        sorted_edges = sorted(
                            field._hebbian._edges.items(),
                            key=lambda x: -x[1].get("weight", 0)
                            if isinstance(x[1], dict)
                            else -x[1],
                        )
                        for (n1, n2), w in sorted_edges[:5]:
                            nn1 = field._nodes.get(n1)
                            nn2 = field._nodes.get(n2)
                            if nn1 and nn2 and nn1.is_occupied and nn2.is_occupied:
                                high_success_pairs.append(
                                    {
                                        "source": n1[:12],
                                        "target": n2[:12],
                                        "strength": round(
                                            float(
                                                w.get("weight", w)
                                                if isinstance(w, dict)
                                                else w
                                            ),
                                            3,
                                        ),
                                    }
                                )
                if high_success_pairs:
                    suggestions.append(
                        {
                            "type": "reinforce_path",
                            "priority": "low",
                            "description": f"{len(high_success_pairs)} pathways have high Hebbian strength. Using these paths more often will crystallize them into permanent fast-paths.",
                            "top_paths": high_success_pairs[:3],
                        }
                    )

            if agent_context:
                relevant = field.query(agent_context, k=3)
                if relevant:
                    top = relevant[0]
                    if top.get("weight", 0) >= 3.0:
                        suggestions.append(
                            {
                                "type": "reinforce_path",
                                "priority": "medium",
                                "description": f"Context '{agent_context[:50]}' maps to a high-weight memory (w={top['weight']:.1f}). This is a strong knowledge area the agent should leverage.",
                                "memory_id": top["id"][:12],
                                "weight": round(top["weight"], 2),
                            }
                        )

        suggestions.sort(
            key=lambda s: (
                0
                if s.get("priority") == "high"
                else (1 if s.get("priority") == "medium" else 2)
            )
        )
        return suggestions

    def run_targeted_cycle(self, target_domain: str) -> Dict[str, Any]:
        field = self._field
        cycle_start = time.time()
        report: Dict[str, Any] = {
            "type": "targeted",
            "target_domain": target_domain,
            "cycle": self._evolution_metrics["total_cycles"] + 1,
            "start_time": cycle_start,
            "phases": {},
        }

        with field._lock:
            metrics_before = self._snapshot_metrics()
            report["metrics_before"] = metrics_before

            domain_nodes = []
            domain_labels = set()
            for nid, node in field._nodes.items():
                if not node.is_occupied:
                    continue
                if target_domain in node.labels:
                    domain_nodes.append((nid, node))
                    for l in node.labels:
                        if not l.startswith("__"):
                            domain_labels.add(l)

            report["phases"]["SENSE"] = {
                "target_domain": target_domain,
                "domain_node_count": len(domain_nodes),
                "domain_labels": sorted(domain_labels),
                "top_by_weight": [
                    {
                        "id": nid[:12],
                        "weight": round(n.weight, 2),
                        "activation": round(n.activation, 3),
                    }
                    for nid, n in sorted(domain_nodes, key=lambda x: -x[1].weight)[:10]
                ],
            }

            avg_weight = 0.0
            avg_activation = 0.0
            isolated_count = 0
            if domain_nodes:
                avg_weight = sum(n.weight for _, n in domain_nodes) / len(domain_nodes)
                avg_activation = sum(n.activation for _, n in domain_nodes) / len(domain_nodes)
                for nid, node in domain_nodes:
                    has_neighbor = False
                    for fnid in node.face_neighbors:
                        fn = field._nodes.get(fnid)
                        if fn and fn.is_occupied:
                            has_neighbor = True
                            break
                    if not has_neighbor:
                        isolated_count += 1

            analyze: Dict[str, Any] = {
                "domain": target_domain,
                "avg_weight": round(avg_weight, 3),
                "avg_activation": round(avg_activation, 4),
                "isolated_count": isolated_count,
                "issues": [],
            }
            if avg_weight < 0.5:
                analyze["issues"].append("low_average_weight")
            if avg_activation < 0.01:
                analyze["issues"].append("low_activation")
            if isolated_count > len(domain_nodes) * 0.5 and domain_nodes:
                analyze["issues"].append("high_isolation")
            report["phases"]["ANALYZE"] = analyze

            actions_taken: List[str] = []
            for issue in analyze["issues"]:
                if issue == "low_activation":
                    for nid, node in domain_nodes[:20]:
                        node.activation = min(1.0, node.activation + 0.1)
                    actions_taken.append("boosted_activation")
                elif issue == "high_isolation":
                    for nid, node in domain_nodes[:5]:
                        bridge_labels = [l for l in node.labels if not l.startswith("__")][:3]
                        bridge_content = (
                            f"[domain-bridge:{target_domain}] Connecting "
                            f"{node.content[:50]}"
                        )
                        field.store(
                            content=bridge_content,
                            labels=["__domain_bridge__", target_domain] + bridge_labels,
                            weight=max(0.3, node.weight * 0.3),
                        )
                    actions_taken.append("created_domain_bridges")
            report["phases"]["ACT"] = {"actions_taken": actions_taken}

            metrics_after = self._snapshot_metrics()
            delta_occupied = metrics_after["occupied_nodes"] - metrics_before["occupied_nodes"]
            delta_activation = metrics_after["avg_activation"] - metrics_before["avg_activation"]
            report["phases"]["LEARN"] = {
                "delta_occupied": delta_occupied,
                "delta_activation": round(delta_activation, 4),
                "domain_health_before": round(avg_weight * avg_activation, 4),
                "domain_health_after": round(
                    metrics_after.get("avg_activation", 0) * avg_weight, 4
                ),
            }

        cycle_end = time.time()
        report["duration_seconds"] = round(cycle_end - cycle_start, 3)
        report["status"] = "ok"

        self._evolution_metrics["total_cycles"] += 1
        self._last_cycle_time = cycle_end

        self._loop_history.append(report)
        if len(self._loop_history) > self._max_history:
            self._loop_history = self._loop_history[-self._max_history // 2:]

        return report

    def get_domain_health(self, domain_label: str) -> Dict[str, Any]:
        field = self._field
        with field._lock:
            nodes = []
            for nid, node in field._nodes.items():
                if node.is_occupied and domain_label in node.labels:
                    nodes.append((nid, node))

            if not nodes:
                return {
                    "domain": domain_label,
                    "status": "empty",
                    "node_count": 0,
                }

            weights = [n.weight for _, n in nodes]
            activations = [n.activation for _, n in nodes]
            isolated = 0
            for nid, node in nodes:
                has_neighbor = False
                for fnid in node.face_neighbors:
                    fn = field._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    isolated += 1

            all_labels: Dict[str, int] = defaultdict(int)
            for _, node in nodes:
                for l in node.labels:
                    if not l.startswith("__"):
                        all_labels[l] += 1

            health_score = 0.5
            avg_w = sum(weights) / len(weights)
            avg_a = sum(activations) / len(activations)
            iso_rate = isolated / max(1, len(nodes))
            health_score += min(0.2, avg_w * 0.1)
            health_score += min(0.15, avg_a * 0.3)
            health_score -= min(0.3, iso_rate * 0.3)
            health_score = max(0.0, min(1.0, health_score))

            return {
                "domain": domain_label,
                "status": "healthy" if health_score >= 0.6 else ("degraded" if health_score >= 0.3 else "critical"),
                "health_score": round(health_score, 3),
                "node_count": len(nodes),
                "avg_weight": round(avg_w, 3),
                "max_weight": round(max(weights), 2),
                "avg_activation": round(avg_a, 4),
                "isolated_count": isolated,
                "isolation_rate": round(iso_rate, 3),
                "cross_labels": dict(sorted(all_labels.items(), key=lambda x: -x[1])[:10]),
            }

    def propose_memory_merge(self, threshold: float = 0.8) -> List[Dict[str, Any]]:
        field = self._field
        proposals: List[Dict[str, Any]] = []

        with field._lock:
            occupied = [
                (nid, n) for nid, n in field._nodes.items() if n.is_occupied
            ]

            if len(occupied) < 2:
                return proposals

            label_groups: Dict[str, List] = defaultdict(list)
            for nid, node in occupied:
                for l in node.labels:
                    if not l.startswith("__"):
                        label_groups[l].append((nid, node))

            checked: set = set()
            for label, group in label_groups.items():
                if len(group) < 2:
                    continue
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        nid_a, node_a = group[i]
                        nid_b, node_b = group[j]
                        pair_key = tuple(sorted([nid_a, nid_b]))
                        if pair_key in checked:
                            continue
                        checked.add(pair_key)

                        content_sim = 0.0
                        ca = node_a.content or ""
                        cb = node_b.content or ""
                        if ca and cb:
                            words_a = set(ca.lower().split())
                            words_b = set(cb.lower().split())
                            if words_a or words_b:
                                content_sim = len(words_a & words_b) / max(
                                    1, len(words_a | words_b)
                                )

                        weight_sim = 1.0 - abs(node_a.weight - node_b.weight) / max(
                            0.01, max(node_a.weight, node_b.weight)
                        )

                        similarity = content_sim * 0.7 + weight_sim * 0.3

                        if similarity >= threshold:
                            proposals.append({
                                "node_a": nid_a[:12],
                                "node_b": nid_b[:12],
                                "similarity": round(similarity, 3),
                                "content_overlap": round(content_sim, 3),
                                "weight_similarity": round(weight_sim, 3),
                                "shared_labels": sorted(
                                    set(node_a.labels) & set(node_b.labels)
                                    - {"__pulse_bridge__", "__low_priority__"}
                                ),
                                "content_preview_a": (node_a.content or "")[:80],
                                "content_preview_b": (node_b.content or "")[:80],
                            })

        proposals.sort(key=lambda p: -p["similarity"])
        return proposals

    def auto_optimize(self) -> Dict[str, Any]:
        field = self._field
        result: Dict[str, Any] = {
            "self_organize": None,
            "dream": None,
            "cascade": None,
            "actions_taken": [],
            "timestamp": time.time(),
        }

        with field._lock:
            stats = field.stats()
            occupied = stats.get("occupied_nodes", 0)
            avg_activation = stats.get("avg_activation", 0)
            bridge_nodes = stats.get("bridge_nodes", 0)
            isolated_count = len(field.detect_isolated())

        iso_rate = isolated_count / max(1, occupied)
        should_organize = (
            iso_rate > 0.3
            or occupied > 50
            or (self._evolution_metrics["total_cycles"] % 5 == 0)
        )

        if should_organize:
            try:
                so_result = field.run_self_organize()
                if isinstance(so_result, dict):
                    result["self_organize"] = {
                        "clusters_found": so_result.get("clusters_found", 0),
                        "shortcuts_created": so_result.get("shortcuts_created", 0),
                    }
                else:
                    result["self_organize"] = {"raw": str(so_result)}
                result["actions_taken"].append("self_organize")
            except Exception as e:
                result["self_organize"] = {"error": str(e)}

        should_dream = (
            avg_activation < 0.05
            or (self._evolution_metrics["total_cycles"] % 3 == 0)
            or bridge_nodes < occupied * 0.1
        )

        if should_dream:
            try:
                dream_result = field.run_dream_cycle()
                if isinstance(dream_result, dict):
                    result["dream"] = {
                        "dreams_created": dream_result.get("dreams_created", 0),
                        "cross_domain": dream_result.get("cross_domain", 0),
                    }
                else:
                    result["dream"] = {"raw": str(dream_result)}
                result["actions_taken"].append("dream")
            except Exception as e:
                result["dream"] = {"error": str(e)}

        if occupied > 20 and avg_activation < 0.02:
            try:
                cascade_result = field.trigger_cascade()
                if isinstance(cascade_result, dict):
                    result["cascade"] = cascade_result
                else:
                    result["cascade"] = {"triggered": True}
                result["actions_taken"].append("cascade")
            except Exception as e:
                result["cascade"] = {"error": str(e)}

        result["triggers"] = {
            "should_organize": should_organize,
            "should_dream": should_dream,
            "iso_rate": round(iso_rate, 3),
            "avg_activation": round(avg_activation, 4),
        }

        return result

    def _snapshot_metrics(self) -> Dict[str, Any]:
        field = self._field
        stats = field.stats()
        fb = {}
        if hasattr(field, "_feedback") and field._feedback is not None:
            fb = field._feedback.get_stats()
        return {
            "occupied_nodes": stats.get("occupied_nodes", 0),
            "total_nodes": stats.get("total_nodes", 0),
            "avg_activation": stats.get("avg_activation", 0),
            "bridge_nodes": stats.get("bridge_nodes", 0),
            "cascade_count": stats.get("cascade_count", 0),
            "feedback_total": fb.get("total_feedback", 0),
            "feedback_positive_rate": fb.get("positive_rate", 0),
        }

    def _phase_sense(self, field: "HoneycombNeuralField") -> Dict[str, Any]:
        sense: Dict[str, Any] = {}

        all_occupied = [
            (nid, n)
            for nid, n in field._nodes.items()
            if n.is_occupied
        ]
        sorted_by_weight = sorted(
            all_occupied, key=lambda x: -x[1].weight
        )
        sense["top_memories_by_weight"] = [
            {
                "id": nid[:12],
                "weight": round(n.weight, 2),
                "activation": round(n.activation, 3),
                "labels": [
                    l for l in n.labels if not l.startswith("__")
                ][:5],
            }
            for nid, n in sorted_by_weight[:10]
        ]

        sorted_by_activation = sorted(
            all_occupied, key=lambda x: -x[1].activation
        )
        sense["top_memories_by_activation"] = [
            {
                "id": nid[:12],
                "activation": round(n.activation, 3),
                "weight": round(n.weight, 2),
            }
            for nid, n in sorted_by_activation[:10]
        ]

        fb_stats = {}
        if hasattr(field, "_feedback") and field._feedback is not None:
            fb_stats = field._feedback.get_stats()
        sense["feedback_summary"] = fb_stats

        so_stats = {}
        if field._self_organize is not None:
            so_stats = field._self_organize.stats()
        sense["cluster_state"] = so_stats

        dreams = []
        if hasattr(field, "dream_history"):
            dreams = field.dream_history(3)
        sense["recent_dreams"] = dreams

        sense["total_occupied"] = len(all_occupied)
        return sense

    def _phase_analyze(
        self,
        field: "HoneycombNeuralField",
        sense: Dict[str, Any],
    ) -> Dict[str, Any]:
        analyze: Dict[str, Any] = {
            "low_quality_clusters": [],
            "isolated_high_weight": [],
            "weak_feedback_domains": [],
            "failing_patterns": [],
        }

        so_stats = sense.get("cluster_state", {})
        clusters = so_stats.get("active_clusters", 0)
        if clusters > 0 and hasattr(field._self_organize, "_clusters"):
            for cl in field._self_organize._clusters.values():
                if hasattr(cl, "quality_score") and cl.quality_score < 0.3:
                    analyze["low_quality_clusters"].append(
                        {
                            "id": cl.cluster_id[:12],
                            "quality": round(cl.quality_score, 3),
                            "node_count": len(cl.node_ids),
                            "labels": list(cl.labels)[:5] if hasattr(cl, "labels") else [],
                        }
                    )

        isolated = field.detect_isolated()
        for nid in isolated:
            node = field._nodes.get(nid)
            if node and node.is_occupied and node.weight >= 1.5:
                analyze["isolated_high_weight"].append(
                    {
                        "id": nid[:12],
                        "weight": round(node.weight, 2),
                        "labels": [
                            l for l in node.labels if not l.startswith("__")
                        ][:5],
                    }
                )

        fb_stats = sense.get("feedback_summary", {})
        if fb_stats:
            pos_rate = fb_stats.get("positive_rate", 1.0)
            if pos_rate < 0.3 and fb_stats.get("total_feedback", 0) > 5:
                analyze["weak_feedback_domains"].append(
                    {
                        "positive_rate": round(pos_rate, 3),
                        "total_feedback": fb_stats.get("total_feedback", 0),
                        "note": "Agent decisions have low success rate",
                    }
                )

            if hasattr(field, "_feedback") and field._feedback is not None:
                insights = field._feedback.get_learning_insights()
                for ins in insights:
                    neg = ins.get("negative_count", 0)
                    pos = ins.get("positive_count", 0)
                    if neg > pos and neg >= 2:
                        analyze["failing_patterns"].append(
                            {
                                "node_id": ins["node_id"],
                                "negative_count": neg,
                                "positive_count": pos,
                                "ratio": round(
                                    neg / max(1, pos + neg), 3
                                ),
                            }
                        )

        analyze["issue_count"] = (
            len(analyze["low_quality_clusters"])
            + len(analyze["isolated_high_weight"])
            + len(analyze["weak_feedback_domains"])
            + len(analyze["failing_patterns"])
        )
        return analyze

    def _phase_plan(
        self,
        field: "HoneycombNeuralField",
        analyze: Dict[str, Any],
    ) -> Dict[str, Any]:
        plan: Dict[str, Any] = {"suggestions": [], "planned_actions": []}

        for cluster in analyze.get("low_quality_clusters", []):
            labels = cluster.get("labels", [])
            label_str = ", ".join(labels[:3]) if labels else cluster["id"]
            plan["suggestions"].append(
                {
                    "action": "explore_domain",
                    "target": label_str,
                    "reason": f"Cluster {cluster['id']} has quality {cluster['quality']} with {cluster['node_count']} nodes. Needs more experience.",
                }
            )
            plan["planned_actions"].append(
                {"type": "trigger_dream", "target_labels": labels}
            )

        for iso in analyze.get("isolated_high_weight", [])[:5]:
            plan["suggestions"].append(
                {
                    "action": "connect_isolated",
                    "target": iso["id"],
                    "reason": f"High-weight ({iso['weight']}) memory is isolated. Bridge needed.",
                }
            )
            plan["planned_actions"].append(
                {"type": "create_bridge", "target_id": iso["id"]}
            )

        for weak in analyze.get("weak_feedback_domains", []):
            plan["suggestions"].append(
                {
                    "action": "adjust_strategy",
                    "target": "feedback_loop",
                    "reason": f"Positive feedback rate is {weak['positive_rate']}. Strategy needs adjustment.",
                }
            )

        for fail in analyze.get("failing_patterns", []):
            plan["suggestions"].append(
                {
                    "action": "mark_caution",
                    "target": fail["node_id"],
                    "reason": f"Node has {fail['negative_count']} negative outcomes ({fail['ratio']} failure rate).",
                }
            )
            plan["planned_actions"].append(
                {"type": "mark_caution", "target_id": fail["node_id"]}
            )

        return plan

    def _phase_act(
        self,
        field: "HoneycombNeuralField",
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        act: Dict[str, Any] = {
            "bridges_created": 0,
            "paths_reinforced": 0,
            "caution_zones_marked": 0,
            "dreams_triggered": 0,
        }

        for action in plan.get("planned_actions", []):
            try:
                if action["type"] == "create_bridge":
                    target_id = action["target_id"]
                    node = field._nodes.get(target_id)
                    if node and node.is_occupied:
                        labels = [
                            l
                            for l in node.labels
                            if not l.startswith("__")
                        ][:3]
                        bridge_content = (
                            f"[bridge] Connecting {node.content[:60]} "
                            f"via topology bridge"
                        )
                        field.store(
                            content=bridge_content,
                            labels=["__pulse_bridge__"] + labels,
                            weight=max(0.3, node.weight * 0.3),
                        )
                        act["bridges_created"] += 1

                elif action["type"] == "mark_caution":
                    target_prefix = action["target_id"]
                    for nid, n in field._nodes.items():
                        if nid.startswith(target_prefix) and n.is_occupied:
                            if (
                                "__low_priority__" not in n.labels
                            ):
                                n.labels.append("__low_priority__")
                            act["caution_zones_marked"] += 1
                            break

                elif action["type"] == "trigger_dream":
                    labels = action.get("target_labels", [])
                    if labels:
                        relevant = field.query(
                            " ".join(labels[:2]), k=3, labels=labels
                        )
                        if relevant:
                            act["dreams_triggered"] += 1
            except Exception:
                continue

        try:
            dream_result = field.run_dream_cycle()
            act["dream_result"] = {
                "dreams_created": dream_result.get("dreams_created", 0)
                if isinstance(dream_result, dict)
                else 0,
            }
        except Exception:
            pass

        try:
            so_result = field.run_self_organize()
            act["self_organize_result"] = {
                "clusters_found": so_result.get("clusters_found", 0)
                if isinstance(so_result, dict)
                else 0,
                "shortcuts_created": so_result.get(
                    "shortcuts_created", 0
                )
                if isinstance(so_result, dict)
                else 0,
            }
        except Exception:
            pass

        return act

    def _phase_learn(
        self,
        field: "HoneycombNeuralField",
        metrics_before: Dict[str, Any],
    ) -> Dict[str, Any]:
        learn: Dict[str, Any] = {}

        stats_after = field.stats()
        fb_after = {}
        if hasattr(field, "_feedback") and field._feedback is not None:
            fb_after = field._feedback.get_stats()

        delta: Dict[str, Any] = {
            "occupied_delta": stats_after.get("occupied_nodes", 0)
            - metrics_before.get("occupied_nodes", 0),
            "activation_delta": round(
                stats_after.get("avg_activation", 0)
                - metrics_before.get("avg_activation", 0),
                4,
            ),
            "bridge_delta": stats_after.get("bridge_nodes", 0)
            - metrics_before.get("bridge_nodes", 0),
            "cascade_delta": stats_after.get("cascade_count", 0)
            - metrics_before.get("cascade_count", 0),
        }
        learn["delta"] = delta

        quality = 0.5
        occupied_growth = max(0, delta["occupied_delta"])
        activation_growth = max(0, delta["activation_delta"])
        bridge_growth = max(0, delta["bridge_delta"])
        quality += min(0.2, occupied_growth * 0.02)
        quality += min(0.15, activation_growth * 0.3)
        quality += min(0.15, bridge_growth * 0.05)

        fb_pos_rate = fb_after.get("positive_rate", 0.5)
        quality = quality * 0.6 + fb_pos_rate * 0.4
        quality = max(0.0, min(1.0, quality))
        learn["quality_score"] = round(quality, 3)

        self._quality_samples.append(quality)
        if len(self._quality_samples) > 100:
            self._quality_samples = self._quality_samples[-50:]

        if self._quality_samples:
            avg_quality = sum(self._quality_samples) / len(
                self._quality_samples
            )
            self._evolution_metrics["avg_decision_quality"] = round(
                avg_quality, 3
            )

        pos_rate = fb_after.get("positive_rate", 0.0)
        self._evolution_metrics["positive_feedback_rate"] = round(
            pos_rate, 3
        )

        learn["evolution_metrics"] = dict(self._evolution_metrics)
        return learn
