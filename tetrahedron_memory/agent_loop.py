"""
Agent Memory-Driven Closed Loop — v8.0 Dark Plane + Observer + Regulation Coupled.

The self-evolution engine now integrates with:
- Dark Plane Substrate (H0~H6, PE, coherence, phase transitions)
- Void Channels (cross-domain topological shortcuts)
- Self-Regulation (hormone state, autonomic mode, circadian phase)
- RuntimeObserver (trajectory injection for self-awareness)

Phases:
  SENSE   → field metrics + dark plane state + hormone profile + observer awareness
  ANALYZE → phase transition detection + hormone-influenced risk assessment + void channel mapping
  PLAN    → hormone-modulated strategy (explore vs conserve) + void channel shortcut routing
  ACT     → dark-plane-aware bridge placement + phase-triggered regulation/dream
  LEARN   → PE delta + dark plane integration + observer trajectory feedback
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional

_log = logging.getLogger("tetramem.agent_loop")

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField


_HORMONE_PROFILES = {
    "explorative": {
        "dopamine": 0.6, "cortisol": 0.1, "serotonin": 0.3, "acetylcholine": 0.5,
    },
    "conservative": {
        "dopamine": 0.2, "cortisol": 0.6, "serotonin": 0.5, "acetylcholine": 0.3,
    },
    "balanced": {
        "dopamine": 0.4, "cortisol": 0.3, "serotonin": 0.4, "acetylcholine": 0.4,
    },
    "creative": {
        "dopamine": 0.5, "cortisol": 0.2, "serotonin": 0.3, "acetylcholine": 0.6,
    },
}


class AgentMemoryLoop:

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
            "dark_plane_couplings": 0,
            "void_channel_navigations": 0,
            "phase_transitions_observed": 0,
            "observer_trajectories_injected": 0,
        }
        self._last_cycle_time: float = 0.0
        self._cycle_count_since_report: int = 0
        self._quality_samples: List[float] = []
        self._pe_history: deque = deque(maxlen=20)
        self._coherence_history: deque = deque(maxlen=20)
        self._last_regulation_status: Optional[Dict] = None
        self._last_substrate_stats: Optional[Dict] = None

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

            act = self._phase_act(field, plan, sense)
            report["phases"]["ACT"] = act

            learn = self._phase_learn(field, metrics_before, sense, analyze)
            report["phases"]["LEARN"] = learn

        self._inject_trajectory(field, report)

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
            dp_stats = {}
            if hasattr(field, "_dark_substrate") and field._dark_substrate is not None:
                dp_stats = field._dark_substrate.get_stats()
            vc_stats = {}
            if hasattr(field, "_void_channel") and field._void_channel is not None:
                vc_stats = field._void_channel.get_stats()
            reg_status = {}
            if hasattr(field, "_regulation") and field._regulation is not None:
                reg_status = field._regulation.status()
            obs_stats = {}
            if hasattr(field, "_observer") and field._observer is not None:
                obs_stats = field._observer.get_stats()

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
            "dark_plane_summary": {
                "pe": dp_stats.get("persistent_entropy", 0),
                "coherence": dp_stats.get("coherence", 0),
                "void_energy": dp_stats.get("void_energy", 0),
                "phase_transitions": dp_stats.get("total_phase_transitions", 0),
                "h4_growth": dp_stats.get("h4", {}).get("growth_rate", 0),
                "h5_regulation": dp_stats.get("h5_regulation", 0),
                "h6_cascade": dp_stats.get("h6", {}).get("count", 0),
            } if dp_stats else {},
            "void_channel_summary": {
                "active": vc_stats.get("active_channels", 0),
                "by_dim": vc_stats.get("by_dimension", {}),
            } if vc_stats else {},
            "regulation_summary": {
                "mode": reg_status.get("autonomic", {}).get("mode"),
                "hormones": reg_status.get("hormones", {}),
                "stress": reg_status.get("stress", {}).get("level", 0),
                "circadian": reg_status.get("circadian", {}).get("phase"),
            } if reg_status else {},
            "observer_summary": {
                "enabled": obs_stats.get("enabled", False),
                "memories_stored": obs_stats.get("memories_stored", 0),
            } if obs_stats else {},
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
                    "strategy": c.get("phases", {})
                    .get("PLAN", {})
                    .get("strategy"),
                }
                for c in self._loop_history[-5:]
            ],
            "pe_trend": list(self._pe_history)[-5:],
            "coherence_trend": list(self._coherence_history)[-5:],
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

            if hasattr(field, "_dark_substrate") and field._dark_substrate is not None:
                substrate = field._dark_substrate
                pe = substrate._state.persistent_entropy
                coherence = substrate._state.coherence
                last_pt = substrate._state.last_phase_transition

                if coherence < 0.3:
                    suggestions.append({
                        "type": "low_coherence",
                        "priority": "high",
                        "description": f"Dark plane coherence is {coherence:.3f} (< 0.3). System topology is fragmented. Consider triggering self-organization to restore connectivity.",
                        "coherence": round(coherence, 3),
                    })

                if last_pt and time.time() - last_pt < 300:
                    suggestions.append({
                        "type": "recent_phase_transition",
                        "priority": "high",
                        "description": f"A phase transition occurred recently. The dark plane is restructuring — dream cycle may reveal new cross-domain insights.",
                        "transition_level": substrate._state.total_phase_transitions,
                    })

                if pe > 2.5:
                    suggestions.append({
                        "type": "high_entropy",
                        "priority": "medium",
                        "description": f"Persistent entropy is {pe:.3f} — high topological complexity. Multiple persistent structures are competing. Dream cycle recommended for integration.",
                        "persistent_entropy": round(pe, 3),
                    })

                vc = field._void_channel
                if vc is not None:
                    vc_stats = vc.get_stats()
                    if vc_stats.get("active_channels", 0) > 0:
                        dim3_channels = vc_stats.get("by_dimension", {}).get(3, 0)
                        if dim3_channels > 0:
                            suggestions.append({
                                "type": "void_shortcuts_available",
                                "priority": "high",
                                "description": f"{dim3_channels} dim-3 void channels exist — these are topological shortcuts between distant domains. Navigate via these for faster cross-domain reasoning.",
                                "dim3_count": dim3_channels,
                            })

            if hasattr(field, "_regulation") and field._regulation is not None:
                reg = field._regulation.status()
                stress = reg.get("stress", {}).get("level", 0)
                mode = reg.get("autonomic", {}).get("mode", "balanced")
                hormones = reg.get("hormones", {})

                if stress > 0.7:
                    suggestions.append({
                        "type": "high_stress",
                        "priority": "high",
                        "description": f"System stress is {stress:.3f}. Autonomic mode: {mode}. Consider conservative strategy — avoid triggering expensive operations.",
                        "stress_level": round(stress, 3),
                        "mode": mode,
                    })

                dopamine = hormones.get("dopamine", 0)
                if dopamine > 0.6:
                    suggestions.append({
                        "type": "exploration_window",
                        "priority": "medium",
                        "description": f"Dopamine is elevated ({dopamine:.3f}). Good window for creative exploration — dream cycle and cross-domain queries are favored.",
                        "dopamine": round(dopamine, 3),
                    })

            if hasattr(field, "_feedback") and field._feedback is not None:
                insights = field._feedback.get_learning_insights()
                for ins in insights:
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
                    for cl in field._self_organize._clusters if hasattr(field._self_organize, '_clusters') else []:
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

            dark_plane_context = {}
            if hasattr(field, "_dark_substrate") and field._dark_substrate is not None:
                sub = field._dark_substrate
                domain_in_dark = False
                for nid, _ in domain_nodes[:20]:
                    channels = []
                    if hasattr(field, "_void_channel") and field._void_channel is not None:
                        channels = field._void_channel.get_channels_for_node(nid)
                    if channels:
                        domain_in_dark = True
                        break
                dark_plane_context = {
                    "has_void_channels": domain_in_dark,
                    "pe": sub._state.persistent_entropy,
                    "coherence": sub._state.coherence,
                }

            report["phases"]["SENSE"] = {
                "target_domain": target_domain,
                "domain_node_count": len(domain_nodes),
                "domain_labels": sorted(domain_labels),
                "dark_plane_context": dark_plane_context,
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

            void_channel_count = 0
            if hasattr(field, "_void_channel") and field._void_channel is not None:
                for nid, _ in nodes[:30]:
                    void_channel_count += len(field._void_channel.get_channels_for_node(nid))

            health_score = 0.5
            avg_w = sum(weights) / len(weights)
            avg_a = sum(activations) / len(activations)
            iso_rate = isolated / max(1, len(nodes))
            health_score += min(0.2, avg_w * 0.1)
            health_score += min(0.15, avg_a * 0.3)
            health_score -= min(0.3, iso_rate * 0.3)
            health_score += min(0.1, void_channel_count * 0.02)
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
                "void_channel_count": void_channel_count,
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
            "regulation_trigger": None,
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

        should_regulate = False
        if hasattr(field, "_regulation") and field._regulation is not None:
            reg = field._regulation.status()
            stress = reg.get("stress", {}).get("level", 0)
            if stress > 0.8:
                should_regulate = True

        should_organize = (
            iso_rate > 0.3
            or occupied > 50
            or (self._evolution_metrics["total_cycles"] % 5 == 0)
        )

        if hasattr(field, "_dark_substrate") and field._dark_substrate is not None:
            sub = field._dark_substrate
            coherence = sub._state.coherence
            if coherence < 0.2:
                should_organize = True

        should_dream = (
            avg_activation < 0.05
            or (self._evolution_metrics["total_cycles"] % 3 == 0)
            or bridge_nodes < occupied * 0.1
        )

        if hasattr(field, "_dark_substrate") and field._dark_substrate is not None:
            sub = field._dark_substrate
            pe = sub._state.persistent_entropy
            if pe > 2.0:
                should_dream = True

        if should_regulate:
            try:
                reg_result = field._regulation.regulate()
                result["regulation_trigger"] = {
                    "stress_before": field._regulation.status().get("stress", {}).get("level", 0),
                    "actions": reg_result if isinstance(reg_result, list) else [],
                }
                result["actions_taken"].append("regulation")
            except Exception as e:
                _log.error("regulation trigger failed: %s", e)

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
                _log.error("self_organize failed: %s", e)
                result["self_organize"] = {"error": str(e)}

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
                _log.error("dream cycle failed: %s", e)
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
                _log.error("cascade trigger failed: %s", e)
                result["cascade"] = {"error": str(e)}

        result["triggers"] = {
            "should_organize": should_organize,
            "should_dream": should_dream,
            "should_regulate": should_regulate,
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
        dp = {}
        if hasattr(field, "_dark_substrate") and field._dark_substrate is not None:
            sub = field._dark_substrate
            dp = {
                "pe": sub._state.persistent_entropy,
                "coherence": sub._state.coherence,
                "void_energy": sub._state.void_energy,
                "h4_growth": sub._state.h4.growth_rate,
            }
        return {
            "occupied_nodes": stats.get("occupied_nodes", 0),
            "total_nodes": stats.get("total_nodes", 0),
            "avg_activation": stats.get("avg_activation", 0),
            "bridge_nodes": stats.get("bridge_nodes", 0),
            "cascade_count": stats.get("cascade_count", 0),
            "feedback_total": fb.get("total_feedback", 0),
            "feedback_positive_rate": fb.get("positive_rate", 0),
            "dark_plane": dp,
        }

    def _sense_dark_plane(self, field: "HoneycombNeuralField") -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "available": False,
            "pe": 0.0, "coherence": 0.0, "void_energy": 0.0,
            "dark_energy": 0.0, "channel_energy": 0.0,
            "h3": {}, "h4": {}, "h5": {}, "h6": {},
            "phase_transitions": 0, "last_transition": None,
            "cascade_potential": 0.0, "psi_field": 0.0,
            "cross_dim_coupling": {},
        }
        if not hasattr(field, "_dark_substrate") or field._dark_substrate is None:
            return result
        sub = field._dark_substrate
        result["available"] = True
        result["pe"] = sub._state.persistent_entropy
        result["coherence"] = sub._state.coherence
        result["void_energy"] = sub._state.void_energy
        result["dark_energy"] = sub._state.dark_energy
        result["channel_energy"] = sub._state.channel_energy
        result["h3"] = {"count": sub._state.h3.count, "energy": sub._state.h3.energy, "growth": sub._state.h3.growth_rate}
        result["h4"] = {"count": sub._state.h4.count, "energy": sub._state.h4.energy, "growth": sub._state.h4.growth_rate}
        result["h5"] = {"count": sub._state.h5.count, "energy": sub._state.h5.energy, "growth": sub._state.h5.growth_rate}
        result["h6"] = {"count": sub._state.h6.count, "energy": sub._state.h6.energy, "growth": sub._state.h6.growth_rate}
        result["phase_transitions"] = sub._state.total_phase_transitions
        result["last_transition"] = sub._state.last_phase_transition
        result["cascade_potential"] = sub._state.cascade_potential
        result["psi_field"] = sub._psi_field
        result["cross_dim_coupling"] = sub._compute_cross_dim_coupling()
        self._pe_history.append(result["pe"])
        self._coherence_history.append(result["coherence"])
        self._last_substrate_stats = result
        return result

    def _sense_regulation(self, field: "HoneycombNeuralField") -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "available": False,
            "mode": "balanced", "hormones": {}, "stress": 0.0,
            "circadian_phase": "work", "emergency": False,
        }
        if not hasattr(field, "_regulation") or field._regulation is None:
            return result
        reg = field._regulation.status()
        result["available"] = True
        result["mode"] = reg.get("autonomic", {}).get("mode", "balanced")
        result["hormones"] = reg.get("hormones", {})
        result["stress"] = reg.get("stress", {}).get("level", 0)
        result["circadian_phase"] = reg.get("circadian", {}).get("phase", "work")
        result["emergency"] = reg.get("stress", {}).get("emergency_mode", False)
        self._last_regulation_status = result
        return result

    def _sense_observer(self, field: "HoneycombNeuralField") -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "available": False,
            "enabled": False, "memories_stored": 0,
            "total_events": 0, "dropped_loop": 0,
        }
        if not hasattr(field, "_observer") or field._observer is None:
            return result
        obs = field._observer.get_stats()
        result["available"] = True
        result["enabled"] = obs.get("enabled", False)
        result["memories_stored"] = obs.get("memories_stored", 0)
        result["total_events"] = obs.get("total_events_received", 0)
        result["dropped_loop"] = obs.get("events_dropped_loop", 0)
        return result

    def _sense_void_channels(self, field: "HoneycombNeuralField") -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "available": False,
            "active_channels": 0, "by_dimension": {1: 0, 2: 0, 3: 0},
            "avg_strength": 0.0, "total_coupling": 0.0,
        }
        if not hasattr(field, "_void_channel") or field._void_channel is None:
            return result
        vc = field._void_channel
        stats = vc.get_stats()
        result["available"] = True
        result["active_channels"] = stats.get("active_channels", 0)
        result["by_dimension"] = stats.get("by_dimension", {1: 0, 2: 0, 3: 0})
        result["avg_strength"] = stats.get("avg_strength", 0)
        result["total_coupling"] = stats.get("total_energy_coupling", 0)
        return result

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

        sense["dark_plane"] = self._sense_dark_plane(field)
        sense["regulation"] = self._sense_regulation(field)
        sense["observer"] = self._sense_observer(field)
        sense["void_channels"] = self._sense_void_channels(field)

        return sense

    def _determine_strategy(self, regulation: Dict, dark_plane: Dict) -> str:
        if not regulation.get("available"):
            return "balanced"

        hormones = regulation.get("hormones", {})
        stress = regulation.get("stress", 0)
        dopamine = hormones.get("dopamine", 0.4)
        cortisol = hormones.get("cortisol", 0.3)
        acetylcholine = hormones.get("acetylcholine", 0.4)

        if stress > 0.7 or cortisol > 0.6:
            return "conservative"

        coherence = dark_plane.get("coherence", 0.5)
        pe = dark_plane.get("pe", 0)

        if dopamine > 0.5 and acetylcholine > 0.5 and coherence > 0.4:
            return "creative"

        if dopamine > 0.5 and pe > 1.5:
            return "explorative"

        if regulation.get("circadian_phase") == "consolidation":
            return "conservative"

        return "balanced"

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
            "dark_plane_issues": [],
            "hormone_state": {},
            "strategy": "balanced",
        }

        so_stats = sense.get("cluster_state", {})
        clusters = so_stats.get("active_clusters", 0)
        if clusters > 0 and hasattr(field._self_organize, "_clusters"):
            for cl in field._self_organize._clusters:
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

        dp = sense.get("dark_plane", {})
        if dp.get("available"):
            if dp.get("coherence", 1) < 0.3:
                analyze["dark_plane_issues"].append({
                    "issue": "low_coherence",
                    "value": round(dp["coherence"], 3),
                    "impact": "topological structure is fragmented",
                })
            if dp.get("pe", 0) > 2.5:
                analyze["dark_plane_issues"].append({
                    "issue": "high_entropy",
                    "value": round(dp["pe"], 3),
                    "impact": "persistent structures competing",
                })
            h4_growth = dp.get("h4", {}).get("growth", 0)
            if h4_growth > 0.1:
                analyze["dark_plane_issues"].append({
                    "issue": "h4_emerging",
                    "value": round(h4_growth, 3),
                    "impact": "many-body entanglement emerging — may indicate phase transition",
                })
            if dp.get("cascade_potential", 0) > 0.5:
                analyze["dark_plane_issues"].append({
                    "issue": "cascade_imminent",
                    "value": round(dp["cascade_potential"], 3),
                    "impact": "dark plane energy concentrated enough for cascade",
                })

        reg = sense.get("regulation", {})
        if reg.get("available"):
            analyze["hormone_state"] = {
                "mode": reg["mode"],
                "stress": reg["stress"],
                "circadian": reg["circadian_phase"],
                "emergency": reg["emergency"],
                "hormones": reg["hormones"],
            }

        analyze["strategy"] = self._determine_strategy(reg, dp)
        analyze["void_channels"] = sense.get("void_channels", {})

        analyze["issue_count"] = (
            len(analyze["low_quality_clusters"])
            + len(analyze["isolated_high_weight"])
            + len(analyze["weak_feedback_domains"])
            + len(analyze["failing_patterns"])
            + len(analyze["dark_plane_issues"])
        )
        return analyze

    def _phase_plan(
        self,
        field: "HoneycombNeuralField",
        analyze: Dict[str, Any],
    ) -> Dict[str, Any]:
        strategy = analyze.get("strategy", "balanced")
        plan: Dict[str, Any] = {"suggestions": [], "planned_actions": [], "strategy": strategy}
        profile = _HORMONE_PROFILES.get(strategy, _HORMONE_PROFILES["balanced"])

        explore_weight = profile["dopamine"]
        conserve_weight = profile["cortisol"]
        creativity_weight = profile["acetylcholine"]

        for cluster in analyze.get("low_quality_clusters", []):
            labels = cluster.get("labels", [])
            label_str = ", ".join(labels[:3]) if labels else cluster["id"]
            plan["suggestions"].append(
                {
                    "action": "explore_domain",
                    "target": label_str,
                    "reason": f"Cluster {cluster['id']} quality={cluster['quality']} ({cluster['node_count']} nodes). Strategy={strategy}.",
                }
            )
            if explore_weight > 0.3:
                plan["planned_actions"].append(
                    {"type": "trigger_dream", "target_labels": labels, "depth": "L2" if creativity_weight > 0.5 else "L1"}
                )

        for iso in analyze.get("isolated_high_weight", []):
            plan["suggestions"].append(
                {
                    "action": "connect_isolated",
                    "target": iso["id"],
                    "reason": f"High-weight ({iso['weight']}) memory isolated. Strategy={strategy}.",
                }
            )
            dp_issues = analyze.get("dark_plane_issues", [])
            dark_target = "deep"
            if any(d["issue"] == "low_coherence" for d in dp_issues):
                dark_target = "shallow"
            elif any(d["issue"] == "cascade_imminent" for d in dp_issues):
                dark_target = "abyss"

            plan["planned_actions"].append(
                {"type": "create_bridge", "target_id": iso["id"], "dark_plane_level": dark_target}
            )

        vc = analyze.get("void_channels", {})
        if vc.get("available") and vc.get("active_channels", 0) > 0:
            dim3 = vc.get("by_dimension", {}).get(3, 0)
            if dim3 > 0 and creativity_weight > 0.4:
                plan["suggestions"].append({
                    "action": "leverage_void_channels",
                    "target": f"{dim3} dim-3 channels",
                    "reason": f"Void channels available for cross-domain navigation. Strategy={strategy} favors creative use.",
                })
                plan["planned_actions"].append(
                    {"type": "navigate_void_channels"}
                )

        for weak in analyze.get("weak_feedback_domains", []):
            plan["suggestions"].append(
                {
                    "action": "adjust_strategy",
                    "target": "feedback_loop",
                    "reason": f"Positive rate={weak['positive_rate']}. Strategy adjustment needed.",
                }
            )

        for fail in analyze.get("failing_patterns", []):
            plan["suggestions"].append(
                {
                    "action": "mark_caution",
                    "target": fail["node_id"],
                    "reason": f"Node has {fail['negative_count']} negatives ({fail['ratio']} failure rate).",
                }
            )
            plan["planned_actions"].append(
                {"type": "mark_caution", "target_id": fail["node_id"]}
            )

        for dp_issue in analyze.get("dark_plane_issues", []):
            if dp_issue["issue"] == "low_coherence":
                plan["planned_actions"].append(
                    {"type": "trigger_self_organize", "reason": "low_coherence"}
                )
            elif dp_issue["issue"] == "cascade_imminent" and explore_weight > 0.4:
                plan["planned_actions"].append(
                    {"type": "trigger_cascade", "reason": "cascade_potential"}
                )
            elif dp_issue["issue"] == "h4_emerging":
                plan["planned_actions"].append(
                    {"type": "trigger_dream", "target_labels": [], "depth": "L3", "reason": "h4_emerging"}
                )

        if analyze.get("hormone_state", {}).get("emergency"):
            plan["planned_actions"] = [
                a for a in plan["planned_actions"]
                if a.get("type") in ("mark_caution", "trigger_self_organize")
            ]
            plan["suggestions"] = [
                s for s in plan["suggestions"]
                if s.get("action") in ("mark_caution", "adjust_strategy")
            ]

        return plan

    def _phase_act(
        self,
        field: "HoneycombNeuralField",
        plan: Dict[str, Any],
        sense: Dict[str, Any],
    ) -> Dict[str, Any]:
        act: Dict[str, Any] = {
            "bridges_created": 0,
            "paths_reinforced": 0,
            "caution_zones_marked": 0,
            "dreams_triggered": 0,
            "void_navigations": 0,
            "dark_plane_couplings": 0,
            "regulation_triggers": 0,
            "cascade_triggers": 0,
        }

        dp = sense.get("dark_plane", {})

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
                        dark_level = action.get("dark_plane_level", "deep")
                        bridge_content = (
                            f"[bridge:{dark_level}] Connecting {node.content[:50]} "
                            f"via topology bridge"
                        )
                        bridge_weight = max(0.3, node.weight * 0.3)
                        if dp.get("available") and dp.get("pe", 0) > 1.5:
                            bridge_weight *= 1.0 + 0.1 * min(dp["pe"], 3.0)

                        field.store(
                            content=bridge_content,
                            labels=["__pulse_bridge__", f"__dark_{dark_level}__"] + labels,
                            weight=bridge_weight,
                        )
                        act["bridges_created"] += 1
                        act["dark_plane_couplings"] += 1

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

                elif action["type"] == "trigger_self_organize":
                    so_result = field.run_self_organize()
                    act["self_organize_result"] = {
                        "clusters_found": so_result.get("clusters_found", 0) if isinstance(so_result, dict) else 0,
                    }

                elif action["type"] == "trigger_cascade":
                    cascade_result = field.trigger_cascade()
                    act["cascade_triggers"] += 1
                    act["cascade_result"] = cascade_result if isinstance(cascade_result, dict) else {"triggered": True}

                elif action["type"] == "navigate_void_channels":
                    if hasattr(field, "_void_channel") and field._void_channel is not None:
                        vc = field._void_channel
                        active = [ch for ch in vc._channels if ch.is_active and ch.dimension >= 2]
                        if active:
                            strongest = max(active, key=lambda ch: ch.strength)
                            node_a = field._nodes.get(strongest.node_a)
                            node_b = field._nodes.get(strongest.node_b)
                            if node_a and node_b and node_a.is_occupied and node_b.is_occupied:
                                nav_content = (
                                    f"[void-nav:dim{strongest.dimension}] "
                                    f"{node_a.content[:30]} <-> {node_b.content[:30]}"
                                )
                                nav_labels = list(set(
                                    [l for l in node_a.labels if not l.startswith("__")][:2]
                                    + [l for l in node_b.labels if not l.startswith("__")][:2]
                                ))
                                field.store(
                                    content=nav_content,
                                    labels=["__void_nav__"] + nav_labels,
                                    weight=strongest.strength * 0.5,
                                )
                                act["void_navigations"] += 1
                                self._evolution_metrics["void_channel_navigations"] += 1

            except Exception:
                _log.warning("planned action %s failed, skipping", action.get("type", "unknown"))
                continue

        strategy = plan.get("strategy", "balanced")
        should_dream = strategy in ("explorative", "creative", "balanced")
        should_organize = strategy in ("conservative", "balanced")

        if should_dream:
            try:
                dream_result = field.run_dream_cycle()
                act["dream_result"] = {
                    "dreams_created": dream_result.get("dreams_created", 0)
                    if isinstance(dream_result, dict)
                    else 0,
                }
            except Exception:
                _log.warning("dream cycle in ACT phase failed", exc_info=True)

        if should_organize:
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
                _log.warning("self_organize in ACT phase failed", exc_info=True)

        return act

    def _phase_learn(
        self,
        field: "HoneycombNeuralField",
        metrics_before: Dict[str, Any],
        sense: Dict[str, Any],
        analyze: Dict[str, Any],
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

        dp_before = metrics_before.get("dark_plane", {})
        dp_now = sense.get("dark_plane", {})
        if dp_before.get("pe", 0) > 0 and dp_now.get("pe", 0) > 0:
            pe_delta = dp_now["pe"] - dp_before["pe"]
            if pe_delta < 0:
                quality += min(0.1, abs(pe_delta) * 0.05)
            coherence_delta = dp_now.get("coherence", 0) - dp_before.get("coherence", 0)
            if coherence_delta > 0:
                quality += min(0.1, coherence_delta * 0.1)

        dp_issues = analyze.get("dark_plane_issues", [])
        if dp_issues:
            quality -= min(0.1, len(dp_issues) * 0.02)

        quality = max(0.0, min(1.0, quality))
        learn["quality_score"] = round(quality, 3)

        learn["dark_plane_learning"] = {
            "pe_before": round(dp_before.get("pe", 0), 3),
            "pe_after": round(dp_now.get("pe", 0), 3),
            "pe_delta": round(dp_now.get("pe", 0) - dp_before.get("pe", 0), 3),
            "coherence_delta": round(
                dp_now.get("coherence", 0) - dp_before.get("coherence", 0), 3
            ),
            "issues_count": len(dp_issues),
        }

        obs = sense.get("observer", {})
        learn["observer_feedback"] = {
            "available": obs.get("available", False),
            "memories_stored": obs.get("memories_stored", 0),
        }

        strategy_used = analyze.get("strategy", "balanced")
        learn["strategy_used"] = strategy_used

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

        if dp_now.get("available"):
            self._evolution_metrics["dark_plane_couplings"] = self._evolution_metrics.get("dark_plane_couplings", 0) + 1

        learn["evolution_metrics"] = dict(self._evolution_metrics)
        return learn

    def _inject_trajectory(
        self, field: "HoneycombNeuralField", report: Dict[str, Any]
    ) -> None:
        if not hasattr(field, "_observer") or field._observer is None:
            return
        try:
            from .runtime_observer import LogEvent
            obs = field._observer
            if not obs._enabled:
                return
            cycle = report.get("cycle", 0)
            strategy = report.get("phases", {}).get("PLAN", {}).get("strategy", "?")
            quality = report.get("phases", {}).get("LEARN", {}).get("quality_score", 0)
            issue_count = report.get("phases", {}).get("ANALYZE", {}).get("issue_count", 0)
            duration = report.get("duration_seconds", 0)
            dp_issues = report.get("phases", {}).get("ANALYZE", {}).get("dark_plane_issues", [])

            msg = (
                f"Evolution cycle #{cycle}: strategy={strategy} quality={quality:.2f} "
                f"issues={issue_count} dark_plane_issues={len(dp_issues)} duration={duration:.1f}s"
            )
            level = "INFO" if quality >= 0.4 else "WARNING"
            if dp_issues:
                level = "WARNING"
            if quality < 0.2:
                level = "ERROR"

            evt = LogEvent(
                timestamp=time.time(),
                level=level,
                module="agent_loop",
                message=msg,
            )
            obs.ingest(evt)
            self._evolution_metrics["observer_trajectories_injected"] = (
                self._evolution_metrics.get("observer_trajectories_injected", 0) + 1
            )
        except Exception:
            pass

