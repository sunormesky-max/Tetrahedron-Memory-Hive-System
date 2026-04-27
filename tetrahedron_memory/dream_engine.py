from __future__ import annotations

import logging
import random
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .pcnn_types import PCNNConfig, PulseType

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField

logger = logging.getLogger("tetramem.honeycomb")


class DreamCycleResult:
    __slots__ = (
        "cycle_time", "sources_used", "dreams_created", "cross_domain",
        "insights", "depth_levels", "rejected_low_quality",
    )

    def __init__(self):
        self.cycle_time: float = time.time()
        self.sources_used: int = 0
        self.dreams_created: int = 0
        self.cross_domain: int = 0
        self.insights: List[Dict] = []
        self.depth_levels: Dict[str, int] = {"L1_cross_domain": 0, "L2_same_domain": 0, "L3_meta": 0}
        self.rejected_low_quality: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_time": self.cycle_time,
            "sources_used": self.sources_used,
            "dreams_created": self.dreams_created,
            "cross_domain": self.cross_domain,
            "insights": self.insights[:10],
            "depth_levels": self.depth_levels,
            "rejected_low_quality": self.rejected_low_quality,
        }


class DreamEngine:
    """
    Autonomous creative memory recombination engine.

    Dream Cycle:
    1. Select 2-3 high-weight memory clusters (different label domains)
    2. Extract key elements from each cluster
    3. Recombine into new dream memories (cross-domain synthesis)
    4. Score dream creativity: label distance * weight product * activation resonance
    5. Store high-scoring dreams as __dream__ memories
    6. Emit cascade pulses from dream nodes to propagate insights

    This mirrors human REM sleep creativity: disparate neural patterns
    recombine during dream states to produce novel associations.
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._history: List[DreamCycleResult] = []
        self._max_history = 30
        self._total_dreams = 0
        self._lock = threading.Lock()
        self._dream_effectiveness: Dict[str, Dict[str, Any]] = {}
        self._max_effectiveness_tracking = 200
        self._is_dreaming = False

    def _extract_content_summary(self, content: str, max_chars: int = 40) -> str:
        if not content:
            return ""
        clean = content.lstrip("[").split("] ", 1)
        text = clean[-1] if len(clean) > 1 else clean[0]
        text = text.strip()
        for sep in ["。", "，", "；", "\n", "——", "：", "|"]:
            if sep in text and text.index(sep) < max_chars:
                text = text[:text.index(sep)]
                break
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text

    def _extract_deep_structure(self, content: str) -> Dict:
        if not content:
            return {"core_concept": "", "key_principles": [], "methods": [], "constraints": []}
        clean = content.lstrip("[").split("] ", 1)
        text = clean[-1] if len(clean) > 1 else clean[0]
        sections = text.split("\n")
        core = ""
        principles = []
        methods = []
        constraints = []
        for sec in sections:
            stripped = sec.strip()
            if not stripped:
                continue
            if stripped.startswith("【") and "】" in stripped:
                bracket_content = stripped
                if any(k in bracket_content for k in ["核心", "本质", "关键", "根本", "基础", "原理", "逻辑"]):
                    inner = bracket_content.split("】", 1)[1].strip() if "】" in bracket_content else ""
                    if inner:
                        principles.append(inner[:60])
                elif any(k in bracket_content for k in ["流程", "步骤", "方法", "操作", "做法", "如何"]):
                    inner = bracket_content.split("】", 1)[1].strip() if "】" in bracket_content else ""
                    if inner:
                        methods.append(inner[:60])
                elif any(k in bracket_content for k in ["注意", "限制", "前提", "条件", "必须", "不可", "约束"]):
                    inner = bracket_content.split("】", 1)[1].strip() if "】" in bracket_content else ""
                    if inner:
                        constraints.append(inner[:60])
                elif not core:
                    core = bracket_content.split("】", 1)[1].strip()[:40] if "】" in bracket_content else ""
            elif any(stripped.startswith(p) for p in ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⒈", "⒉", "⒊", "1.", "2.", "3.", "4.", "5.", "- ", "* "]):
                item = stripped.lstrip("①②③④⑤⑥⑦⑧⑨⑩⒈⒉⒊⒋⒌1234567890.-* ").strip()
                if item:
                    principles.append(item[:50])
            elif not core and len(stripped) > 5:
                core = stripped[:40]
        if not core:
            for sep in ["。", "，", "\n", "——"]:
                if sep in text:
                    core = text[:text.index(sep)][:40]
                    break
            if not core:
                core = text[:40]
        return {
            "core_concept": core,
            "key_principles": principles[:5],
            "methods": methods[:3],
            "constraints": constraints[:3],
        }

    def _compute_expertise_depth(self, domain_label: str, domain_nodes: List) -> float:
        if not domain_nodes:
            return 0.0
        total_weight = sum(n.weight for _, n in domain_nodes)
        avg_weight = total_weight / len(domain_nodes)
        node_ids = {nid for nid, _ in domain_nodes}
        internal_links = 0
        for nid, n in domain_nodes:
            for fnid in n.face_neighbors + n.edge_neighbors:
                if fnid in node_ids:
                    internal_links += 1
        density = internal_links / max(len(domain_nodes) * 4, 1)
        avg_content_len = sum(len(n.content) for _, n in domain_nodes) / max(len(domain_nodes), 1)
        content_depth = min(1.0, avg_content_len / 500.0)
        crystal_count = sum(1 for _, n in domain_nodes if n.crystal_channels)
        crystal_ratio = crystal_count / max(len(domain_nodes), 1)
        return min(1.0, avg_weight / 5.0 * 0.25 + density * 0.25 + content_depth * 0.3 + crystal_ratio * 0.2)

    def _generate_deep_insight(self, struct_a: Dict, struct_b: Dict,
                                domain_a: str, domain_b: str,
                                depth_a: float, depth_b: float) -> str:
        core_a = struct_a["core_concept"]
        core_b = struct_b["core_concept"]
        principles_a = struct_a["key_principles"]
        principles_b = struct_b["key_principles"]
        methods_a = struct_a["methods"]
        methods_b = struct_b["methods"]
        constraints_a = struct_a["constraints"]
        constraints_b = struct_b["constraints"]
        combined_depth = (depth_a + depth_b) / 2

        insight_sections = []

        header = f"{domain_a}核心⟨{core_a}⟩" if domain_a != domain_b else f"同域对照⟨{core_a}⟩↔⟨{core_b}⟩"
        if domain_a != domain_b:
            header += f" ↔ {domain_b}核心⟨{core_b}⟩"
        insight_sections.append(header)

        principle_insights = []
        if principles_a and principles_b:
            for pa in principles_a[:2]:
                for pb in principles_b[:2]:
                    principle_insights.append(f"{domain_a}「{pa[:25]}」↔{domain_b}「{pb[:25]}」")
        elif principles_a:
            principle_insights.append(f"{domain_a}原理: {principles_a[0][:40]}")
        elif principles_b:
            principle_insights.append(f"{domain_b}原理: {principles_b[0][:40]}")
        if principle_insights:
            insight_sections.append("【原理映射】" + " | ".join(principle_insights[:3]))

        method_insights = []
        if methods_a and methods_b:
            method_insights.append(f"{domain_a}「{methods_a[0][:25]}」→ {domain_b}「{methods_b[0][:25]}」")
            method_insights.append(f"{domain_b}「{methods_b[0][:25]}」→ {domain_a}「{methods_a[0][:25]}」")
        elif methods_a:
            method_insights.append(f"{domain_a}方法可迁移: {methods_a[0][:40]}")
        elif methods_b:
            method_insights.append(f"{domain_b}方法可迁移: {methods_b[0][:40]}")
        if method_insights:
            insight_sections.append("【方法迁移】" + " | ".join(method_insights[:2]))

        constraint_insights = []
        if constraints_a and constraints_b:
            constraint_insights.append(f"{domain_a}「{constraints_a[0][:25]}」× {domain_b}「{constraints_b[0][:25]}」")
        if constraint_insights:
            insight_sections.append("【约束碰撞】" + " | ".join(constraint_insights))

        if combined_depth > 0.5:
            insight_sections.append(f"【深度{combined_depth:.0%}】交叉创新潜力高")
        elif combined_depth > 0.3:
            insight_sections.append(f"【深度{combined_depth:.0%}】有一定创新基础")
        else:
            insight_sections.append(f"【深度{combined_depth:.0%}】需继续深化知识")

        return " | ".join(insight_sections)

    def _trace_dream_path(self, field, nid_a: str, nid_b: str, max_hops: int = 8) -> List[str]:
        if nid_a == nid_b:
            return []
        visited = {nid_a}
        parent = {}
        queue = [nid_a]
        for _ in range(max_hops):
            next_q = []
            for fid in queue:
                fn = field._nodes.get(fid)
                if fn is None:
                    continue
                for nnid in fn.face_neighbors[:6] + fn.edge_neighbors[:4]:
                    if nnid == nid_b:
                        path = [nid_b]
                        cur = fid
                        while cur in parent:
                            path.append(cur)
                            cur = parent[cur]
                        path.append(nid_a)
                        path.reverse()
                        occupied_waypoints = []
                        for pid in path:
                            pn = field._nodes.get(pid)
                            if pn and pn.is_occupied and pid not in (nid_a, nid_b):
                                occupied_waypoints.append(pid)
                        return occupied_waypoints
                    if nnid not in visited:
                        visited.add(nnid)
                        parent[nnid] = fid
                        next_q.append(nnid)
            queue = next_q
            if not queue:
                break
        return []

    def _dream_cascade(self, source_a: str, source_b: str) -> Optional[Dict]:
        """
        Run a PULSE CASCADE between two source memories to discover
        creative bridge concepts through the actual learned topology.
        """
        field = self._field

        cascade_result = field._dream_cascade_pulse(source_a, source_b)
        if cascade_result is None:
            return None

        path_nodes = cascade_result.get("path_nodes", [])
        resonance_nodes = cascade_result.get("resonance_nodes", {})

        if not resonance_nodes:
            return None

        return {
            "source_a": source_a,
            "source_b": source_b,
            "path_nodes": path_nodes,
            "resonance_nodes": resonance_nodes,
            "cascade_strength": cascade_result.get("cascade_strength", 0.0),
        }

    def _evaluate_dream_quality(self, dream_id: str) -> float:
        """
        Post-creation quality assessment using pulse resonance.
        """
        field = self._field
        node = field._nodes.get(dream_id)
        if node is None or not node.is_occupied:
            return 0.0

        reach = 0
        intensity_sum = 0.0
        visited = {dream_id}
        queue = [dream_id]
        for _ in range(3):
            next_q = []
            for nid in queue:
                n = field._nodes.get(nid)
                if n is None:
                    continue
                for fnid in n.face_neighbors[:8] + n.edge_neighbors[:6]:
                    if fnid in visited:
                        continue
                    visited.add(fnid)
                    fn = field._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        reach += 1
                        intensity_sum += fn.pulse_accumulator
                        next_q.append(fnid)
            queue = next_q

        node_labels = set(l for l in node.labels if not l.startswith("__"))
        bridge_score = 0.0
        for nid in visited:
            n = field._nodes.get(nid)
            if n and n.is_occupied:
                n_labels = set(l for l in n.labels if not l.startswith("__"))
                if n_labels and node_labels:
                    overlap = len(node_labels & n_labels)
                    if overlap > 0 and node_labels != n_labels:
                        bridge_score += 0.2

        reinforcement = min(1.0, node.reinforcement_count / 10.0)

        reach_norm = min(1.0, reach / 20.0)
        intensity_norm = min(1.0, intensity_sum / 10.0)
        bridge_norm = min(1.0, bridge_score)

        quality = 0.30 * reach_norm + 0.25 * intensity_norm + 0.25 * bridge_norm + 0.20 * reinforcement
        return min(1.0, max(0.0, quality))

    def _generate_cascade_insight(self, source_a_id: str, source_b_id: str,
                                   path_nodes: List[str],
                                   resonance_nodes: Dict[str, float]) -> str:
        """
        Generate insight text from cascade results using actual intermediate concepts.
        """
        field = self._field

        na = field._nodes.get(source_a_id)
        nb = field._nodes.get(source_b_id)
        if na is None or nb is None:
            return ""

        summary_a = self._extract_content_summary(na.content, 30)
        summary_b = self._extract_content_summary(nb.content, 30)

        labels_a = [l for l in na.labels if not l.startswith("__")][:1]
        labels_b = [l for l in nb.labels if not l.startswith("__")][:1]
        domain_a = labels_a[0] if labels_a else "domain_a"
        domain_b = labels_b[0] if labels_b else "domain_b"

        bridge_parts = []
        sorted_resonance = sorted(resonance_nodes.items(), key=lambda x: -x[1])[:5]
        for rid, strength in sorted_resonance:
            rn = field._nodes.get(rid)
            if rn and rn.is_occupied:
                rs = self._extract_content_summary(rn.content, 25)
                r_labels = [l for l in rn.labels if not l.startswith("__")][:1]
                if rs:
                    label_hint = r_labels[0] if r_labels else ""
                    bridge_parts.append((label_hint, rs, strength))

        parts = []
        parts.append(f"{domain_a}核心⟨{summary_a}⟩")
        for label_hint, rs, strength in bridge_parts:
            if label_hint:
                parts.append(f"经由{label_hint}(共振{strength:.1f}): {rs}")
            else:
                parts.append(f"桥接(共振{strength:.1f}): {rs}")
        parts.append(f"{domain_b}核心⟨{summary_b}⟩")

        insight = "[cascade-dream] " + " -> ".join(parts)

        if bridge_parts:
            methods = []
            for label_hint, rs, _ in bridge_parts[:2]:
                if label_hint:
                    methods.append(f"{label_hint}「{rs[:20]}」")
            if methods:
                insight += " | 【脉冲桥接】" + " | ".join(methods)

        return insight

    def run_dream_cycle(self) -> DreamCycleResult:
        self._is_dreaming = True
        result = DreamCycleResult()
        field = self._field
        cfg = PCNNConfig

        with field._lock:
            occupied = [
                (nid, n) for nid, n in field._nodes.items()
                if n.is_occupied and n.weight >= cfg.DREAM_MIN_SOURCE_WEIGHT
                and "__dream__" not in n.labels
                and "__pulse_bridge__" not in n.labels
                and "__consolidated__" not in n.labels
            ]
            if len(occupied) < 4:
                result.insights = [{"note": "insufficient source memories for dreaming"}]
                self._is_dreaming = False
                return result

            label_domains: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
            for nid, node in occupied:
                for lbl in node.labels:
                    if not lbl.startswith("__"):
                        label_domains[lbl].append((nid, node))

            top_domains = sorted(
                label_domains.items(),
                key=lambda x: -sum(n.weight for _, n in x[1])
            )[:8]

            if len(top_domains) < 2:
                result.insights = [{"note": "insufficient label diversity for dreaming"}]
                self._is_dreaming = False
                return result

            domain_depths = {}
            for dname, dnodes in top_domains:
                domain_depths[dname] = self._compute_expertise_depth(dname, dnodes)

            eligible_domains = {
                dname for dname, depth in domain_depths.items()
                if depth >= cfg.DREAM_MIN_DOMAIN_DEPTH
            }

            result.sources_used = len(occupied)

            cascade_attempts = min(3, cfg.DREAM_MAX_RECOMBINATIONS // 3)
            for _ in range(cascade_attempts):
                if len(top_domains) < 2:
                    break
                d1_idx = random.randint(0, len(top_domains) - 1)
                d2_candidates = [i for i in range(len(top_domains)) if i != d1_idx]
                if not d2_candidates:
                    continue
                d2_idx = random.choice(d2_candidates)
                domain_a_nodes = top_domains[d1_idx][1]
                domain_b_nodes = top_domains[d2_idx][1]
                if not domain_a_nodes or not domain_b_nodes:
                    continue
                src_a = random.choice(domain_a_nodes)
                src_b = random.choice(domain_b_nodes)
                nid_a, node_a = src_a
                nid_b, node_b = src_b
                if (node_a.weight < cfg.DREAM_SOURCE_MIN_WEIGHT
                        or node_b.weight < cfg.DREAM_SOURCE_MIN_WEIGHT):
                    continue

                cascade_data = self._dream_cascade(nid_a, nid_b)
                if cascade_data and cascade_data.get("resonance_nodes"):
                    resonance = cascade_data["resonance_nodes"]
                    max_res = max(resonance.values()) if resonance else 0
                    if max_res < 0.1:
                        continue

                    insight_text = self._generate_cascade_insight(
                        nid_a, nid_b,
                        cascade_data.get("path_nodes", []),
                        resonance,
                    )
                    if not insight_text:
                        continue

                    domain_a_name = top_domains[d1_idx][0]
                    domain_b_name = top_domains[d2_idx][0]
                    dream_labels = list(set([
                        domain_a_name, domain_b_name, "__dream__",
                    ]))
                    dream_weight = max(
                        0.5,
                        min(
                            cfg.DREAM_INSIGHT_WEIGHT * max_res,
                            max(node_a.weight, node_b.weight) * 0.8,
                        )
                    )

                    try:
                        dream_id = field.store(
                            content=insight_text,
                            labels=dream_labels,
                            weight=dream_weight,
                            metadata={
                                "dream_source_a": nid_a[:12],
                                "dream_source_b": nid_b[:12],
                                "creativity_score": round(max_res, 3),
                                "dream_type": "cascade",
                                "dream_depth_level": "L1_cross_domain",
                                "cascade_resonance_count": len(resonance),
                                "insight_type": "cascade_pulse",
                            },
                        )
                        result.dreams_created += 1
                        self._total_dreams += 1
                        result.cross_domain += 1
                        result.depth_levels["L1_cross_domain"] = result.depth_levels.get("L1_cross_domain", 0) + 1

                        quality = self._evaluate_dream_quality(dream_id)

                        with self._lock:
                            self._dream_effectiveness[dream_id] = {
                                "created_time": time.time(),
                                "initial_weight": dream_weight,
                                "creativity": max_res,
                                "depth_level": "L1_cross_domain",
                                "reinforcement_count": 0,
                                "cascade_quality": round(quality, 3),
                            }
                            if len(self._dream_effectiveness) > self._max_effectiveness_tracking:
                                oldest = sorted(
                                    self._dream_effectiveness.items(),
                                    key=lambda x: x[1]["created_time"]
                                )
                                for oid, _ in oldest[:len(self._dream_effectiveness) - self._max_effectiveness_tracking // 2]:
                                    del self._dream_effectiveness[oid]

                        result.insights.append({
                            "dream_id": dream_id[:12],
                            "domains": [domain_a_name, domain_b_name],
                            "creativity": round(max_res, 3),
                            "weight": round(dream_weight, 3),
                            "cross_domain": True,
                            "depth_level": "L1_cross_domain",
                            "cascade_quality": round(quality, 3),
                            "resonance_nodes": len(resonance),
                            "insight_preview": insight_text[:80],
                        })

                        field._emit_pulse(
                            dream_id, strength=dream_weight * 0.6,
                            pulse_type=PulseType.CASCADE,
                        )
                        self._apply_dream_quality_feedback(dream_id, dream_weight)
                    except Exception as e:
                        logger.warning("Cascade dream creation failed: %s", e)

            dream_nodes = [
                (nid, n) for nid, n in field._nodes.items()
                if n.is_occupied and "__dream__" in n.labels
            ]
            if len(dream_nodes) >= 3:
                self._run_meta_dream(field, dream_nodes, result, cfg)

            for _ in range(cfg.DREAM_MAX_RECOMBINATIONS):
                if len(top_domains) < 2:
                    break

                mode = random.random()
                if mode < 0.4:
                    deepest_eligible = [
                        (n, d) for n, d in domain_depths.items() if n in eligible_domains
                    ]
                    if not deepest_eligible:
                        deepest_eligible = list(domain_depths.items())
                    deepest_eligible.sort(key=lambda x: -x[1])
                    d1_idx = next(
                        (i for i, (n, _) in enumerate(top_domains) if n == deepest_eligible[0][0]), 0
                    )
                    d2_candidates = [i for i in range(len(top_domains)) if i != d1_idx]
                    if not d2_candidates:
                        continue
                    d2_idx = random.choice(d2_candidates)
                elif mode < 0.7:
                    d1_idx = random.randint(0, len(top_domains) - 1)
                    d2_idx = random.randint(0, len(top_domains) - 1)
                    if d1_idx == d2_idx:
                        continue
                else:
                    d1_idx = random.randint(0, len(top_domains) - 1)
                    same_domain_nodes = top_domains[d1_idx][1]
                    if len(same_domain_nodes) < 2:
                        continue
                    sampled = random.sample(same_domain_nodes, 2)
                    src_a = sampled[0]
                    src_b = sampled[1]
                    nid_a, node_a = src_a
                    nid_b, node_b = src_b
                    if (node_a.weight < cfg.DREAM_SOURCE_MIN_WEIGHT
                            or node_a.activation < cfg.DREAM_SOURCE_MIN_ACTIVATION
                            or node_b.weight < cfg.DREAM_SOURCE_MIN_WEIGHT
                            or node_b.activation < cfg.DREAM_SOURCE_MIN_ACTIVATION):
                        continue
                    domain_a_name = top_domains[d1_idx][0]
                    domain_b_name = domain_a_name
                    depth_a = domain_depths.get(domain_a_name, 0)
                    depth_b = depth_a
                    struct_a = self._extract_deep_structure(node_a.content)
                    struct_b = self._extract_deep_structure(node_b.content)
                    creativity = self._score_creativity(node_a, node_b, domain_a_name, domain_b_name)
                    creativity = min(1.0, creativity * 1.3)
                    if creativity < cfg.DREAM_INSIGHT_MIN_CREATIVITY:
                        result.rejected_low_quality += 1
                        continue
                    self._create_deep_dream(field, result, nid_a, nid_b, node_a, node_b,
                                            domain_a_name, domain_b_name, struct_a, struct_b,
                                            depth_a, depth_b, creativity, cfg, depth_level="L2_same_domain")
                    continue

                domain_a_name, domain_a_nodes = top_domains[d1_idx]
                domain_b_name, domain_b_nodes = top_domains[d2_idx]

                src_a = random.choice(domain_a_nodes)
                src_b = random.choice(domain_b_nodes)
                nid_a, node_a = src_a
                nid_b, node_b = src_b

                if (node_a.weight < cfg.DREAM_SOURCE_MIN_WEIGHT
                        or node_a.activation < cfg.DREAM_SOURCE_MIN_ACTIVATION
                        or node_b.weight < cfg.DREAM_SOURCE_MIN_WEIGHT
                        or node_b.activation < cfg.DREAM_SOURCE_MIN_ACTIVATION):
                    continue

                spatial_dist = float(np.linalg.norm(node_a.position - node_b.position))
                max_dist = field._spacing * 8
                spatial_factor = min(1.0, spatial_dist / max_dist)
                if spatial_factor < 0.15:
                    continue

                field_tension_a = field._reflection_field.get_dream_tension(nid_a) if field._reflection_field else 0.5
                field_tension_b = field._reflection_field.get_dream_tension(nid_b) if field._reflection_field else 0.5
                tension_product = field_tension_a * field_tension_b

                creativity = self._score_creativity(node_a, node_b, domain_a_name, domain_b_name)
                creativity = min(1.0, creativity * (1.0 + tension_product * 0.3))

                if creativity < cfg.DREAM_INSIGHT_MIN_CREATIVITY:
                    result.rejected_low_quality += 1
                    continue

                depth_a = domain_depths.get(domain_a_name, 0)
                depth_b = domain_depths.get(domain_b_name, 0)
                struct_a = self._extract_deep_structure(node_a.content)
                struct_b = self._extract_deep_structure(node_b.content)

                self._create_deep_dream(field, result, nid_a, nid_b, node_a, node_b,
                                        domain_a_name, domain_b_name, struct_a, struct_b,
                                        depth_a, depth_b, creativity, cfg, depth_level="L1_cross_domain")

        with self._lock:
            self._history.append(result)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history // 2:]

        self._is_dreaming = False
        return result

    def _create_deep_dream(self, field, result, nid_a, nid_b, node_a, node_b,
                           domain_a_name, domain_b_name, struct_a, struct_b,
                           depth_a, depth_b, creativity, cfg, depth_level="L1_cross_domain"):
        summary_a = struct_a["core_concept"]
        summary_b = struct_b["core_concept"]
        waypoints = self._trace_dream_path(field, nid_a, nid_b)
        waypoint_summaries = []
        for wid in waypoints[:3]:
            wn = field._nodes.get(wid)
            if wn and wn.is_occupied:
                ws = self._extract_content_summary(wn.content, 30)
                if ws:
                    wlabels = [l for l in wn.labels if not l.startswith("__")][:2]
                    waypoint_summaries.append((wlabels, ws))

        deep_insight = self._generate_deep_insight(
            struct_a, struct_b, domain_a_name, domain_b_name, depth_a, depth_b
        )

        dream_parts = []
        dream_parts.append(f"{domain_a_name}核心: {summary_a}")
        for wlabels, ws in waypoint_summaries:
            label_hint = "/".join(wlabels) if wlabels else ""
            dream_parts.append(f"经由{label_hint}: {ws}" if label_hint else f"中间: {ws}")
        dream_parts.append(f"{domain_b_name}核心: {summary_b}")

        dream_content = "[dream] " + " -> ".join(dream_parts) + " | " + deep_insight
        dream_labels = list(set([
            domain_a_name, domain_b_name, "__dream__",
        ]))
        dream_weight = max(
            0.5,
            min(
                cfg.DREAM_INSIGHT_WEIGHT * creativity * (1.0 + (depth_a + depth_b) * 0.2),
                max(node_a.weight, node_b.weight) * 0.8,
            )
        )

        path_length = len(waypoints) + 2
        spatial_dist = round(float(np.linalg.norm(node_a.position - node_b.position)), 2)

        try:
            dream_id = field.store(
                content=dream_content,
                labels=dream_labels,
                weight=dream_weight,
                metadata={
                    "dream_source_a": nid_a[:12],
                    "dream_source_b": nid_b[:12],
                    "creativity_score": round(creativity, 3),
                    "dream_type": "cross_domain" if domain_a_name != domain_b_name else "intra_domain",
                    "dream_depth_level": depth_level,
                    "spatial_distance": spatial_dist,
                    "topo_path_length": path_length,
                    "waypoint_count": len(waypoints),
                    "expertise_depth_a": round(depth_a, 3),
                    "expertise_depth_b": round(depth_b, 3),
                    "insight_type": "deep_structural",
                },
            )
            result.dreams_created += 1
            self._total_dreams += 1
            result.depth_levels[depth_level] = result.depth_levels.get(depth_level, 0) + 1

            with self._lock:
                self._dream_effectiveness[dream_id] = {
                    "created_time": time.time(),
                    "initial_weight": dream_weight,
                    "creativity": creativity,
                    "depth_level": depth_level,
                    "reinforcement_count": 0,
                }
                if len(self._dream_effectiveness) > self._max_effectiveness_tracking:
                    oldest = sorted(
                        self._dream_effectiveness.items(),
                        key=lambda x: x[1]["created_time"]
                    )
                    for oid, _ in oldest[:len(self._dream_effectiveness) - self._max_effectiveness_tracking // 2]:
                        del self._dream_effectiveness[oid]

            is_cross = domain_a_name != domain_b_name
            if is_cross:
                result.cross_domain += 1

            result.insights.append({
                "dream_id": dream_id[:12],
                "domains": [domain_a_name, domain_b_name],
                "creativity": round(creativity, 3),
                "weight": round(dream_weight, 3),
                "cross_domain": is_cross,
                "depth_level": depth_level,
                "spatial_distance": spatial_dist,
                "topo_path_length": path_length,
                "expertise_depth": round((depth_a + depth_b) / 2, 3),
                "insight_preview": deep_insight[:80],
            })

            field._emit_pulse(
                dream_id, strength=dream_weight * 0.6,
                pulse_type=PulseType.CASCADE,
            )
            self._apply_dream_quality_feedback(dream_id, dream_weight)
        except Exception as e:
            logger.warning("Dream creation failed: %s", e)

        return result

    def _run_meta_dream(self, field, dream_nodes: List, result: DreamCycleResult, cfg):
        if len(dream_nodes) < 3:
            return
        sampled = random.sample(dream_nodes, min(3, len(dream_nodes)))
        patterns = []
        domain_set = set()
        for nid, node in sampled:
            meta = node.metadata or {}
            domains = meta.get("dream_type", "")
            cr = meta.get("creativity_score", 0)
            dl = meta.get("dream_depth_level", "")
            patterns.append(f"[{dl}]creativity={cr:.2f}")
            for lbl in node.labels:
                if not lbl.startswith("__"):
                    domain_set.add(lbl)
        meta_content = (
            "[meta-dream] 梦境模式反思: "
            + " | ".join(patterns)
            + f" | 覆盖域: {','.join(list(domain_set)[:5])}"
        )
        meta_labels = list(domain_set)[:3] + ["__dream__", "__meta_dream__"]
        try:
            meta_id = field.store(
                content=meta_content,
                labels=meta_labels,
                weight=0.8,
                metadata={
                    "dream_type": "meta_dream",
                    "dream_depth_level": "L3_meta",
                    "source_dreams": [nid[:12] for nid, _ in sampled],
                },
            )
            result.dreams_created += 1
            self._total_dreams += 1
            result.depth_levels["L3_meta"] = result.depth_levels.get("L3_meta", 0) + 1
            result.insights.append({
                "dream_id": meta_id[:12],
                "domains": list(domain_set)[:3],
                "creativity": 0.7,
                "weight": 0.8,
                "cross_domain": True,
                "depth_level": "L3_meta",
                "insight_preview": meta_content[:80],
            })
            self._apply_dream_quality_feedback(meta_id, 0.8)
        except Exception as e:
            logger.warning("Meta-dream creation failed: %s", e)

    def track_reinforcement(self, dream_id: str):
        with self._lock:
            if dream_id in self._dream_effectiveness:
                self._dream_effectiveness[dream_id]["reinforcement_count"] += 1

    def _score_creativity(self, node_a, node_b, domain_a: str, domain_b: str) -> float:
        cfg = PCNNConfig
        field = self._field
        cross_bonus = cfg.DREAM_CROSS_DOMAIN_BONUS if domain_a != domain_b else 1.0
        weight_factor = (node_a.weight * node_b.weight) / 25.0
        activation_factor = (node_a.activation + node_b.activation) / 10.0
        structural = self._structural_distance(node_a, node_b)
        crystal_boost_a = sum(node_a.crystal_channels.values()) if node_a.crystal_channels else 0
        crystal_boost_b = sum(node_b.crystal_channels.values()) if node_b.crystal_channels else 0
        crystal_factor = min(1.0, (crystal_boost_a + crystal_boost_b) / 10.0)
        hebbian_factor = 0.0
        hebbian_w = field._hebbian.get_path_bias(node_a.id, node_b.id)
        if hebbian_w > 0:
            hebbian_factor = min(1.0, hebbian_w * 0.3)
        pcnn_factor = 0.0
        if node_a.fired and node_b.fired:
            pcnn_factor = 0.15
        elif node_a.internal_activity > 0.5 or node_b.internal_activity > 0.5:
            pcnn_factor = 0.08
        resonance_factor = 0.0
        for ev in field._resonance_events:
            if node_a.id[:8] in ev.get("node_ids", []) or node_b.id[:8] in ev.get("node_ids", []):
                resonance_factor = 0.1
                break
        moran_factor = 0.0
        if field._spatial_autocorrelation > 0.1:
            moran_factor = 0.05
        base = (cross_bonus * weight_factor * activation_factor) / (cross_bonus + 1)
        creativity = base * (0.30 + 0.25 * structural + 0.12 * crystal_factor + 0.08 * min(1.0, weight_factor)
                            + 0.08 * hebbian_factor + 0.07 * pcnn_factor + 0.05 * resonance_factor + 0.05 * moran_factor)
        return min(1.0, creativity)

    def _structural_distance(self, node_a, node_b) -> float:
        try:
            dist = float(np.linalg.norm(node_a.position - node_b.position))
            return min(1.0, dist / (self._field._spacing * 8))
        except Exception as e:
            logger.debug("Structural distance fallback: %s", e)
            return 0.5

    def get_history(self, n: int = 10) -> List[Dict]:
        with self._lock:
            return [r.to_dict() for r in self._history[-n:]]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_reinforced = sum(
                1 for v in self._dream_effectiveness.values()
                if v["reinforcement_count"] > 0
            )
            effectiveness_rate = (
                total_reinforced / len(self._dream_effectiveness)
                if self._dream_effectiveness else 0.0
            )
            return {
                "total_dreams_created": self._total_dreams,
                "dream_cycles_run": len(self._history),
                "latest_dreams": sum(r.dreams_created for r in self._history[-3:]),
                "total_rejected_low_quality": sum(r.rejected_low_quality for r in self._history),
                "dream_effectiveness_rate": round(effectiveness_rate, 3),
                "dreams_tracked": len(self._dream_effectiveness),
                "dreams_reinforced": total_reinforced,
            }

    def run_pulse_cascade_dream(self) -> Optional[Dict[str, Any]]:
        field = self._field
        cfg = PCNNConfig

        with field._lock:
            occupied = [
                (nid, n) for nid, n in field._nodes.items()
                if n.is_occupied and n.weight >= cfg.DREAM_MIN_SOURCE_WEIGHT
                and "__dream__" not in n.labels
                and "__pulse_bridge__" not in n.labels
            ]
            if len(occupied) < 2:
                return None

            label_domains: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
            for nid, node in occupied:
                for lbl in node.labels:
                    if not lbl.startswith("__"):
                        label_domains[lbl].append((nid, node))

            if len(label_domains) < 2:
                return None

            top_domains = sorted(
                label_domains.items(),
                key=lambda x: -sum(n.weight for _, n in x[1])
            )

            domain_names = [d[0] for d in top_domains]
            d1_idx = random.randint(0, len(top_domains) - 1)
            d2_candidates = [i for i in range(len(top_domains)) if i != d1_idx]
            if not d2_candidates:
                return None
            d2_idx = random.choice(d2_candidates)

            domain_a_name = domain_names[d1_idx]
            domain_b_name = domain_names[d2_idx]

            nodes_a = sorted(top_domains[d1_idx][1], key=lambda x: -x[1].weight)
            nodes_b = sorted(top_domains[d2_idx][1], key=lambda x: -x[1].weight)

            nid_a, node_a = nodes_a[0]
            nid_b, node_b = nodes_b[0]

            if node_a.weight < cfg.DREAM_SOURCE_MIN_WEIGHT or node_b.weight < cfg.DREAM_SOURCE_MIN_WEIGHT:
                return None

            field._emit_pulse(nid_a, strength=node_a.weight * 0.5, pulse_type=PulseType.CASCADE)
            field._emit_pulse(nid_b, strength=node_b.weight * 0.5, pulse_type=PulseType.CASCADE)

            path_a_to_b = self._trace_dream_path(field, nid_a, nid_b, max_hops=12)
            path_b_to_a = self._trace_dream_path(field, nid_b, nid_a, max_hops=12)

            all_intermediates = set(path_a_to_b) | set(path_b_to_a)

            intermediate_nodes = {}
            for iid in all_intermediates:
                inode = field._nodes.get(iid)
                if inode and inode.is_occupied:
                    resonance = inode.pulse_accumulator + inode.activation * 0.3
                    intermediate_nodes[iid] = resonance

            if not intermediate_nodes:
                return None

            best_intermediate_id = max(intermediate_nodes, key=lambda k: intermediate_nodes[k])
            best_resonance = intermediate_nodes[best_intermediate_id]

            bridge_parts = []
            sorted_intermediates = sorted(intermediate_nodes.items(), key=lambda x: -x[1])[:5]
            for iid, res in sorted_intermediates:
                inode = field._nodes.get(iid)
                if inode and inode.is_occupied:
                    summary = self._extract_content_summary(inode.content, 25)
                    ilabels = [l for l in inode.labels if not l.startswith("__")][:1]
                    label_hint = ilabels[0] if ilabels else ""
                    bridge_parts.append(f"{'('+label_hint+')' if label_hint else ''}{summary}(r={res:.2f})")

            cascade_path_summary = " -> ".join([
                self._extract_content_summary(node_a.content, 25),
                *bridge_parts,
                self._extract_content_summary(node_b.content, 25),
            ])

            insight_text = (
                f"[pulse-cascade-dream] {domain_a_name} → {domain_b_name} | "
                f"路径: {cascade_path_summary} | "
                f"中间节点: {len(intermediate_nodes)} | "
                f"最高共振: {best_resonance:.3f}"
            )

            dream_labels = list(set([domain_a_name, domain_b_name, "__dream__", "__pulse_cascade__"]))
            dream_weight = max(
                0.5,
                min(
                    cfg.DREAM_INSIGHT_WEIGHT * best_resonance,
                    max(node_a.weight, node_b.weight) * 0.75,
                )
            )

            try:
                dream_id = field.store(
                    content=insight_text,
                    labels=dream_labels,
                    weight=dream_weight,
                    metadata={
                        "dream_source_a": nid_a[:12],
                        "dream_source_b": nid_b[:12],
                        "dream_type": "pulse_cascade",
                        "dream_depth_level": "L1_cross_domain",
                        "intermediate_count": len(intermediate_nodes),
                        "best_resonance": round(best_resonance, 3),
                        "best_intermediate": best_intermediate_id[:12],
                        "creativity_score": round(best_resonance, 3),
                        "insight_type": "pulse_cascade",
                    },
                )
                self._total_dreams += 1

                self._apply_dream_quality_feedback(dream_id, dream_weight)

                with self._lock:
                    self._dream_effectiveness[dream_id] = {
                        "created_time": time.time(),
                        "initial_weight": dream_weight,
                        "creativity": best_resonance,
                        "depth_level": "L1_cross_domain",
                        "reinforcement_count": 0,
                        "cascade_quality": round(best_resonance, 3),
                    }

                return {
                    "dream_id": dream_id[:12],
                    "domains": [domain_a_name, domain_b_name],
                    "intermediate_nodes": len(intermediate_nodes),
                    "best_resonance": round(best_resonance, 3),
                    "cascade_path": cascade_path_summary,
                    "insight_preview": insight_text[:80],
                }
            except Exception as e:
                logger.warning("Pulse cascade dream creation failed: %s", e)
                return None

    def generate_creative_insight(self, node_a_id: str, node_b_id: str) -> Optional[Dict[str, Any]]:
        field = self._field

        with field._lock:
            node_a = field._nodes.get(node_a_id)
            node_b = field._nodes.get(node_b_id)

            if node_a is None or node_b is None:
                return None
            if not node_a.is_occupied or not node_b.is_occupied:
                return None

            try:
                geo_distance = float(np.linalg.norm(node_a.position - node_b.position))
            except Exception:
                logger.debug("geo_distance fallback for %s↔%s", node_a_id[:8], node_b_id[:8], exc_info=True)
                geo_distance = field._spacing * 4.0

            max_distance = field._spacing * 10.0
            norm_distance = min(1.0, geo_distance / max_distance)

            labels_a = set(l for l in node_a.labels if not l.startswith("__"))
            labels_b = set(l for l in node_b.labels if not l.startswith("__"))

            if labels_a and labels_b:
                overlap_count = len(labels_a & labels_b)
                union_count = len(labels_a | labels_b)
                label_similarity = overlap_count / union_count if union_count > 0 else 0.0
            elif labels_a or labels_b:
                label_similarity = 0.0
            else:
                label_similarity = 0.5

            hebbian_strength = 0.0
            if field._hebbian:
                hebbian_strength = field._hebbian.get_path_bias(node_a_id, node_b_id)

            topo_path = self._trace_dream_path(field, node_a_id, node_b_id, max_hops=10)
            topo_proximity = 1.0 / (1.0 + len(topo_path)) if topo_path else 0.0

            cross_domain = len(labels_a & labels_b) == 0 and len(labels_a) > 0 and len(labels_b) > 0
            cross_domain_bonus = 1.5 if cross_domain else 1.0

            struct_a = self._extract_deep_structure(node_a.content)
            struct_b = self._extract_deep_structure(node_b.content)

            structural_similarity = 0.0
            if struct_a["key_principles"] and struct_b["key_principles"]:
                shared_p = len(set(p[:15] for p in struct_a["key_principles"])
                                   & set(p[:15] for p in struct_b["key_principles"]))
                max_p = max(len(struct_a["key_principles"]), len(struct_b["key_principles"]), 1)
                structural_similarity = shared_p / max_p

            creativity_score = min(1.0, (
                norm_distance * 0.35
                + label_similarity * 0.20
                + structural_similarity * 0.15
                + topo_proximity * 0.15
                + min(1.0, hebbian_strength * 0.5) * 0.10
                + (0.05 if cross_domain else 0.0)
            ) * cross_domain_bonus)

            reasoning_parts = []
            reasoning_parts.append(f"几何距离={norm_distance:.3f}")
            reasoning_parts.append(f"标签相似度={label_similarity:.3f}")
            reasoning_parts.append(f"结构相似度={structural_similarity:.3f}")
            reasoning_parts.append(f"拓扑邻近={topo_proximity:.3f}")
            reasoning_parts.append(f"赫布强度={hebbian_strength:.3f}")
            reasoning_parts.append(f"跨域={'是' if cross_domain else '否'}")

            suggested_connections = []
            if cross_domain and creativity_score > 0.4:
                suggested_connections.append({
                    "type": "cross_domain_bridge",
                    "from": node_a_id[:12],
                    "to": node_b_id[:12],
                    "strength": round(creativity_score, 3),
                })

            if topo_path and creativity_score > 0.3:
                for mid_id in topo_path[:3]:
                    mid_node = field._nodes.get(mid_id)
                    if mid_node and mid_node.is_occupied:
                        mid_labels = [l for l in mid_node.labels if not l.startswith("__")][:2]
                        suggested_connections.append({
                            "type": "intermediate_bridge",
                            "via_node": mid_id[:12],
                            "via_labels": mid_labels,
                            "resonance": round(mid_node.pulse_accumulator, 3),
                        })

            if hebbian_strength > 0.3:
                suggested_connections.append({
                    "type": "hebbian_reinforcement",
                    "current_strength": round(hebbian_strength, 3),
                    "recommendation": "crystallize" if hebbian_strength > PCNNConfig.CRYSTALLIZE_THRESHOLD * 0.7 else "reinforce",
                })

            return {
                "score": round(creativity_score, 3),
                "reasoning": " | ".join(reasoning_parts),
                "geo_distance": round(geo_distance, 3),
                "label_overlap": round(label_similarity, 3),
                "structural_similarity": round(structural_similarity, 3),
                "cross_domain": cross_domain,
                "suggested_connections": suggested_connections,
                "core_a": struct_a["core_concept"][:50],
                "core_b": struct_b["core_concept"][:50],
            }

    def _apply_dream_quality_feedback(self, dream_id: str, dream_weight: float):
        field = self._field

        try:
            field._emit_pulse(
                dream_id, strength=dream_weight * 0.8,
                pulse_type=PulseType.REINFORCING,
            )

            dream_node = field._nodes.get(dream_id)
            if dream_node is None:
                return

            for fnid in dream_node.face_neighbors[:8]:
                fn = field._nodes.get(fnid)
                if fn and fn.is_occupied:
                    field._emit_pulse(
                        fnid, strength=dream_weight * 0.3,
                        pulse_type=PulseType.DREAM_PULSE,
                    )
        except Exception as e:
            logger.warning("Dream quality feedback pulse failed: %s", e)
