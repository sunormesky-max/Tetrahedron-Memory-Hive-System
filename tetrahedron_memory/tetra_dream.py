"""
Dream Law — TetraMesh-native dream cycle.

Per the TetraMem-XL design specification (Grok production plan):
  1. PH-weighted random walk through the mesh
  2. Discover topologically related but spatially distant memories
  3. Deep fusion synthesis (LLM-ready callback architecture)
  4. Insert new tetrahedron near path centroid
  5. Trigger self-organization
  6. Track persistent entropy — target >= 18% drop after dream cycles

Core principles enforced:
  - NO deletion/pruning of memories (Eternity principle)
  - Dream memories that would be "pruned" are instead reintegrated
  - All noise is transformed through integration, never removed
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from tetrahedron_memory.phase_transition import PhaseTransitionDetector

import numpy as np

from .persistent_entropy import EntropyTracker, compute_persistent_entropy

logger = logging.getLogger("tetramem.dream")


DreamSynthesisInput = Dict[str, Any]
DreamSynthesisFn = Callable[[List[DreamSynthesisInput]], Optional[str]]


@dataclass
class DreamRecord:
    __slots__ = (
        "dream_id",
        "tetra_id",
        "source_tetra_ids",
        "source_clusters",
        "synthesis_content",
        "fusion_quality",
        "entropy_before",
        "entropy_after",
        "entropy_delta",
        "creation_time",
        "labels",
        "reintegrated",
        "reintegration_count",
        "walk_path_hash",
    )

    dream_id: str
    tetra_id: str
    source_tetra_ids: List[str]
    source_clusters: List[List[str]]
    synthesis_content: str
    fusion_quality: float
    entropy_before: float
    entropy_after: float
    entropy_delta: float
    creation_time: float
    labels: List[str]
    reintegrated: bool
    reintegration_count: int
    walk_path_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dream_id": self.dream_id,
            "tetra_id": self.tetra_id,
            "source_tetra_ids": self.source_tetra_ids,
            "source_clusters": self.source_clusters,
            "synthesis_content": self.synthesis_content[:200],
            "fusion_quality": self.fusion_quality,
            "entropy_before": self.entropy_before,
            "entropy_after": self.entropy_after,
            "entropy_delta": self.entropy_delta,
            "creation_time": self.creation_time,
            "labels": self.labels,
            "reintegrated": self.reintegrated,
            "reintegration_count": self.reintegration_count,
            "walk_path_hash": self.walk_path_hash,
        }


class DreamStore:
    def __init__(self, max_records: int = 500):
        self._records: Dict[str, DreamRecord] = {}
        self._by_source: Dict[str, Set[str]] = defaultdict(set)
        self._max_records = max_records
        self._chronological: List[str] = []

    def record(self, dream: DreamRecord) -> None:
        self._records[dream.dream_id] = dream
        self._chronological.append(dream.dream_id)
        for sid in dream.source_tetra_ids:
            self._by_source[sid].add(dream.dream_id)
        if len(self._records) > self._max_records:
            oldest_id = self._chronological.pop(0)
            old = self._records.pop(oldest_id, None)
            if old:
                for sid in old.source_tetra_ids:
                    self._by_source[sid].discard(oldest_id)

    def get(self, dream_id: str) -> Optional[DreamRecord]:
        return self._records.get(dream_id)

    def get_by_source(self, tetra_id: str) -> List[DreamRecord]:
        dream_ids = self._by_source.get(tetra_id, set())
        return [self._records[did] for did in dream_ids if did in self._records]

    def get_recent(self, n: int = 10) -> List[DreamRecord]:
        ids = self._chronological[-n:]
        return [self._records[did] for did in ids if did in self._records]

    def mark_reintegrated(self, dream_id: str) -> None:
        rec = self._records.get(dream_id)
        if rec:
            rec.reintegrated = True
            rec.reintegration_count += 1

    def quality_stats(self) -> Dict[str, Any]:
        if not self._records:
            return {"count": 0, "avg_quality": 0.0, "avg_entropy_delta": 0.0}
        qualities = [r.fusion_quality for r in self._records.values()]
        deltas = [r.entropy_delta for r in self._records.values()]
        return {
            "count": len(self._records),
            "avg_quality": float(np.mean(qualities)),
            "max_quality": float(max(qualities)),
            "min_quality": float(min(qualities)),
            "avg_entropy_delta": float(np.mean(deltas)),
            "reintegrated_count": sum(1 for r in self._records.values() if r.reintegrated),
        }

    @property
    def size(self) -> int:
        return len(self._records)


def fusion_quality_score(
    source_inputs: List[DreamSynthesisInput],
    synthesized_content: Optional[str],
) -> float:
    """Topology-aware fusion quality score (v2).

    Scoring dimensions:
      1. Source diversity (0-0.15) — number of distinct sources
      2. Label diversity (0-0.10) — unique label spread
      3. Weight balance (0-0.10) — how balanced source weights are
      4. Content richness (0-0.15) — synthesis output length/quality
      5. Topological connectivity (0-0.20) — shared faces/edges between sources
      6. Source depth (0-0.15) — integration_count of sources (deeper = richer)
      7. Centroid dispersion (0-0.15) — spatial spread indicates bridging value
    """
    if not synthesized_content or not source_inputs or len(source_inputs) < 2:
        return 0.0

    n_sources = len(source_inputs)

    # 1. Source diversity
    diversity_bonus = min(n_sources / 5.0, 1.0) * 0.15

    # 2. Label diversity
    all_labels = []
    for inp in source_inputs:
        all_labels.extend(inp.get("labels", []))
    unique_labels = len(set(all_labels) - {"__dream__", "__system__"})
    label_diversity = min(unique_labels / 8.0, 1.0) * 0.10

    # 3. Weight balance
    weight_range = [inp.get("weight", 1.0) for inp in source_inputs]
    w_min, w_max = min(weight_range), max(weight_range)
    weight_balance = 1.0 - min(abs(w_max - w_min) / max(w_max, 0.1), 1.0)
    balance_score = weight_balance * 0.10

    # 4. Content richness
    content_len = len(synthesized_content)
    if content_len < 10:
        richness = 0.0
    elif content_len < 50:
        richness = 0.05
    elif content_len < 200:
        richness = 0.10
    else:
        richness = 0.15

    # 5. Topological connectivity — shared labels as proxy for shared topology
    label_sets = [set(inp.get("labels", [])) - {"__dream__", "__system__"} for inp in source_inputs]
    shared_count = 0
    pair_count = 0
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            pair_count += 1
            if label_sets[i] & label_sets[j]:
                shared_count += 1
    topo_connectivity = (shared_count / max(pair_count, 1)) * 0.20

    # 6. Source depth — higher integration_count means deeper memories being fused
    depths = [inp.get("integration_count", 0) for inp in source_inputs]
    avg_depth = sum(depths) / len(depths)
    depth_score = min(avg_depth / 10.0, 1.0) * 0.15

    # 7. Centroid dispersion — spatially spread sources create more valuable bridges
    centroids = [inp.get("centroid") for inp in source_inputs if inp.get("centroid")]
    dispersion = 0.0
    if len(centroids) >= 2:
        centroid_arr = np.array(centroids)
        pairwise_dists = []
        for i in range(len(centroid_arr)):
            for j in range(i + 1, len(centroid_arr)):
                pairwise_dists.append(float(np.linalg.norm(centroid_arr[i] - centroid_arr[j])))
        if pairwise_dists:
            avg_dist = sum(pairwise_dists) / len(pairwise_dists)
            dispersion = min(avg_dist / 2.0, 1.0) * 0.15

    return min(1.0, diversity_bonus + label_diversity + balance_score + richness + topo_connectivity + depth_score + dispersion)


def default_synthesis(contents: List[str], labels: List[List[str]]) -> Optional[str]:
    all_labels = set()
    for ll in labels:
        all_labels.update(ll)
    all_labels.discard("__dream__")
    all_labels.discard("__system__")
    label_str = ", ".join(all_labels) if all_labels else "general"
    preview = " | ".join(c[:80] for c in contents)
    return f"[dream:{label_str}] Abstracted from: {preview}"


def _build_synthesis_inputs(
    mesh: Any,
    group: List[str],
    max_items: int = 3,
) -> List[DreamSynthesisInput]:
    inputs = []
    for tid in group[:max_items]:
        t = mesh.get_tetrahedron(tid)
        if t is None:
            continue
        inputs.append(
            {
                "content": t.content,
                "labels": list(t.labels),
                "weight": t.weight,
                "centroid": t.centroid.tolist()
                if hasattr(t.centroid, "tolist")
                else list(t.centroid),
                "integration_count": t.integration_count,
                "access_count": t.access_count,
                "metadata": dict(t.metadata),
                "tetra_id": tid,
            }
        )
    return inputs





class DreamProtocol:
    """Three-phase dream protocol: THINK -> EXECUTE -> REFLECT.

    Per the TetraMem-XL v2.0 spec, dreams should follow a structured
    cognition loop rather than blind synthesis:

    THINK: Analyze sources, identify what should be synthesized
    EXECUTE: Produce synthesis (LLM callback or default)
    REFLECT: Evaluate quality, accept/reject, track outcome

    This replaces ad-hoc synthesis_fn with a structured protocol.

    Usage:
        protocol = DreamProtocol(think_fn=my_analyzer, execute_fn=my_llm, reflect_fn=my_evaluator)
        result = protocol.run(source_inputs)
        if result.accepted:
            mesh.store(result.content, ...)
    """

    def __init__(
        self,
        think_fn: Optional[Callable[[List[DreamSynthesisInput]], Optional[Dict[str, Any]]]] = None,
        execute_fn: Optional[Callable[[List[DreamSynthesisInput], Optional[Dict[str, Any]]], Optional[str]]] = None,
        reflect_fn: Optional[Callable[[str, List[DreamSynthesisInput]], float]] = None,
        quality_threshold: float = 0.3,
    ):
        self._think_fn = think_fn
        self._execute_fn = execute_fn
        self._reflect_fn = reflect_fn or fusion_quality_score
        self._quality_threshold = quality_threshold
        self._history: List[Dict[str, Any]] = []
        self._accepted = 0
        self._rejected = 0

    def run(self, source_inputs: List[DreamSynthesisInput]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "phase": "init",
            "analysis": None,
            "content": None,
            "quality": 0.0,
            "accepted": False,
            "attempts": 0,
        }

        # Phase 1: THINK
        result["phase"] = "think"
        analysis = None
        if self._think_fn is not None:
            try:
                analysis = self._think_fn(source_inputs)
            except Exception:
                analysis = None
        if analysis is None:
            analysis = self._default_think(source_inputs)
        result["analysis"] = analysis

        # Phase 2: EXECUTE
        result["phase"] = "execute"
        synthesized = None
        if self._execute_fn is not None:
            try:
                synthesized = self._execute_fn(source_inputs, analysis)
            except Exception:
                synthesized = None
        result["attempts"] = 1

        if synthesized is None:
            synthesized = self._default_execute(source_inputs, analysis)
        result["content"] = synthesized

        # Phase 3: REFLECT
        result["phase"] = "reflect"
        quality = self._reflect_fn(source_inputs, synthesized)
        result["quality"] = quality
        result["accepted"] = quality >= self._quality_threshold
        result["phase"] = "complete"

        if result["accepted"]:
            self._accepted += 1
        else:
            self._rejected += 1
        self._history.append({
            "quality": quality,
            "accepted": result["accepted"],
            "source_count": len(source_inputs),
            "content_preview": (synthesized or "")[:100],
        })
        if len(self._history) > 100:
            self._history = self._history[-50:]

        return result

    def _default_think(self, inputs: List[DreamSynthesisInput]) -> Dict[str, Any]:
        all_labels = set()
        total_weight = 0.0
        max_depth = 0
        for inp in inputs:
            all_labels.update(inp.get("labels", []))
            total_weight += inp.get("weight", 1.0)
            max_depth = max(max_depth, inp.get("integration_count", 0))
        all_labels.discard("__dream__")
        all_labels.discard("__system__")
        return {
            "label_inventory": sorted(all_labels),
            "total_weight": total_weight,
            "max_depth": max_depth,
            "n_sources": len(inputs),
            "strategy": "bridge" if len(all_labels) > 3 else ("deepen" if max_depth > 3 else "surface"),
        }

    def _default_execute(self, inputs: List[DreamSynthesisInput], analysis: Dict[str, Any]) -> Optional[str]:
        if not inputs or len(inputs) < 2:
            return None
        strategy = analysis.get("strategy", "surface")
        labels = analysis.get("label_inventory", [])
        label_str = ", ".join(labels[:4]) if labels else "general"
        primary = max(inputs, key=lambda x: x.get("weight", 1.0))
        content = primary.get("content", "")[:80]
        n = len(inputs)
        return "[dream:" + strategy + ":" + label_str + "] " + content + " + " + str(n - 1) + " related"

    def get_statistics(self) -> Dict[str, Any]:
        if not self._history:
            return {"accepted": 0, "rejected": 0, "acceptance_rate": 0.0}
        return {
            "accepted": self._accepted,
            "rejected": self._rejected,
            "acceptance_rate": self._accepted / max(len(self._history), 1),
            "avg_quality": float(np.mean([h["quality"] for h in self._history])),
            "total_cycles": len(self._history),
        }


class TetraDreamCycle:
    """
    TetraMesh-native dream cycle implementing the Dream Law.

    Parameters
    ----------
    mesh : TetraMesh
        The tetrahedral memory mesh.
    synthesis_fn : callable, optional
        Deep fusion function for production LLM integration.
        Signature: (List[DreamSynthesisInput]) -> Optional[str]
        Receives structured input with content, labels, weights, topology info.
        Return None to skip, or a string for the synthesized memory.
        Replace with LLM call for production:
            def llm_synthesis(inputs):
                prompt = build_fusion_prompt(inputs)
                return llm_client.generate(prompt)
    organizer : callable, optional
        Called with (mesh) after each dream to trigger self-organization.
    cycle_interval : float
        Seconds between dream cycles.
    walk_steps : int
        Number of steps in the random walk.
    max_dream_tetra : int
        Soft cap — when exceeded, oldest dreams get reintegrated (not deleted).
    dream_weight : float
        Initial weight for dream tetrahedra.
    """

    def __init__(
        self,
        mesh: Any,
        synthesis_fn: Optional[DreamSynthesisFn] = None,
        organizer: Optional[Callable] = None,
        cycle_interval: float = 300.0,
        walk_steps: int = 12,
        max_dream_tetra: int = 50,
        dream_weight: float = 0.5,
        legacy_compat_fn: Optional[Callable] = None,
        zigzag_tracker: Optional[Any] = None,
        llm_executor: Optional[Any] = None,
    ):
        self.mesh = mesh
        self.organizer = organizer
        self.cycle_interval = cycle_interval
        self.walk_steps = walk_steps
        self.max_dream_tetra = max_dream_tetra
        self.dream_weight = dream_weight
        self._zigzag_tracker = zigzag_tracker

        if llm_executor is not None:
            self._protocol = DreamProtocol(
                think_fn=llm_executor.think,
                execute_fn=llm_executor.execute,
                reflect_fn=llm_executor.reflect,
                quality_threshold=0.5,
            )
        elif synthesis_fn is not None:
            self._protocol = DreamProtocol(execute_fn=lambda inputs, _: synthesis_fn(inputs))
        elif legacy_compat_fn is not None:
            self._legacy_fn = legacy_compat_fn
            wrapped = self._wrap_legacy_fn(legacy_compat_fn)
            self._protocol = DreamProtocol(execute_fn=lambda inputs, _: wrapped(inputs))
        else:
            self._protocol = DreamProtocol()

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cycle_count = 0
        self._dreams_created = 0
        self._dreams_reintegrated = 0
        self._last_stats: Dict[str, Any] = {}
        self._entropy_tracker = EntropyTracker()
        self._dream_store = DreamStore()
        self._phase_detector = phase_detector or PhaseTransitionDetector()

    def _wrap_legacy_fn(self, fn):
        def wrapper(inputs):
            contents = [i["content"] for i in inputs]
            labels = [i["labels"] for i in inputs]
            return fn(contents, labels)
        return wrapper

    def _is_duplicate_content(self, content: str) -> bool:
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        for tid, t in self.mesh.tetrahedra.items():
            existing_hash = hashlib.sha256(t.content.encode()).hexdigest()[:12]
            if content_hash == existing_hash:
                return True
            if len(content) > 20 and len(t.content) > 20:
                shorter = content if len(content) < len(t.content) else t.content
                longer = t.content if len(content) < len(t.content) else content
                if shorter[:40] in longer:
                    return True
        return False

    def _content_is_just_rewrite(self, synthesized: str, source_contents: list) -> bool:
        if not synthesized or not source_contents:
            return False
        syn_words = set(synthesized)
        for src in source_contents:
            if not src:
                continue
            src_words = set(src)
            if len(syn_words) == 0 or len(src_words) == 0:
                continue
            overlap = len(syn_words & src_words) / max(len(syn_words | src_words), 1)
            if overlap > 0.75:
                return True
        return False

    def _default_deep_synthesis(self, inputs: List[DreamSynthesisInput]) -> Optional[str]:
        if not inputs or len(inputs) < 2:
            return None

        label_counts: Dict[str, int] = {}
        for inp in inputs:
            for lbl in inp.get("labels", []):
                if lbl.startswith("__"):
                    continue
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

        shared_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= 2}
        unique_labels = {lbl for lbl, cnt in label_counts.items() if cnt == 1}

        total_weight = sum(i.get("weight", 1.0) for i in inputs)
        if total_weight <= 0:
            total_weight = 1.0

        sorted_inputs = sorted(inputs, key=lambda x: x.get("weight", 1.0), reverse=True)
        primary = sorted_inputs[0]
        secondary = sorted_inputs[1:]

        depth = sum(i.get("integration_count", 0) for i in inputs)
        depth_tag = "abstract" if depth > 8 else ("deep" if depth > 3 else "surface")

        parts: List[str] = []
        if shared_labels:
            parts.append("Core: " + ", ".join(sorted(shared_labels)))

        core_content = primary["content"][:120]
        if secondary:
            supporting = []
            cum_w = primary.get("weight", 1.0)
            for inp in secondary:
                w = inp.get("weight", 1.0)
                if cum_w + w > total_weight * 0.85:
                    break
                supporting.append(inp["content"][:80])
                cum_w += w
            if supporting:
                parts.append("Built on: " + primary["content"][:80])
                parts.append("Informed by: " + "; ".join(supporting[:3]))
            else:
                parts.append("Essence: " + core_content)
        else:
            parts.append("Essence: " + core_content)

        if unique_labels:
            parts.append("Extends: " + ", ".join(sorted(unique_labels)[:4]))

        topo_desc = []
        conn_types_seen = set()
        for inp in inputs:
            md = inp.get("metadata", {})
            src = md.get("source_clusters")
            if src:
                for cluster in src:
                    for sid in cluster:
                        conn_types_seen.add(sid[:8])
        if conn_types_seen:
            topo_desc.append("Bridges " + str(len(conn_types_seen)) + " regions")

        result = "[dream:" + depth_tag + "] "
        result += " | ".join(parts)
        if topo_desc:
            result += " | " + "; ".join(topo_desc)
        return result

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="tetra-dream", daemon=True)
        self._thread.start()
        logger.info("TetraDreamCycle started (interval=%.0fs)", self.cycle_interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None

    def trigger_now(self) -> Dict[str, Any]:
        stats = self._execute()
        self._last_stats = stats
        return stats

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "cycle_count": self._cycle_count,
            "dreams_created": self._dreams_created,
            "dreams_reintegrated": self._dreams_reintegrated,
            "last_stats": self._last_stats,
            "entropy": self._entropy_tracker.get_summary(),
            "dream_store": self._dream_store.quality_stats(),
        }

    def get_dream_store(self) -> DreamStore:
        return self._dream_store

    def get_dream_trace(self, tetra_id: str) -> List[Dict[str, Any]]:
        records = self._dream_store.get_by_source(tetra_id)
        return [r.to_dict() for r in records]

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self.cycle_interval):
            try:
                stats = self._execute()
                self._last_stats = stats
                self._cycle_count += 1
            except Exception as e:
                logger.error("DreamCycle error: %s", e, exc_info=True)

    def _execute(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "phase": "idle",
            "walk_visited": 0,
            "clusters_found": 0,
            "dreams_created": 0,
            "dreams_reintegrated": 0,
            "entropy_before": 0.0,
            "entropy_after": 0.0,
            "entropy_delta": 0.0,
            "mapping_cone": None,
            "dream_guidance": None,
        }

        guidance = None
        if self._zigzag_tracker is not None:
            guidance = self._zigzag_tracker.get_dream_guidance()
            stats["dream_guidance"] = guidance

        pre_snapshot = None
        if self._zigzag_tracker is not None:
            pre_snapshot = self._zigzag_tracker.record_snapshot(self.mesh)

        tetrahedra = self.mesh.tetrahedra
        if len(tetrahedra) < 3:
            stats["phase"] = "too_few_tetra"
            return stats

        guided_reintegration_threshold = self.max_dream_tetra
        if guidance and guidance.get("guidance") != "no_history":
            if guidance.get("expected_benefit", 0) > 0.3:
                guided_reintegration_threshold = max(1, int(self.max_dream_tetra * 0.6))

        st = self.mesh.compute_ph()
        if st is not None:
            entropy_before = compute_persistent_entropy(st)
            self._entropy_tracker.record(entropy_before)
            stats["entropy_before"] = entropy_before
        elif pre_snapshot is not None:
            entropy_before = pre_snapshot.total_entropy
            stats["entropy_before"] = entropy_before

        regular = {tid: t for tid, t in tetrahedra.items() if "__dream__" not in t.labels}
        if len(regular) < 2:
            stats["phase"] = "no_regular_tetra"
            return stats

        dream_count = sum(1 for t in tetrahedra.values() if "__dream__" in t.labels)

        global_tension, tensions = self._phase_detector.compute_global_tension(self.mesh)
        stats["global_tension"] = global_tension

        if self._phase_detector.should_trigger(global_tension):
            clusters = self._phase_detector.identify_tension_clusters(tensions, self.mesh)
            if clusters:
                pt_result = self._phase_detector.execute_transition(self.mesh, tensions, clusters)
                stats["phase_transition"] = pt_result
                logger.info("Phase transition triggered: tension=%.2f clusters=%d", global_tension, len(clusters))

        if tensions and max(tensions.values()) > 1.0:
            path, path_types = self._tension_guided_walk(regular, tensions)
        else:
            path, path_types = self._random_walk(regular)
        stats["walk_visited"] = len(path)

        if len(path) < 3:
            stats["phase"] = "walk_too_short"
            return stats

        clusters = self._cluster_by_topology(path, path_types)
        stats["clusters_found"] = len(clusters)

        if len(clusters) < 2:
            stats["phase"] = "single_cluster"
            return stats

        if dream_count >= guided_reintegration_threshold:
            reintegrated = self._reintegrate_dreams()
            stats["dreams_reintegrated"] = reintegrated
            self._dreams_reintegrated += reintegrated

        created_ids = self._synthesize_and_insert_tracked(clusters)
        stats["dreams_created"] = len(created_ids)
        self._dreams_created += len(created_ids)

        reintegrated_ids = self._reintegrate_dreams_tracked()
        stats["dreams_reintegrated"] += len(reintegrated_ids)
        self._dreams_reintegrated += len(reintegrated_ids)

        if self.organizer:
            try:
                self.organizer(self.mesh)
            except Exception:
                pass

        st = self.mesh.compute_ph()
        if st is not None:
            entropy_after = compute_persistent_entropy(st)
            self._entropy_tracker.record(entropy_after)
            stats["entropy_after"] = entropy_after
            if stats["entropy_before"] > 0:
                stats["entropy_delta"] = (stats["entropy_before"] - entropy_after) / stats[
                    "entropy_before"
                ]
                for did in created_ids:
                    rec = self._dream_store.get(did[:16] + "0" * (16 - len(did[:16])))
                    if rec is None:
                        for r in self._dream_store._records.values():
                            if r.tetra_id == did:
                                rec = r
                                break
                    if rec is not None:
                        rec.entropy_before = stats["entropy_before"]
                        rec.entropy_after = entropy_after
                        rec.entropy_delta = stats["entropy_delta"]

        if self._zigzag_tracker is not None:
            post_snapshot = self._zigzag_tracker.record_snapshot(self.mesh)
            if pre_snapshot is not None:
                cone = self._zigzag_tracker.construct_mapping_cone(
                    pre_snapshot=pre_snapshot,
                    post_snapshot=post_snapshot,
                    dream_tetra_created=created_ids,
                    dream_tetra_reintegrated=reintegrated_ids,
                )
                stats["mapping_cone"] = {
                    "cycle_id": cone.cycle_id,
                    "entropy_delta": cone.entropy_delta,
                    "stable_features": sum(len(v) for v in cone.stability_cert.values()),
                    "cone_h0": cone.cone_h0,
                    "cone_h1": cone.cone_h1,
                    "cone_h2": cone.cone_h2,
                }

        stats["phase"] = "complete"
        return stats

    def _compute_topological_tension(self, pool):
        tensions = {}
        for tid, tetra in pool.items():
            t_score = 0.0
            neighbors_all = []
            for method in (self.mesh._face_neighbors, self.mesh._edge_neighbors):
                neighbors_all.extend(method(tid))
            if not neighbors_all:
                continue
            n_weights = []
            shared_labels = 0
            for nid in set(neighbors_all):
                nt = self.mesh.get_tetrahedron(nid)
                if nt is None:
                    continue
                n_weights.append(nt.weight)
                if set(tetra.labels) & set(nt.labels):
                    shared_labels += 1
            if n_weights:
                avg_nw = sum(n_weights) / len(n_weights)
                if avg_nw > 1.0 and tetra.weight < avg_nw * 0.5:
                    t_score += 2.0 * (avg_nw - tetra.weight)
                weight_var = sum((w - avg_nw) ** 2 for w in n_weights) / len(n_weights)
                t_score += weight_var * 0.5
            isolation = 1.0 - (shared_labels / max(len(neighbors_all), 1))
            t_score += isolation * 0.8
            tensions[tid] = t_score
        return tensions

    def _tension_guided_walk(self, pool, tensions, steps=8):
        if not tensions:
            return self._random_walk(pool, entropy_bias=True)
        sorted_t = sorted(tensions.items(), key=lambda x: x[1], reverse=True)
        seed_id = sorted_t[0][0] if sorted_t[0][1] > 0 else self._pick_time_priority_seed(pool)
        if seed_id is None:
            seed_id = random.choice(list(pool.keys()))
        visited = [seed_id]
        conn_types = ["tension_seed"]
        current = seed_id
        for _ in range(steps):
            neighbors = self._get_weighted_neighbors(current)
            if not neighbors:
                break
            tension_weighted = []
            for nid, ctype, w in neighbors:
                t = tensions.get(nid, 0.0)
                boosted = w * (1.0 + t * 0.5)
                tension_weighted.append((nid, ctype, boosted))
            weights = [w for _, _, w in tension_weighted]
            total = sum(weights)
            if total == 0:
                break
            idx = random.choices(range(len(tension_weighted)), weights=weights, k=1)[0]
            nid, ctype, _ = tension_weighted[idx]
            visited.append(nid)
            conn_types.append(ctype)
            current = nid
        return visited, conn_types

    def _random_walk(
        self, pool: Dict[str, Any], entropy_bias: bool = True
    ) -> Tuple[List[str], List[str]]:
        seed_id = self._pick_time_priority_seed(pool)
        if seed_id is None:
            seed_id = (
                self._pick_entropy_weighted_seed(pool)
            if entropy_bias
            else random.choice(list(pool.keys()))
        )
        visited = [seed_id]
        conn_types: List[str] = ["seed"]

        current = seed_id
        for _ in range(self.walk_steps):
            neighbors = self._get_weighted_neighbors(current)
            if entropy_bias:
                neighbors = self._apply_entropy_bias(current, neighbors)
            if not neighbors:
                candidates = [tid for tid in pool if tid not in visited]
                if not candidates:
                    break
                current = random.choice(candidates)
                visited.append(current)
                conn_types.append("random")
                continue

            weights = [w for _, _, w in neighbors]
            total = sum(weights)
            if total == 0:
                break
            probs = [w / total for w in weights]

            idx = random.choices(range(len(neighbors)), weights=probs, k=1)[0]
            nid, ctype, _ = neighbors[idx]
            visited.append(nid)
            conn_types.append(ctype)
            current = nid

        return visited, conn_types

    def _pick_time_priority_seed(self, pool: Dict[str, Any]) -> Optional[str]:
        """Seed selection prioritized by time law.

        Memories with high filtration (old + rarely accessed) get
        priority for integration. Time drives consolidation.
        """
        if not pool:
            return None

        time_lambda = self.mesh._time_lambda
        scored = []
        for tid, tetra in pool.items():
            fil = tetra.filtration(time_lambda)
            access_decay = 1.0 / (1.0 + tetra.access_count * 0.1)
            time_priority = fil * access_decay
            scored.append((time_priority, tid))

        scored.sort(reverse=True)

        top_n = min(5, len(scored))
        weights = [s[0] for s in scored[:top_n]]
        import math
        weights = [w if math.isfinite(w) and w > 0 else 1e-6 for w in weights]
        total_priority = sum(weights)
        if total_priority <= 0:
            return None

        import random as _rng
        idx = _rng.choices(range(top_n), weights=weights, k=1)[0]
        return scored[idx][1]

    def _pick_entropy_weighted_seed(self, pool: Dict[str, Any]) -> str:
        candidates = list(pool.keys())
        if len(candidates) <= 1:
            return candidates[0]
        scored = []
        for tid in candidates:
            t = self.mesh.get_tetrahedron(tid)
            if t is None:
                scored.append((tid, 1.0))
                continue
            neighbor_count = 0
            total_weight = 0.0
            for nb_set_method in (
                self.mesh._face_neighbors,
                self.mesh._edge_neighbors,
                self.mesh._vertex_neighbors,
            ):
                for nid in nb_set_method(tid):
                    nt = self.mesh.get_tetrahedron(nid)
                    if nt:
                        neighbor_count += 1
                        total_weight += nt.weight
            avg_w = total_weight / max(neighbor_count, 1)
            diversity = min(neighbor_count, 12) * (1.0 + abs(t.weight - avg_w))
            scored.append((tid, diversity))
        total = sum(s for _, s in scored)
        if total <= 0:
            return random.choice(candidates)
        return random.choices([tid for tid, _ in scored], weights=[s for _, s in scored], k=1)[0]

    def _apply_entropy_bias(
        self, current_id: str, neighbors: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, float]]:
        if not neighbors:
            return neighbors
        current_t = self.mesh.get_tetrahedron(current_id)
        if current_t is None:
            return neighbors
        biased = []
        for nid, ctype, w in neighbors:
            nt = self.mesh.get_tetrahedron(nid)
            if nt is None:
                biased.append((nid, ctype, w))
                continue
            label_diff = len(set(current_t.labels) ^ set(nt.labels))
            entropy_boost = 1.0 + 0.15 * min(label_diff, 5)
            low_w_penalty = 1.0
            if nt.weight < 0.5:
                low_w_penalty = 1.5
            biased.append((nid, ctype, w * entropy_boost * low_w_penalty))
        return biased

    def _get_weighted_neighbors(self, tetra_id: str) -> List[Tuple[str, str, float]]:
        face_nb = self.mesh._face_neighbors(tetra_id)
        edge_nb = self.mesh._edge_neighbors(tetra_id)
        vertex_nb = self.mesh._vertex_neighbors(tetra_id)

        neighbors = []
        for nid in face_nb:
            t = self.mesh.get_tetrahedron(nid)
            if t:
                neighbors.append((nid, "face", t.weight * 3.0))
        for nid in edge_nb:
            t = self.mesh.get_tetrahedron(nid)
            if t:
                neighbors.append((nid, "edge", t.weight * 1.5))
        for nid in vertex_nb:
            t = self.mesh.get_tetrahedron(nid)
            if t:
                neighbors.append((nid, "vertex", t.weight * 0.5))
        return neighbors

    def _cluster_by_topology(self, path: List[str], path_types: List[str]) -> List[List[str]]:
        face_groups: Dict[str, List[str]] = {}
        for i in range(1, len(path)):
            if path_types[i] == "face" and i > 0:
                key = path[i - 1]
                face_groups.setdefault(key, []).append(path[i])

        clusters = [members for members in face_groups.values() if len(members) >= 2]

        if len(clusters) < 2:
            mid = len(path) // 2
            clusters = [path[:mid], path[mid:]]

        return clusters

    def _synthesize_and_insert_tracked(self, clusters: List[List[str]]) -> List[str]:
        created_ids = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                group_a = clusters[i]
                group_b = clusters[j]

                inputs_a = _build_synthesis_inputs(self.mesh, group_a)
                inputs_b = _build_synthesis_inputs(self.mesh, group_b)

                if not inputs_a or not inputs_b:
                    continue

                all_inputs = inputs_a + inputs_b

                protocol_result = self._protocol.run(all_inputs)
                synthesized = protocol_result.get("content")
                if synthesized is None or not protocol_result.get("accepted", False):
                    continue

                if self._is_duplicate_content(synthesized):
                    continue

                source_contents = [inp.get("content", "") for inp in all_inputs]
                if self._content_is_just_rewrite(synthesized, source_contents):
                    continue

                centroids_a = [np.array(inp["centroid"]) for inp in inputs_a if inp.get("centroid")]
                centroids_b = [np.array(inp["centroid"]) for inp in inputs_b if inp.get("centroid")]

                if centroids_a and centroids_b:
                    bridge_point = (
                        np.mean(centroids_a, axis=0) + np.mean(centroids_b, axis=0)
                    ) / 2.0
                    bridge_point += np.random.normal(0, 0.02, size=3)
                else:
                    bridge_point = np.random.randn(3)
                    bridge_point /= np.linalg.norm(bridge_point) + 1e-12

                labels_a = set()
                for inp in inputs_a:
                    labels_a.update(inp.get("labels", []))
                labels_b = set()
                for inp in inputs_b:
                    labels_b.update(inp.get("labels", []))
                labels_a.discard("__dream__")
                labels_a.discard("__system__")
                labels_b.discard("__dream__")
                labels_b.discard("__system__")
                bridge_labels = labels_a & labels_b
                if not bridge_labels:
                    bridge_labels = (labels_a | labels_b) - {"__dream__", "__system__"}
                bridge_labels = list(bridge_labels)[:6]

                source_ids_a = [inp["tetra_id"] for inp in inputs_a if "tetra_id" in inp]
                source_ids_b = [inp["tetra_id"] for inp in inputs_b if "tetra_id" in inp]
                all_source_ids = source_ids_a + source_ids_b

                quality = protocol_result.get("quality", 0.0)
                confidence = "high" if quality >= 0.5 else "low"
                dream_weight = self.dream_weight * quality if quality > 0 else self.dream_weight * 0.1
                dream_weight = max(dream_weight, self.dream_weight)
                dream_labels = list(bridge_labels) + ["__dream__"]
                if confidence == "low":
                    dream_labels.append("low_confidence")

                walk_hash = hashlib.md5(
                    ("_".join(group_a[:3] + group_b[:3])).encode()
                ).hexdigest()[:8] if group_a and group_b else "empty"

                tid = self.mesh.store(
                    content=synthesized,
                    seed_point=bridge_point,
                    labels=dream_labels,
                    metadata={
                        "type": "dream",
                        "source_clusters": [source_ids_a[:3], source_ids_b[:3]],
                        "fusion_depth": len(all_inputs),
                        "fusion_quality": quality,
                        "confidence": confidence,
                    },
                    weight=dream_weight,
                )

                dream_rec = DreamRecord(
                    dream_id=hashlib.sha256(
                        (tid + str(time.time())).encode()
                    ).hexdigest()[:16],
                    tetra_id=tid,
                    source_tetra_ids=all_source_ids[:10],
                    source_clusters=[source_ids_a[:3], source_ids_b[:3]],
                    synthesis_content=synthesized,
                    fusion_quality=quality,
                    entropy_before=0.0,
                    entropy_after=0.0,
                    entropy_delta=0.0,
                    creation_time=time.time(),
                    labels=list(bridge_labels),
                    reintegrated=False,
                    reintegration_count=0,
                    walk_path_hash=walk_hash,
                )
                self._dream_store.record(dream_rec)

                created_ids.append(tid)
                logger.info("Dream tetra created: %s (quality=%.2f)", tid[:8], quality)

        return created_ids

    def _reintegrate_dreams(self) -> int:
        return len(self._reintegrate_dreams_tracked())

    def _reintegrate_dreams_tracked(self) -> List[str]:
        to_reintegrate = []
        reintegration_threshold = self.dream_weight * 0.3
        with self.mesh._lock:
            for tid, tetra in self.mesh.tetrahedra.items():
                if "__dream__" in tetra.labels:
                    tetra.weight *= 0.85
                    if tetra.weight < reintegration_threshold:
                        to_reintegrate.append(tid)

            for tid in to_reintegrate:
                tetra = self.mesh.get_tetrahedron(tid)
                if tetra is not None:
                    tetra.catalyze_integration(strength=2.0)
                    for rec in self._dream_store._records.values():
                        if rec.tetra_id == tid:
                            self._dream_store.mark_reintegrated(rec.dream_id)
                            break

        return to_reintegrate
