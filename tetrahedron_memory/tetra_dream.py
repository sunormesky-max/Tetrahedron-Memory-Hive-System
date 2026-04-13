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

import logging
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .persistent_entropy import EntropyTracker, compute_persistent_entropy

logger = logging.getLogger("tetramem.dream")


DreamSynthesisInput = Dict[str, Any]
DreamSynthesisFn = Callable[[List[DreamSynthesisInput]], Optional[str]]


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
    ):
        self.mesh = mesh
        self.organizer = organizer
        self.cycle_interval = cycle_interval
        self.walk_steps = walk_steps
        self.max_dream_tetra = max_dream_tetra
        self.dream_weight = dream_weight
        self._zigzag_tracker = zigzag_tracker

        if synthesis_fn is not None:
            self.synthesis_fn = synthesis_fn
        elif legacy_compat_fn is not None:
            self._legacy_fn = legacy_compat_fn
            self.synthesis_fn = self._wrap_legacy_fn(legacy_compat_fn)
        else:
            self._legacy_fn = default_synthesis
            self.synthesis_fn = self._default_deep_synthesis

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cycle_count = 0
        self._dreams_created = 0
        self._dreams_reintegrated = 0
        self._last_stats: Dict[str, Any] = {}
        self._entropy_tracker = EntropyTracker()

    def _wrap_legacy_fn(self, fn):
        def wrapper(inputs):
            contents = [i["content"] for i in inputs]
            labels = [i["labels"] for i in inputs]
            return fn(contents, labels)

        return wrapper

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
        }

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

    def _random_walk(
        self, pool: Dict[str, Any], entropy_bias: bool = True
    ) -> Tuple[List[str], List[str]]:
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

    def _synthesize_and_insert(self, clusters: List[List[str]]) -> int:
        created = 0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                group_a = clusters[i]
                group_b = clusters[j]

                inputs_a = _build_synthesis_inputs(self.mesh, group_a)
                inputs_b = _build_synthesis_inputs(self.mesh, group_b)

                if not inputs_a or not inputs_b:
                    continue

                all_inputs = inputs_a + inputs_b

                synthesized = self.synthesis_fn(all_inputs)
                if synthesized is None:
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

                shared_labels = set()
                for inp in all_inputs:
                    shared_labels.update(inp.get("labels", []))
                shared_labels.discard("__dream__")
                shared_labels.discard("__system__")

                source_ids_a = [inp["tetra_id"] for inp in inputs_a if "tetra_id" in inp]
                source_ids_b = [inp["tetra_id"] for inp in inputs_b if "tetra_id" in inp]

                tid = self.mesh.store(
                    content=synthesized,
                    seed_point=bridge_point,
                    labels=list(shared_labels) + ["__dream__"],
                    metadata={
                        "type": "dream",
                        "source_clusters": [source_ids_a[:3], source_ids_b[:3]],
                        "fusion_depth": len(all_inputs),
                    },
                    weight=self.dream_weight,
                )
                created += 1
                logger.info("Dream tetra created: %s", tid[:8])

        return created

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

                synthesized = self.synthesis_fn(all_inputs)
                if synthesized is None:
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

                shared_labels = set()
                for inp in all_inputs:
                    shared_labels.update(inp.get("labels", []))
                shared_labels.discard("__dream__")
                shared_labels.discard("__system__")

                source_ids_a = [inp["tetra_id"] for inp in inputs_a if "tetra_id" in inp]
                source_ids_b = [inp["tetra_id"] for inp in inputs_b if "tetra_id" in inp]

                tid = self.mesh.store(
                    content=synthesized,
                    seed_point=bridge_point,
                    labels=list(shared_labels) + ["__dream__"],
                    metadata={
                        "type": "dream",
                        "source_clusters": [source_ids_a[:3], source_ids_b[:3]],
                        "fusion_depth": len(all_inputs),
                    },
                    weight=self.dream_weight,
                )
                created_ids.append(tid)
                logger.info("Dream tetra created: %s", tid[:8])

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

        return to_reintegrate
