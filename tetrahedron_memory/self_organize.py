from __future__ import annotations

import hashlib
import logging
import math
import random
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .pcnn_types import PCNNConfig, PulseType

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField

logger = logging.getLogger("tetramem.honeycomb")


class SemanticCluster:
    __slots__ = ("cluster_id", "labels", "node_ids", "centroid", "avg_weight", "total_activation", "quality_score")

    def __init__(self, cluster_id: str, labels: Set[str]):
        self.cluster_id = cluster_id
        self.labels = labels
        self.node_ids: List[str] = []
        self.centroid: Optional[np.ndarray] = None
        self.avg_weight: float = 0.0
        self.total_activation: float = 0.0
        self.quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "labels": list(self.labels),
            "node_count": len(self.node_ids),
            "avg_weight": round(self.avg_weight, 3),
            "total_activation": round(self.total_activation, 3),
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "quality_score": round(self.quality_score, 3),
        }


class OrganizeResult:
    __slots__ = (
        "organize_time", "clusters_found", "clusters_reinforced",
        "entropy_before", "entropy_after", "consolidations_done",
        "shortcuts_created", "migrations_done", "details",
        "clusters_promoted", "bridges_created",
    )

    def __init__(self):
        self.organize_time: float = time.time()
        self.clusters_found: int = 0
        self.clusters_reinforced: int = 0
        self.entropy_before: float = 0.0
        self.entropy_after: float = 0.0
        self.consolidations_done: int = 0
        self.shortcuts_created: int = 0
        self.migrations_done: int = 0
        self.details: str = ""
        self.clusters_promoted: int = 0
        self.bridges_created: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "organize_time": self.organize_time,
            "clusters_found": self.clusters_found,
            "clusters_reinforced": self.clusters_reinforced,
            "entropy_before": round(self.entropy_before, 4),
            "entropy_after": round(self.entropy_after, 4),
            "consolidations_done": self.consolidations_done,
            "shortcuts_created": self.shortcuts_created,
            "migrations_done": self.migrations_done,
            "details": self.details,
            "clusters_promoted": self.clusters_promoted,
            "bridges_created": self.bridges_created,
        }


class SelfOrganizeEngine:
    """
    Unified self-organization engine — autonomous topology optimization.

    Four modules per cycle:
      1. Cluster Detection: label-cooccurrence + spatial proximity grouping
      2. Entropy Balance: weight redistribution to prevent concentration
      3. Memory Consolidation: merge weak duplicate memories
      4. Topological Shortcuts: virtual edges for semantically close but
         topologically distant memory pairs

    Runs periodically in the pulse loop, separate from SelfCheckEngine
    which handles anomaly detection and repair.
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._history: List[OrganizeResult] = []
        self._max_history = 30
        self._clusters: List[SemanticCluster] = []
        self._shortcuts: Dict[Tuple[str, str], float] = {}
        self._shortcut_by_node: Dict[str, List[Tuple[Tuple[str, str], float]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._last_entropy: float = 0.0

    def should_run_early(self) -> bool:
        cfg = PCNNConfig
        return self._last_entropy > cfg.ENTROPY_HIGH_THRESHOLD

    def run_cycle(self) -> OrganizeResult:
        result = OrganizeResult()
        field = self._field

        with field._lock:
            occupied = [(nid, n) for nid, n in field._nodes.items() if n.is_occupied]
            if len(occupied) < 3:
                result.details = "insufficient nodes for organization"
                with self._lock:
                    self._history.append(result)
                    if len(self._history) > self._max_history:
                        self._history = self._history[-self._max_history // 2:]
                return result

            result.entropy_before = self._compute_entropy(occupied)
            self._last_entropy = result.entropy_before

            self._detect_clusters(field, occupied, result)
            self._rebalance_entropy(field, occupied, result)
            self._migrate_memories(field, occupied, result)
            self._consolidate_memories(field, occupied, result)
            self._create_shortcuts(field, occupied, result)
            self._create_knowledge_bridges(field, result)

            result.entropy_after = self._compute_entropy(occupied)
            self._reinforce_clusters(field, result)
            self._promote_clusters(field, result)

            with self._lock:
                self._history.append(result)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history // 2:]

        return result

    def _compute_entropy(self, occupied: List[Tuple[str, Any]]) -> float:
        if not occupied:
            return 0.0
        weights = [n.weight for _, n in occupied]
        total = sum(weights)
        if total <= 0:
            return 0.0
        probs = [w / total for w in weights if w > 0]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
        return entropy / max(max_entropy, 1e-10)

    def _detect_clusters(self, field, occupied, result: OrganizeResult):
        cfg = PCNNConfig
        label_groups: Dict[str, List[str]] = defaultdict(list)
        for nid, node in occupied:
            for lbl in node.labels:
                if not lbl.startswith("__"):
                    label_groups[lbl].append(nid)

        candidate_labels = [
            lbl for lbl, nids in label_groups.items()
            if len(nids) >= cfg.CLUSTER_MIN_SIZE
        ]

        clusters: List[SemanticCluster] = []
        used_nodes: Set[str] = set()

        for lbl in sorted(candidate_labels, key=lambda l: -len(label_groups[l]))[:cfg.CLUSTER_MAX_LABELS]:
            nids = label_groups[lbl]
            available = [n for n in nids if n not in used_nodes]
            if len(available) < cfg.CLUSTER_MIN_SIZE:
                continue

            cluster_labels = {lbl}
            cluster_nodes = []
            for nid in available:
                node = field._nodes.get(nid)
                if node is None:
                    continue
                other_labels = set(node.labels) - {"__pulse_bridge__", "__system__"}
                cluster_labels.update(other_labels)
                cluster_nodes.append((nid, node))

            if len(cluster_nodes) < cfg.CLUSTER_MIN_SIZE:
                continue

            positions = [n.position for _, n in cluster_nodes]
            centroid = np.mean(positions, axis=0)
            avg_w = float(np.mean([n.weight for _, n in cluster_nodes]))
            total_act = sum(n.activation for _, n in cluster_nodes)

            node_ids_set = {nid for nid, _ in cluster_nodes}
            internal_links = 0
            total_possible = len(cluster_nodes) * 4
            for nid, n in cluster_nodes:
                for fnid in n.face_neighbors[:6] + n.edge_neighbors[:4]:
                    if fnid in node_ids_set:
                        internal_links += 1
            connectivity = internal_links / max(total_possible, 1)

            all_label_sets = [set(n.labels) - {"__pulse_bridge__", "__system__"} for _, n in cluster_nodes]
            if len(all_label_sets) > 1:
                label_intersections = []
                for i in range(len(all_label_sets)):
                    for j in range(i + 1, len(all_label_sets)):
                        shared = len(all_label_sets[i] & all_label_sets[j])
                        total = len(all_label_sets[i] | all_label_sets[j])
                        label_intersections.append(shared / max(total, 1))
                label_coherence = float(np.mean(label_intersections)) if label_intersections else 0.0
            else:
                label_coherence = 1.0

            distances = [float(np.linalg.norm(p - centroid)) for p in positions]
            avg_dist = float(np.mean(distances)) if distances else 0.0
            spacing = field._spacing if field._spacing > 0 else 1.0
            spatial_compactness = max(0.0, 1.0 - avg_dist / (spacing * 6))

            quality_score = 0.35 * connectivity + 0.35 * label_coherence + 0.30 * spatial_compactness

            cluster = SemanticCluster(f"cluster_{lbl}", cluster_labels)
            cluster.node_ids = [nid for nid, _ in cluster_nodes]
            cluster.centroid = centroid
            cluster.avg_weight = avg_w
            cluster.total_activation = total_act
            cluster.quality_score = quality_score
            clusters.append(cluster)

            for nid, _ in cluster_nodes:
                used_nodes.add(nid)

        self._clusters = clusters
        result.clusters_found = len(clusters)

    def _reinforce_clusters(self, field, result: OrganizeResult):
        for cluster in self._clusters:
            if len(cluster.node_ids) < 2:
                continue
            for nid in cluster.node_ids:
                node = field._nodes.get(nid)
                if node and node.is_occupied:
                    neighbor_in_cluster = 0
                    for fnid in node.face_neighbors[:6]:
                        fn = field._nodes.get(fnid)
                        if fn and fn.is_occupied:
                            shared = set(fn.labels) & cluster.labels
                            if len(shared) > 0:
                                neighbor_in_cluster += 1
                    if neighbor_in_cluster < 2:
                        boost = cluster.avg_weight * 0.05
                        node.activation = min(10.0, node.activation + boost)
                        field._hebbian.record_path(
                            [cluster.node_ids[0], nid],
                            success=True,
                            strength=cluster.avg_weight * 0.3,
                        )
                        result.clusters_reinforced += 1

    def _rebalance_entropy(self, field, occupied, result: OrganizeResult):
        cfg = PCNNConfig
        weights = [n.weight for _, n in occupied]
        avg_w = float(np.mean(weights))
        max_w = float(max(weights))
        min_w = float(min(weights))

        if max_w <= 0 or avg_w <= 0:
            return

        concentration = max_w / (avg_w + 1e-10)

        if concentration > 5.0:
            for nid, node in occupied:
                if node.weight > avg_w * 4:
                    suppress = node.weight * cfg.ENTROPY_SUPPRESS_FACTOR
                    node.weight = max(min_w, node.weight - suppress)
                elif node.weight < avg_w * 0.3:
                    boost = avg_w * cfg.ENTROPY_BOOST_FACTOR
                    node.weight = min(max_w, node.weight + boost)
                    node.activation = min(10.0, node.activation + boost * 0.5)

    def _migrate_memories(self, field, occupied, result: OrganizeResult):
        occupied_count = len(occupied)
        if occupied_count < 20:
            return
        max_migrations = min(5, max(1, occupied_count // 200))
        quality_list = []
        for nid, node in occupied:
            if node.weight < 0.5 or "__consolidated__" in node.labels:
                continue
            geo_q = node.metadata.get("geometric_quality", 0.5)
            bcc_c = node.metadata.get("bcc_cell_coherence", 0.5)
            neighbors_occ = sum(
                1 for fnid in node.face_neighbors[:6]
                if field._nodes.get(fnid) and field._nodes[fnid].is_occupied
            )
            density_penalty = max(0, (neighbors_occ - 4) * 0.08)
            quality = 0.4 * geo_q + 0.3 * bcc_c + 0.3 * (node.weight / 10.0) - density_penalty
            quality_list.append((nid, node, quality))
        quality_list.sort(key=lambda x: x[2])
        migrated = 0
        for nid, node, q in quality_list[:15]:
            if migrated >= max_migrations:
                break
            node_labels = set(node.labels) - {
                "__pulse_bridge__", "__system__", "__consolidated__", "__low_priority__"
            }
            best_target = None
            best_ts = q + 0.12
            search_pool = node.face_neighbors + node.edge_neighbors + node.vertex_neighbors
            for fnid in search_pool:
                fn = field._nodes.get(fnid)
                if fn is None or fn.is_occupied:
                    continue
                t_geo = field._compute_node_geometric_quality(fnid)
                t_bcc = field._bcc_cell_coherence(fnid)
                label_match = 0
                for fnn in fn.face_neighbors[:6]:
                    fnn_n = field._nodes.get(fnn)
                    if fnn_n and fnn_n.is_occupied and node_labels & set(fnn_n.labels):
                        label_match += 1
                t_cell_q = field._cell_quality_factor(fnid)
                ts = 0.3 * t_geo + 0.25 * t_bcc + 0.25 * min(label_match / 3.0, 1.0) + 0.2 * t_cell_q
                if ts > best_ts:
                    best_target = fnid
                    best_ts = ts
            if best_target is None:
                continue
            target = field._nodes[best_target]
            target.content = node.content
            target.labels = list(node.labels)
            target.weight = node.weight
            target.activation = node.activation
            target.base_activation = node.base_activation
            target.metadata = dict(node.metadata)
            target.creation_time = node.creation_time
            target.feeding = node.feeding
            target.crystal_channels = dict(node.crystal_channels)
            target.metadata["geometric_quality"] = float(field._compute_node_geometric_quality(best_target))
            target.metadata["geo_topo_divergence"] = float(field._compute_geometric_topo_divergence(best_target))
            target.metadata["bcc_cell_coherence"] = float(field._bcc_cell_coherence(best_target))
            target.metadata["migrated_from"] = nid[:12]
            target.metadata["migration_time"] = time.time()
            target.touch()
            chash = hashlib.sha256(target.content.encode()).hexdigest()[:12]
            field._content_hash_index[chash] = best_target
            for lbl in target.labels:
                if not lbl.startswith("__"):
                    field._label_index[lbl].discard(nid)
                    field._label_index[lbl].add(best_target)
            for tok in field._extract_tokens(target.content):
                field._content_token_index[tok].discard(nid)
                field._content_token_index[tok].add(best_target)
            node.content = None
            node.labels = []
            node.weight = 0.0
            node.activation = 0.0
            node.metadata = {}
            node.crystal_channels = {}
            if nid in field._occupied_ids:
                field._occupied_ids.discard(nid)
                if field._occupied_count > 0:
                    field._occupied_count -= 1
            field._emit_pulse(best_target, strength=target.weight * 0.3, pulse_type=PulseType.REINFORCING)
            migrated += 1
        result.migrations_done = migrated

    def _consolidate_memories(self, field, occupied, result: OrganizeResult):
        cfg = PCNNConfig
        low_weight = [(nid, n) for nid, n in occupied if n.weight < 1.0 and n.is_occupied]
        if len(low_weight) < 2:
            return

        consolidated = 0
        checked = set()
        low_map = {nid: n for nid, n in low_weight}

        for nid_a, node_a in low_weight:
            if consolidated >= cfg.CONSOLIDATION_MAX_PER_CYCLE:
                break
            if nid_a in checked:
                continue
            if node_a.crystal_channels:
                continue
            if "__low_priority__" in node_a.labels:
                continue

            tokens_a = field._extract_tokens(node_a.content)
            if not tokens_a:
                continue

            best_match = None
            best_sim = 0.0

            candidate_nids = set()
            for tok in tokens_a:
                candidate_nids.update(field._content_token_index.get(tok, set()))
            candidate_nids.discard(nid_a)

            for nid_b in candidate_nids:
                if nid_b in checked:
                    continue
                node_b = low_map.get(nid_b)
                if node_b is None:
                    continue
                if node_b.crystal_channels:
                    continue

                tokens_b = field._extract_tokens(node_b.content)
                if not tokens_b:
                    continue

                intersection = len(tokens_a & tokens_b)
                union = len(tokens_a | tokens_b)
                if union == 0:
                    continue
                jaccard = intersection / union

                labels_a = set(node_a.labels) - {"__pulse_bridge__", "__system__", "__consolidated__"}
                labels_b = set(node_b.labels) - {"__pulse_bridge__", "__system__", "__consolidated__"}
                labels_differ = bool(labels_a - labels_b) or bool(labels_b - labels_a)

                if jaccard >= cfg.CONSOLIDATION_MIN_SIMILARITY and jaccard > best_sim and labels_differ:
                    best_sim = jaccard
                    best_match = (nid_b, node_b)

            if best_match is None:
                continue

            nid_b, node_b = best_match
            receiver, donor = (node_a, node_b) if node_a.weight >= node_b.weight else (node_b, node_a)
            receiver_nid, donor_nid = (nid_a, nid_b) if node_a.weight >= node_b.weight else (nid_b, nid_a)

            transfer = donor.weight * cfg.CONSOLIDATION_WEIGHT_TRANSFER
            receiver.weight = min(10.0, receiver.weight + transfer)
            receiver.activation = min(10.0, receiver.activation + donor.activation * 0.3)
            receiver.labels = list(set(receiver.labels) | set(donor.labels))
            receiver.labels = [l for l in receiver.labels if not l.startswith("__duplicate")]

            donor.metadata["consolidated_into"] = receiver_nid[:12]
            donor.metadata["consolidation_weight"] = round(donor.weight, 3)
            donor.labels.append("__consolidated__")
            donor.weight *= (1.0 - cfg.CONSOLIDATION_WEIGHT_TRANSFER)
            donor.activation *= 0.3

            field._hebbian.record_path(
                [receiver_nid, donor_nid], success=True, strength=receiver.weight * 0.2
            )

            checked.add(nid_a)
            checked.add(nid_b)
            consolidated += 1

        result.consolidations_done = consolidated

    def _create_shortcuts(self, field, occupied, result: OrganizeResult):
        cfg = PCNNConfig
        shortcuts_this_cycle = 0
        exclude = {"__pulse_bridge__", "__system__", "__consolidated__", "__dream__"}
        label_inv: Dict[str, List[str]] = defaultdict(list)
        for nid, node in occupied:
            for lbl in node.labels:
                if lbl not in exclude:
                    label_inv[lbl].append(nid)

        candidate_pairs: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for lbl, nids in label_inv.items():
            if len(nids) < 2:
                continue
            sampled = nids if len(nids) <= 30 else random.sample(nids, 30)
            for i in range(len(sampled)):
                for j in range(i + 1, len(sampled)):
                    a, b = sampled[i], sampled[j]
                    key = (min(a, b), max(a, b))
                    candidate_pairs[key].add(lbl)

        scored: List[Tuple[float, Tuple[str, str], Set[str]]] = []
        for key, shared in candidate_pairs.items():
            if len(shared) < cfg.SHORTCUT_MIN_LABEL_OVERLAP:
                continue
            if key in self._shortcuts:
                continue
            na, nb = field._nodes.get(key[0]), field._nodes.get(key[1])
            if na is None or nb is None:
                continue
            if nb.id in na.face_neighbors or nb.id in na.edge_neighbors:
                continue
            all_labels_a = set(na.labels) - exclude
            all_labels_b = set(nb.labels) - exclude
            jaccard = len(shared) / max(len(all_labels_a | all_labels_b), 1)
            scored.append((jaccard, key, shared))
        scored.sort(key=lambda x: -x[0])

        for jaccard, key, shared in scored[:200]:
            if shortcuts_this_cycle >= cfg.SHORTCUT_MAX_PER_CYCLE:
                break
            nid_a, nid_b = key
            distance = self._topo_distance(field, nid_a, nid_b)
            if distance is None or distance <= 2 or distance > cfg.SHORTCUT_MAX_DISTANCE:
                continue
            all_labels_a = set(field._nodes[nid_a].labels) - exclude
            all_labels_b = set(field._nodes[nid_b].labels) - exclude
            strength = cfg.SHORTCUT_VIRTUAL_STRENGTH * len(shared) / max(len(all_labels_a | all_labels_b), 1)
            self._shortcuts[key] = strength
            self._shortcut_by_node[nid_a].append((key, strength))
            self._shortcut_by_node[nid_b].append((key, strength))
            field._hebbian.record_path(
                [nid_a, nid_b], success=True, strength=strength * 2.0
            )
            shortcuts_this_cycle += 1

        if len(self._shortcuts) > 500:
            sorted_s = sorted(self._shortcuts.items(), key=lambda x: x[1])
            for k, _ in sorted_s[:len(self._shortcuts) - 300]:
                del self._shortcuts[k]
            self._shortcut_by_node.clear()
            for sc_key, sc_str in self._shortcuts.items():
                self._shortcut_by_node[sc_key[0]].append((sc_key, sc_str))
                self._shortcut_by_node[sc_key[1]].append((sc_key, sc_str))

        result.shortcuts_created = shortcuts_this_cycle

    def _topo_distance(self, field, nid_a: str, nid_b: str) -> Optional[int]:
        if nid_a == nid_b:
            return 0
        visited = {nid_a}
        frontier = [nid_a]
        for depth in range(8):
            next_frontier = []
            for fid in frontier:
                fn = field._nodes.get(fid)
                if fn is None:
                    continue
                for nnid in fn.face_neighbors[:6] + fn.edge_neighbors[:4]:
                    if nnid == nid_b:
                        return depth + 1
                    if nnid not in visited:
                        visited.add(nnid)
                        next_frontier.append(nnid)
            frontier = next_frontier
            if not frontier:
                break
        return None

    def _promote_clusters(self, field, result: OrganizeResult):
        cfg = PCNNConfig
        promoted = 0
        for cluster in self._clusters:
            if cluster.quality_score < cfg.CLUSTER_QUALITY_PROMOTION_THRESHOLD:
                continue
            if len(cluster.node_ids) < 2:
                continue
            node_ids_set = set(cluster.node_ids)
            for nid_a in cluster.node_ids:
                node_a = field._nodes.get(nid_a)
                if node_a is None or not node_a.is_occupied:
                    continue
                for nid_b in node_a.face_neighbors + node_a.edge_neighbors:
                    if nid_b not in node_ids_set or nid_b <= nid_a:
                        continue
                    node_b = field._nodes.get(nid_b)
                    if node_b is None or not node_b.is_occupied:
                        continue
                    field._hebbian.record_path(
                        [nid_a, nid_b],
                        success=True,
                        strength=cluster.avg_weight * 0.5,
                    )
                    if not node_a.crystal_channels.get(nid_b):
                        node_a.crystal_channels[nid_b] = cluster.quality_score * 0.3
                    if not node_b.crystal_channels.get(nid_a):
                        node_b.crystal_channels[nid_a] = cluster.quality_score * 0.3
            promoted += 1
        result.clusters_promoted = promoted

    def _create_knowledge_bridges(self, field, result: OrganizeResult):
        if len(self._clusters) < 2:
            return
        bridges = 0
        max_bridges = 3
        for i in range(len(self._clusters)):
            if bridges >= max_bridges:
                break
            ca = self._clusters[i]
            for j in range(i + 1, len(self._clusters)):
                if bridges >= max_bridges:
                    break
                cb = self._clusters[j]
                shared_labels = ca.labels & cb.labels
                if shared_labels:
                    continue
                if ca.centroid is None or cb.centroid is None:
                    continue
                dist = float(np.linalg.norm(ca.centroid - cb.centroid))
                spacing = field._spacing if field._spacing > 0 else 1.0
                if dist > spacing * 10:
                    continue
                edge_nids_a = set()
                for nid in ca.node_ids:
                    n = field._nodes.get(nid)
                    if n:
                        for fnid in n.face_neighbors + n.edge_neighbors:
                            edge_nids_a.add(fnid)
                for nid in cb.node_ids:
                    if nid in edge_nids_a:
                        shared_labels = ca.labels | cb.labels
                        bridge_labels = list(shared_labels)[:3] + ["__pulse_bridge__"]
                        bridge_content = (
                            f"[bridge] {ca.cluster_id} ↔ {cb.cluster_id} "
                            f"| 桥接知识簇: {','.join(list(ca.labels)[:2])} ↔ {','.join(list(cb.labels)[:2])}"
                        )
                        try:
                            bridge_id = field.store(
                                content=bridge_content,
                                labels=bridge_labels,
                                weight=min(ca.avg_weight, cb.avg_weight) * 0.7,
                                metadata={
                                    "bridge_type": "knowledge_bridge",
                                    "cluster_a": ca.cluster_id,
                                    "cluster_b": cb.cluster_id,
                                    "distance": round(dist, 2),
                                },
                            )
                            bridges += 1
                        except Exception as e:
                            logger.debug("Knowledge bridge creation failed: %s", e)
                        break
        result.bridges_created = bridges

    def get_clusters(self) -> List[Dict]:
        with self._lock:
            return [c.to_dict() for c in self._clusters]

    def get_shortcuts(self, n: int = 20) -> List[Dict]:
        with self._lock:
            sorted_s = sorted(self._shortcuts.items(), key=lambda x: -x[1])[:n]
            return [
                {"nodes": (k[0][:8], k[1][:8]), "strength": round(v, 3)}
                for k, v in sorted_s
            ]

    def get_history(self, n: int = 10) -> List[Dict]:
        with self._lock:
            return [r.to_dict() for r in self._history[-n:]]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            avg_quality = (
                float(np.mean([c.quality_score for c in self._clusters]))
                if self._clusters else 0.0
            )
            return {
                "total_organize_cycles": len(self._history),
                "active_clusters": len(self._clusters),
                "active_shortcuts": len(self._shortcuts),
                "latest_entropy": self._history[-1].entropy_after if self._history else None,
                "total_consolidations": sum(r.consolidations_done for r in self._history),
                "total_shortcuts_created": sum(r.shortcuts_created for r in self._history),
                "total_clusters_promoted": sum(r.clusters_promoted for r in self._history),
                "total_bridges_created": sum(r.bridges_created for r in self._history),
                "avg_cluster_quality": round(avg_quality, 3),
                "last_entropy": round(self._last_entropy, 4),
            }
