from __future__ import annotations

import heapq
import math
import re
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField


class GeometricSemanticReasoner:
    """
    Pure geometric topology semantic reasoning. NO embeddings.
    Uses the BCC lattice structure itself as the semantic model.

    Key insight: In a BCC lattice, memories that are:
    - Face-neighbors share 4 atoms = STRONG semantic relation
    - Edge-neighbors share 2 atoms = MODERATE semantic relation
    - Vertex-neighbors share 1 atom = WEAK semantic relation
    - Same BCC unit = SAME domain

    This gives us a geometric analog to semantic similarity:
    - face_distance(0) = synonymous (same concept cluster)
    - edge_distance(1) = related (shared context)
    - vertex_distance(2) = associated (shared attribute)
    - cross_domain(3+) = analogical (structural similarity only)

    Enhanced with:
    - Geometric Analogy Discovery (isomorphic sub-graphs across domains)
    - Spatial Semantic Clustering (kNN + label overlap)
    - Topological Path Reasoning (multi-hop weighted A* paths)
    - Cross-Domain Bridge Detection (conceptual bridge nodes)
    - Concept Drift Detection (temporal centroid tracking)
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._concept_regions: Dict[str, np.ndarray] = {}
        self._distance_cache: Dict[Tuple[str, str], float] = {}
        self._pattern_index: Dict[str, Set[str]] = defaultdict(set)
        self._node_labels: Dict[str, List[str]] = {}
        self._node_positions: Dict[str, np.ndarray] = {}

        self._concept_region_history: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        self._placement_timestamps: Dict[str, float] = {}
        self._drift_events: List[Dict[str, Any]] = []
        self._bridge_cache: Dict[str, float] = {}
        self._bridge_cache_time: float = 0.0
        self._cluster_cache: List[Dict[str, Any]] = []
        self._cluster_cache_time: float = 0.0

    # ------------------------------------------------------------------
    # Original methods (preserved exactly)
    # ------------------------------------------------------------------

    def index_node(self, node_id: str, content: str, labels: List[str],
                   position: np.ndarray):
        patterns = self._extract_structural_patterns(content)
        for p in patterns:
            self._pattern_index[p].add(node_id)

        self._node_labels[node_id] = labels[:]
        self._node_positions[node_id] = position.copy()

        primary = labels[0] if labels else "__default__"
        if primary not in self._concept_regions:
            self._concept_regions[primary] = position.copy()
        else:
            alpha = 0.1
            self._concept_regions[primary] = (
                (1 - alpha) * self._concept_regions[primary] + alpha * position
            )

        stale = [k for k, v in self._distance_cache.items()
                 if node_id in k]
        for k in stale:
            del self._distance_cache[k]

        self.track_concept_drift(node_id, position)

    def remove_node(self, node_id: str):
        for p, ids in list(self._pattern_index.items()):
            ids.discard(node_id)
            if not ids:
                del self._pattern_index[p]
        self._node_labels.pop(node_id, None)
        self._node_positions.pop(node_id, None)
        stale = [k for k in self._distance_cache.items() if node_id in k]
        for k in stale:
            del self._distance_cache[k]
        self._bridge_cache.pop(node_id, None)
        self._placement_timestamps.pop(node_id, None)

    def compute_semantic_distance(self, node_a: str, node_b: str) -> float:
        cache_key = (min(node_a, node_b), max(node_a, node_b))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        field = self._field
        na = field._nodes.get(node_a)
        nb = field._nodes.get(node_b)
        if na is None or nb is None:
            return 1.0

        topo_dist = self._topological_distance(node_a, node_b)
        topo_score = min(1.0, topo_dist / 8.0)

        shared_ratio = self._shared_neighbor_ratio(node_a, node_b)

        labels_a = set(self._node_labels.get(node_a, []))
        labels_b = set(self._node_labels.get(node_b, []))
        labels_a = {l for l in labels_a if not l.startswith("__")}
        labels_b = {l for l in labels_b if not l.startswith("__")}
        if labels_a or labels_b:
            jaccard = len(labels_a & labels_b) / max(len(labels_a | labels_b), 1)
        else:
            jaccard = 0.0

        patterns_a = set()
        patterns_b = set()
        for p, ids in self._pattern_index.items():
            if node_a in ids:
                patterns_a.add(p)
            if node_b in ids:
                patterns_b.add(p)
        if patterns_a or patterns_b:
            pat_overlap = len(patterns_a & patterns_b) / max(len(patterns_a | patterns_b), 1)
        else:
            pat_overlap = 0.0

        crystal_conn = 0.0
        if na.crystal_channels and node_b in na.crystal_channels:
            crystal_conn = min(1.0, na.crystal_channels[node_b] / 5.0)
        elif nb.crystal_channels and node_a in nb.crystal_channels:
            crystal_conn = min(1.0, nb.crystal_channels[node_a] / 5.0)

        distance = (
            0.40 * topo_score
            + 0.20 * (1.0 - shared_ratio)
            + 0.15 * (1.0 - jaccard)
            + 0.15 * (1.0 - pat_overlap)
            + 0.10 * (1.0 - crystal_conn)
        )
        distance = max(0.0, min(1.0, distance))
        self._distance_cache[cache_key] = distance
        return distance

    def find_analogical_pairs(self, k: int = 5) -> List[Tuple[str, str, float]]:
        pattern_groups = {p: ids for p, ids in self._pattern_index.items()
                          if len(ids) >= 2}
        if not pattern_groups:
            return []

        candidates: List[Tuple[str, str, float]] = []
        field = self._field
        seen_pairs: Set[Tuple[str, str]] = set()

        for pattern, node_ids in pattern_groups.items():
            ids_list = list(node_ids)
            for i in range(len(ids_list)):
                for j in range(i + 1, min(i + 5, len(ids_list))):
                    a, b = ids_list[i], ids_list[j]
                    pair_key = (min(a, b), max(a, b))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    na = field._nodes.get(a)
                    nb = field._nodes.get(b)
                    if na is None or nb is None:
                        continue
                    if not na.is_occupied or not nb.is_occupied:
                        continue

                    labels_a = {l for l in self._node_labels.get(a, []) if not l.startswith("__")}
                    labels_b = {l for l in self._node_labels.get(b, []) if not l.startswith("__")}
                    label_overlap = len(labels_a & labels_b)
                    if label_overlap > 0:
                        continue

                    spatial_dist = float(np.linalg.norm(na.position - nb.position))
                    if spatial_dist < field._spacing * 4:
                        continue

                    analogy_score = min(1.0, spatial_dist / (field._spacing * 10))
                    candidates.append((a, b, analogy_score))

        candidates.sort(key=lambda x: -x[2])
        return candidates[:k]

    def expand_concept(self, node_id: str, depth: int = 2) -> Dict[str, float]:
        field = self._field
        result: Dict[str, float] = {}
        visited: Set[str] = {node_id}

        frontier = [(node_id, 1.0, "face")]
        for _ in range(depth):
            next_frontier = []
            for nid, strength, _ in frontier:
                node = field._nodes.get(nid)
                if node is None:
                    continue

                for fnid in node.face_neighbors[:8]:
                    if fnid in visited:
                        continue
                    visited.add(fnid)
                    decay = strength * 0.8
                    fn = field._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        result[fnid] = result.get(fnid, 0.0) + decay
                    if decay > 0.05:
                        next_frontier.append((fnid, decay, "face"))

                for enid in node.edge_neighbors[:6]:
                    if enid in visited:
                        continue
                    visited.add(enid)
                    decay = strength * 0.5
                    en = field._nodes.get(enid)
                    if en and en.is_occupied:
                        result[enid] = result.get(enid, 0.0) + decay
                    if decay > 0.05:
                        next_frontier.append((enid, decay, "edge"))

                for vnid in node.vertex_neighbors[:4]:
                    if vnid in visited:
                        continue
                    visited.add(vnid)
                    decay = strength * 0.2
                    vn = field._nodes.get(vnid)
                    if vn and vn.is_occupied:
                        result[vnid] = result.get(vnid, 0.0) + decay
                    if decay > 0.05:
                        next_frontier.append((vnid, decay, "vertex"))

            frontier = next_frontier

        return {nid: min(1.0, s) for nid, s in sorted(result.items(), key=lambda x: -x[1])}

    def _topological_distance(self, node_a: str, node_b: str) -> int:
        if node_a == node_b:
            return 0
        field = self._field
        visited = {node_a}
        queue = deque([(node_a, 0)])
        max_hops = 12
        while queue:
            nid, hops = queue.popleft()
            if hops >= max_hops:
                continue
            node = field._nodes.get(nid)
            if node is None:
                continue
            neighbors = list(node.face_neighbors[:8]) + list(node.edge_neighbors[:6])
            if hops < max_hops - 2:
                neighbors.extend(node.vertex_neighbors[:4])
            for nnid in neighbors:
                if nnid == node_b:
                    return hops + 1
                if nnid not in visited:
                    visited.add(nnid)
                    queue.append((nnid, hops + 1))
        return max_hops + 1

    def _shared_neighbor_ratio(self, node_a: str, node_b: str) -> float:
        field = self._field
        na = field._nodes.get(node_a)
        nb = field._nodes.get(node_b)
        if na is None or nb is None:
            return 0.0
        neighbors_a = set(na.face_neighbors + na.edge_neighbors)
        neighbors_b = set(nb.face_neighbors + nb.edge_neighbors)
        if not neighbors_a or not neighbors_b:
            return 0.0
        shared = neighbors_a & neighbors_b
        return len(shared) / max(len(neighbors_a | neighbors_b), 1)

    def _extract_structural_patterns(self, content: str) -> List[str]:
        if not content:
            return []
        patterns = []

        length_bucket = "len_s" if len(content) < 50 else "len_m" if len(content) < 200 else "len_l"
        patterns.append(length_bucket)

        punct_chars = set(re.findall(r'[。，！？；：、\.\!\?\;\:\,\-\(\)\[\]【】]', content))
        if punct_chars:
            patterns.append(f"punct_{len(punct_chars)}")

        cn_matches = re.findall(r'[\u4e00-\u9fff]+', content)
        en_matches = re.findall(r'[a-zA-Z]{2,}', content)
        total_chars = sum(len(m) for m in cn_matches) + sum(len(m) for m in en_matches)
        if total_chars > 0:
            cn_ratio = sum(len(m) for m in cn_matches) / total_chars
            if cn_ratio > 0.7:
                patterns.append("lang_cn")
            elif cn_ratio < 0.3:
                patterns.append("lang_en")
            else:
                patterns.append("lang_mix")

        numbers = re.findall(r'\d+\.?\d*', content)
        if numbers:
            patterns.append("has_numbers")

        if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', content):
            patterns.append("has_date")

        sentences = re.split(r'[。！？\.!\?\n]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            first_words = sentences[0][:10] if sentences else ""
            patterns.append(f"first_{hash(first_words) % 100}")

        if any(c in content for c in ['①②③④⑤', '1.', '2.', '- ', '* ']):
            patterns.append("has_list")

        if '【' in content and '】' in content:
            patterns.append("has_headers")

        return patterns

    def get_stats(self) -> Dict[str, Any]:
        drift_summary = self.get_drift_report()
        return {
            "concept_regions": len(self._concept_regions),
            "pattern_types": len(self._pattern_index),
            "distance_cache_size": len(self._distance_cache),
            "indexed_nodes": len(self._node_labels),
            "drift_events_total": drift_summary["total_events"],
            "tracked_concepts": drift_summary["tracked_concepts"],
            "bridge_cache_size": len(self._bridge_cache),
        }

    # ------------------------------------------------------------------
    # 1. Geometric Analogy Discovery
    # ------------------------------------------------------------------

    def discover_geometric_analogies(
        self,
        max_subgraph_size: int = 4,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find structurally isomorphic sub-graphs across different label
        domains using BCC lattice topology.

        Two sub-graphs are analogical if they share the same topology
        signature (degree sequence + edge-type distribution) while their
        nodes belong to *different* primary label domains.

        Returns up to *k* analogy dicts sorted by isomorphism score.
        """
        field = self._field
        domain_nodes: Dict[str, List[str]] = defaultdict(list)
        for nid, labels in self._node_labels.items():
            node = field._nodes.get(nid)
            if node and node.is_occupied and labels:
                domain_nodes[labels[0]].append(nid)

        domains = [d for d, nodes in domain_nodes.items() if len(nodes) >= 2]
        if len(domains) < 2:
            return []

        sig_map: Dict[str, Tuple[str, str]] = {}
        for domain in domains:
            for nid in domain_nodes[domain]:
                sig = self._compute_subgraph_signature(nid, max_subgraph_size)
                if sig is not None:
                    sig_map[nid] = (domain, sig)

        topology_groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for nid, (domain, topo_hash) in sig_map.items():
            topology_groups[topo_hash].append((nid, domain))

        analogies: List[Dict[str, Any]] = []
        for topo_hash, entries in topology_groups.items():
            if len(entries) < 2:
                continue
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    nid_a, domain_a = entries[i]
                    nid_b, domain_b = entries[j]
                    if domain_a == domain_b:
                        continue

                    iso_score = self._compute_isomorphism_score(
                        nid_a, nid_b, max_subgraph_size
                    )
                    if iso_score < 0.5:
                        continue

                    pos_a = self._node_positions.get(nid_a, np.zeros(3))
                    pos_b = self._node_positions.get(nid_b, np.zeros(3))
                    spatial_dist = float(np.linalg.norm(pos_a - pos_b))

                    labels_a = set(self._node_labels.get(nid_a, []))
                    labels_b = set(self._node_labels.get(nid_b, []))
                    label_diversity = len(labels_a.symmetric_difference(labels_b))

                    analogies.append({
                        "node_a": nid_a,
                        "node_b": nid_b,
                        "domain_a": domain_a,
                        "domain_b": domain_b,
                        "isomorphism_score": round(iso_score, 4),
                        "spatial_distance": round(spatial_dist, 4),
                        "topology_hash": topo_hash,
                        "label_diversity": label_diversity,
                    })

        analogies.sort(key=lambda x: -x["isomorphism_score"])
        return analogies[:k]

    def _compute_subgraph_signature(
        self, seed_id: str, size: int
    ) -> Optional[str]:
        """
        Return a canonical topology hash for the subgraph rooted at
        *seed_id*.  Uses BFS; collects at most *size* occupied nodes and
        encodes the sorted edge-type sequence (F/E/V) as a string key.
        """
        field = self._field
        seed = field._nodes.get(seed_id)
        if seed is None or not seed.is_occupied:
            return None

        visited: Set[str] = {seed_id}
        queue: deque = deque([seed_id])
        edge_sequence: List[str] = []
        count = 0

        while queue and count < size:
            nid = queue.popleft()
            count += 1
            node = field._nodes.get(nid)
            if node is None:
                continue

            candidates: List[Tuple[str, str]] = []
            for fnid in node.face_neighbors[:8]:
                fn = field._nodes.get(fnid)
                if fn and fn.is_occupied and fnid not in visited:
                    candidates.append((fnid, "F"))
            for enid in node.edge_neighbors[:6]:
                en = field._nodes.get(enid)
                if en and en.is_occupied and enid not in visited:
                    candidates.append((enid, "E"))
            for vnid in node.vertex_neighbors[:4]:
                vn = field._nodes.get(vnid)
                if vn and vn.is_occupied and vnid not in visited:
                    candidates.append((vnid, "V"))

            candidates.sort(key=lambda x: {"F": 0, "E": 1, "V": 2}[x[1]])
            for nnid, ntype in candidates:
                if nnid not in visited and count + len(edge_sequence) < size * 3:
                    visited.add(nnid)
                    edge_sequence.append(ntype)
                    queue.append(nnid)

        if len(edge_sequence) < 1:
            return None

        sorted_edges = sorted(edge_sequence)
        degree_hist: Dict[int, int] = defaultdict(int)
        for nid_inner in visited:
            n = field._nodes.get(nid_inner)
            if n is None:
                continue
            d = 0
            for fnid in n.face_neighbors[:8]:
                if fnid in visited:
                    d += 1
            for enid in n.edge_neighbors[:6]:
                if enid in visited:
                    d += 1
            for vnid in n.vertex_neighbors[:4]:
                if vnid in visited:
                    d += 1
            degree_hist[d] += 1

        topo_hash = (
            "".join(sorted_edges)
            + "|"
            + ",".join(f"{d}:{c}" for d, c in sorted(degree_hist.items()))
        )
        return topo_hash

    def _compute_isomorphism_score(
        self, seed_a: str, seed_b: str, size: int
    ) -> float:
        """Compare two sub-graphs via size, degree histogram, and edge-type distribution."""
        sub_a = self._collect_subgraph(seed_a, size)
        sub_b = self._collect_subgraph(seed_b, size)
        if not sub_a or not sub_b:
            return 0.0

        max_len = max(len(sub_a), len(sub_b), 1)
        size_score = 1.0 - abs(len(sub_a) - len(sub_b)) / max_len

        deg_a = self._degree_distribution(sub_a)
        deg_b = self._degree_distribution(sub_b)
        degree_sim = self._histogram_similarity(deg_a, deg_b)

        edge_a = self._edge_type_distribution(sub_a)
        edge_b = self._edge_type_distribution(sub_b)
        edge_sim = self._histogram_similarity(edge_a, edge_b)

        return 0.3 * size_score + 0.35 * degree_sim + 0.35 * edge_sim

    def _collect_subgraph(self, seed_id: str, size: int) -> Set[str]:
        field = self._field
        visited: Set[str] = {seed_id}
        queue: deque = deque([seed_id])
        while queue and len(visited) < size:
            nid = queue.popleft()
            node = field._nodes.get(nid)
            if node is None:
                continue
            for fnid in node.face_neighbors[:8]:
                fn = field._nodes.get(fnid)
                if fn and fn.is_occupied and fnid not in visited and len(visited) < size:
                    visited.add(fnid)
                    queue.append(fnid)
            for enid in node.edge_neighbors[:6]:
                en = field._nodes.get(enid)
                if en and en.is_occupied and enid not in visited and len(visited) < size:
                    visited.add(enid)
                    queue.append(enid)
            for vnid in node.vertex_neighbors[:4]:
                vn = field._nodes.get(vnid)
                if vn and vn.is_occupied and vnid not in visited and len(visited) < size:
                    visited.add(vnid)
                    queue.append(vnid)
        return visited

    def _degree_distribution(self, node_ids: Set[str]) -> Dict[int, int]:
        field = self._field
        dist: Dict[int, int] = defaultdict(int)
        for nid in node_ids:
            node = field._nodes.get(nid)
            if node is None:
                continue
            degree = 0
            for fnid in node.face_neighbors[:8]:
                if fnid in node_ids:
                    degree += 1
            for enid in node.edge_neighbors[:6]:
                if enid in node_ids:
                    degree += 1
            for vnid in node.vertex_neighbors[:4]:
                if vnid in node_ids:
                    degree += 1
            dist[degree] += 1
        return dist

    def _edge_type_distribution(self, node_ids: Set[str]) -> Dict[str, int]:
        field = self._field
        dist: Dict[str, int] = {"F": 0, "E": 0, "V": 0}
        counted: Set[Tuple[str, str]] = set()
        for nid in node_ids:
            node = field._nodes.get(nid)
            if node is None:
                continue
            for fnid in node.face_neighbors[:8]:
                if fnid in node_ids:
                    key = (min(nid, fnid), max(nid, fnid))
                    if key not in counted:
                        counted.add(key)
                        dist["F"] += 1
            for enid in node.edge_neighbors[:6]:
                if enid in node_ids:
                    key = (min(nid, enid), max(nid, enid))
                    if key not in counted:
                        counted.add(key)
                        dist["E"] += 1
            for vnid in node.vertex_neighbors[:4]:
                if vnid in node_ids:
                    key = (min(nid, vnid), max(nid, vnid))
                    if key not in counted:
                        counted.add(key)
                        dist["V"] += 1
        return dist

    @staticmethod
    def _histogram_similarity(hist_a: Dict, hist_b: Dict) -> float:
        all_keys = set(hist_a.keys()) | set(hist_b.keys())
        if not all_keys:
            return 0.0
        total_a = sum(hist_a.values()) or 1
        total_b = sum(hist_b.values()) or 1
        diff = 0.0
        for key in all_keys:
            fa = hist_a.get(key, 0) / total_a
            fb = hist_b.get(key, 0) / total_b
            diff += abs(fa - fb)
        return max(0.0, 1.0 - diff / 2.0)

    # ------------------------------------------------------------------
    # 2. Spatial Semantic Clustering
    # ------------------------------------------------------------------

    def discover_spatial_clusters(
        self,
        k_neighbors: int = 6,
        label_threshold: float = 0.3,
        min_cluster_size: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Discover emergent concept clusters using k-nearest-neighbor in 3D
        BCC-lattice space combined with label overlap (Jaccard).

        Returns clusters sorted by geometric cohesion score.
        """
        field = self._field
        occupied = [
            nid
            for nid in self._node_labels
            if field._nodes.get(nid) and field._nodes.get(nid).is_occupied
        ]
        if len(occupied) < min_cluster_size:
            return []

        positions = np.array(
            [self._node_positions.get(nid, np.zeros(3)) for nid in occupied]
        )
        nid_to_idx = {nid: i for i, nid in enumerate(occupied)}

        clusters: List[Dict[str, Any]] = []
        visited: Set[str] = set()

        for i, nid in enumerate(occupied):
            if nid in visited:
                continue

            pos_i = positions[i]
            diffs = positions - pos_i
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            sorted_indices = np.argsort(dists)
            knn_indices = sorted_indices[1 : k_neighbors + 1]
            knn_ids = [
                occupied[idx]
                for idx in knn_indices
                if idx < len(occupied) and occupied[idx] not in visited
            ]

            if not knn_ids:
                continue

            labels_i = set(self._node_labels.get(nid, []))
            if not labels_i:
                continue

            similar_neighbors: List[str] = []
            for neighbor_id in knn_ids:
                labels_n = set(self._node_labels.get(neighbor_id, []))
                if not labels_n:
                    continue
                overlap = len(labels_i & labels_n) / max(len(labels_i | labels_n), 1)
                if overlap >= label_threshold:
                    similar_neighbors.append(neighbor_id)

            if len(similar_neighbors) < min_cluster_size - 1:
                continue

            cluster_ids = {nid} | set(similar_neighbors)
            overlap_existing = cluster_ids & visited
            if len(overlap_existing) > len(cluster_ids) * 0.5:
                continue

            cluster_positions = np.array(
                [self._node_positions.get(cid, np.zeros(3)) for cid in cluster_ids]
            )
            centroid = np.mean(cluster_positions, axis=0)
            spread = float(np.mean(np.linalg.norm(cluster_positions - centroid, axis=1)))

            all_labels: Dict[str, int] = defaultdict(int)
            for cid in cluster_ids:
                for lbl in self._node_labels.get(cid, []):
                    if not lbl.startswith("__"):
                        all_labels[lbl] += 1

            dominant_labels = sorted(all_labels.items(), key=lambda x: -x[1])[:5]
            cohesion = self._compute_cluster_cohesion(cluster_ids)

            face_count = 0
            edge_count = 0
            vertex_count = 0
            for cid in cluster_ids:
                node = field._nodes.get(cid)
                if node is None:
                    continue
                face_count += len(set(node.face_neighbors[:8]) & cluster_ids)
                edge_count += len(set(node.edge_neighbors[:6]) & cluster_ids)
                vertex_count += len(set(node.vertex_neighbors[:4]) & cluster_ids)

            clusters.append({
                "center_node": nid,
                "centroid": centroid.tolist(),
                "node_ids": list(cluster_ids),
                "size": len(cluster_ids),
                "spatial_spread": round(spread, 4),
                "dominant_labels": [(lbl, cnt) for lbl, cnt in dominant_labels],
                "cohesion_score": round(cohesion, 4),
                "internal_face_edges": face_count // 2,
                "internal_edge_edges": edge_count // 2,
                "internal_vertex_edges": vertex_count // 2,
            })
            visited.update(cluster_ids)

        clusters.sort(key=lambda x: -x["cohesion_score"])
        return clusters

    def _compute_cluster_cohesion(self, node_ids: Set[str]) -> float:
        """
        Geometric cohesion: ratio of internal BCC edges to total edges.
        Face edges weight 1.0, edge-edges 0.5, vertex-edges 0.2.
        """
        field = self._field
        internal_weight = 0.0
        total_weight = 0.0

        for nid in node_ids:
            node = field._nodes.get(nid)
            if node is None:
                continue

            face_set = set(node.face_neighbors[:8])
            edge_set = set(node.edge_neighbors[:6])
            vertex_set = set(node.vertex_neighbors[:4])

            internal_weight += len(face_set & node_ids) * 1.0
            internal_weight += len(edge_set & node_ids) * 0.5
            internal_weight += len(vertex_set & node_ids) * 0.2

            total_weight += len(face_set) * 1.0
            total_weight += len(edge_set) * 0.5
            total_weight += len(vertex_set) * 0.2

        if total_weight == 0:
            return 0.0
        return internal_weight / total_weight

    # ------------------------------------------------------------------
    # 3. Topological Path Reasoning
    # ------------------------------------------------------------------

    def find_reasoning_path(
        self,
        node_a: str,
        node_b: str,
        max_hops: int = 10,
        path_diversity: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find multi-hop reasoning paths between two memories through the
        BCC lattice using weighted A* search.

        Edge weights: face=1, edge=2, vertex=3 (prefer strong connections).
        Returns up to *path_diversity* diverse paths with quality metrics.
        """
        field = self._field
        na = field._nodes.get(node_a)
        nb = field._nodes.get(node_b)
        if na is None or nb is None:
            return []

        paths: List[Dict[str, Any]] = []

        best_path = self._weighted_astar(node_a, node_b, max_hops)
        if best_path:
            paths.append(self._score_path(best_path))

        if path_diversity > 1:
            penalized_edges: Set[Tuple[str, str]] = set()
            if best_path:
                for i in range(len(best_path) - 1):
                    penalized_edges.add(
                        (min(best_path[i], best_path[i + 1]),
                         max(best_path[i], best_path[i + 1]))
                    )

            for _ in range(path_diversity - 1):
                alt_path = self._weighted_astar(
                    node_a, node_b, max_hops,
                    penalized_edges=penalized_edges,
                    penalty=2.0,
                )
                if alt_path and alt_path != best_path:
                    paths.append(self._score_path(alt_path))
                    for i in range(len(alt_path) - 1):
                        penalized_edges.add(
                            (min(alt_path[i], alt_path[i + 1]),
                             max(alt_path[i], alt_path[i + 1]))
                        )
                else:
                    break

        return sorted(paths, key=lambda x: -x["quality_score"])

    def _weighted_astar(
        self,
        start: str,
        goal: str,
        max_hops: int,
        penalized_edges: Optional[Set[Tuple[str, str]]] = None,
        penalty: float = 1.0,
    ) -> Optional[List[str]]:
        """A* through BCC lattice with face=1, edge=2, vertex=3 costs."""
        field = self._field
        goal_pos = self._node_positions.get(goal)
        if goal_pos is None:
            return None
        start_pos = self._node_positions.get(start)
        if start_pos is None:
            return None

        open_set: List[Tuple[float, float, int, str, List[str]]] = []
        counter = 0
        heapq.heappush(open_set, (0.0, 0.0, counter, start, [start]))
        visited: Dict[str, float] = {start: 0.0}

        while open_set:
            f_score, g_score, _, current, path = heapq.heappop(open_set)

            if current == goal:
                return path

            if len(path) > max_hops:
                continue

            node = field._nodes.get(current)
            if node is None:
                continue

            neighbors: List[Tuple[str, float, str]] = []
            for fnid in node.face_neighbors[:8]:
                neighbors.append((fnid, 1.0, "face"))
            for enid in node.edge_neighbors[:6]:
                neighbors.append((enid, 2.0, "edge"))
            for vnid in node.vertex_neighbors[:4]:
                neighbors.append((vnid, 3.0, "vertex"))

            for nnid, edge_cost, edge_type in neighbors:
                nn = field._nodes.get(nnid)
                if nn is None:
                    continue

                extra = 0.0
                if penalized_edges:
                    edge_key = (min(current, nnid), max(current, nnid))
                    if edge_key in penalized_edges:
                        extra = penalty

                new_g = g_score + edge_cost + extra
                if nnid in visited and visited[nnid] <= new_g:
                    continue
                visited[nnid] = new_g

                nn_pos = self._node_positions.get(nnid)
                if nn_pos is None:
                    h = 0.0
                else:
                    h = float(np.linalg.norm(nn_pos - goal_pos))
                f = new_g + h

                counter += 1
                heapq.heappush(open_set, (f, new_g, counter, nnid, path + [nnid]))

        return None

    def _score_path(self, path: List[str]) -> Dict[str, Any]:
        """Compute multi-dimensional quality score for a reasoning path."""
        field = self._field
        edge_types: List[str] = []
        edge_scores: List[float] = []

        for i in range(len(path) - 1):
            nid_a, nid_b = path[i], path[i + 1]
            na = field._nodes.get(nid_a)
            if na is None:
                edge_types.append("unknown")
                edge_scores.append(0.0)
                continue

            if nid_b in na.face_neighbors[:8]:
                edge_types.append("face")
                edge_scores.append(1.0)
            elif nid_b in na.edge_neighbors[:6]:
                edge_types.append("edge")
                edge_scores.append(0.6)
            elif nid_b in na.vertex_neighbors[:4]:
                edge_types.append("vertex")
                edge_scores.append(0.3)
            else:
                edge_types.append("none")
                edge_scores.append(0.0)

        avg_edge = sum(edge_scores) / max(len(edge_scores), 1)
        length_penalty = 1.0 / (1.0 + len(path) * 0.15)

        label_coherence = 0.0
        if len(path) >= 2:
            all_labels = [set(self._node_labels.get(n, [])) for n in path]
            coherent_pairs = 0
            total_pairs = 0
            for i in range(len(all_labels) - 1):
                if all_labels[i] and all_labels[i + 1]:
                    union = len(all_labels[i] | all_labels[i + 1])
                    if union > 0:
                        coherent_pairs += len(all_labels[i] & all_labels[i + 1]) / union
                    total_pairs += 1
            if total_pairs > 0:
                label_coherence = coherent_pairs / total_pairs

        domain_transitions = 0
        for i in range(len(path) - 1):
            la = self._node_labels.get(path[i], [])
            lb = self._node_labels.get(path[i + 1], [])
            pa = la[0] if la else None
            pb = lb[0] if lb else None
            if pa and pb and pa != pb:
                domain_transitions += 1

        quality = 0.4 * avg_edge + 0.3 * length_penalty + 0.3 * label_coherence

        path_labels = []
        for nid in path:
            labels = self._node_labels.get(nid, [])
            if labels:
                path_labels.append(labels[0])

        return {
            "path": path,
            "length": len(path),
            "edge_types": edge_types,
            "avg_edge_quality": round(avg_edge, 4),
            "label_coherence": round(label_coherence, 4),
            "domain_transitions": domain_transitions,
            "quality_score": round(quality, 4),
            "dominant_labels": path_labels,
        }

    # ------------------------------------------------------------------
    # 4. Cross-Domain Bridge Detection
    # ------------------------------------------------------------------

    def detect_bridge_nodes(
        self,
        min_bridge_score: float = 0.4,
        top_k: int = 10,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Identify nodes that serve as conceptual bridges between different
        label domains.

        A bridge node's score combines:
        - **neighbor domain entropy** (Shannon entropy normalised by max)
        - **foreign-neighbour ratio** (proportion of neighbours in a
          different domain than the node itself)
        - **lattice coverage** (how many of the 18 BCC neighbour slots are
          occupied)

        Optionally caches results for 30 s.
        """
        now = time.time()
        if use_cache and self._bridge_cache and (now - self._bridge_cache_time) < 30.0:
            cached = sorted(self._bridge_cache.items(), key=lambda x: -x[1])
            results = []
            for nid, score in cached[:top_k]:
                if score < min_bridge_score:
                    break
                node = self._field._nodes.get(nid)
                if node and node.is_occupied:
                    results.append(self._build_bridge_record(nid, score))
            return results

        field = self._field
        occupied = [
            nid
            for nid in self._node_labels
            if field._nodes.get(nid) and field._nodes.get(nid).is_occupied
        ]
        if len(occupied) < 3:
            return []

        bridges: List[Dict[str, Any]] = []
        new_cache: Dict[str, float] = {}

        for nid in occupied:
            node = field._nodes.get(nid)
            if node is None:
                continue

            own_labels = set(self._node_labels.get(nid, []))
            own_primary = next(
                (l for l in own_labels if not l.startswith("__")), None
            )

            neighbor_domains: Dict[str, int] = defaultdict(int)
            occupied_neighbors = 0

            all_nn = (
                list(node.face_neighbors[:8])
                + list(node.edge_neighbors[:6])
                + list(node.vertex_neighbors[:4])
            )
            for nnid in all_nn:
                nn = field._nodes.get(nnid)
                if nn is None or not nn.is_occupied:
                    continue
                occupied_neighbors += 1
                nn_labels = self._node_labels.get(nnid, [])
                nn_primary = "__none__"
                for l in nn_labels:
                    if not l.startswith("__"):
                        nn_primary = l
                        break
                neighbor_domains[nn_primary] += 1

            if not neighbor_domains:
                continue

            unique_domains = len(neighbor_domains)
            if unique_domains < 2:
                continue

            total_n = sum(neighbor_domains.values())
            entropy = 0.0
            for count in neighbor_domains.values():
                p = count / total_n
                if p > 0:
                    entropy -= p * math.log2(p)

            max_entropy = math.log2(unique_domains) if unique_domains > 1 else 1.0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            foreign_ratio = 0.0
            if total_n > 0:
                same_count = neighbor_domains.get(own_primary or "__none__", 0)
                foreign_ratio = (total_n - same_count) / total_n

            coverage = min(1.0, occupied_neighbors / 18.0)

            bridge_score = (
                norm_entropy * 0.4 + foreign_ratio * 0.35 + coverage * 0.25
            )

            new_cache[nid] = bridge_score

            if bridge_score >= min_bridge_score:
                bridges.append(self._build_bridge_record(nid, bridge_score, neighbor_domains, own_labels, unique_domains, entropy, foreign_ratio))

        self._bridge_cache = new_cache
        self._bridge_cache_time = now

        bridges.sort(key=lambda x: -x["bridge_score"])
        return bridges[:top_k]

    def _build_bridge_record(
        self,
        nid: str,
        bridge_score: float,
        neighbor_domains: Optional[Dict[str, int]] = None,
        own_labels: Optional[Set[str]] = None,
        unique_domains: Optional[int] = None,
        entropy: Optional[float] = None,
        foreign_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        if neighbor_domains is None or own_labels is None:
            field = self._field
            node = field._nodes.get(nid)
            if node is None:
                return {"node_id": nid, "bridge_score": bridge_score}
            own_labels = set(self._node_labels.get(nid, []))
            neighbor_domains = defaultdict(int)
            for nnid in list(node.face_neighbors[:8]) + list(node.edge_neighbors[:6]) + list(node.vertex_neighbors[:4]):
                nn = field._nodes.get(nnid)
                if nn and nn.is_occupied:
                    nn_labels = self._node_labels.get(nnid, [])
                    nn_primary = "__none__"
                    for l in nn_labels:
                        if not l.startswith("__"):
                            nn_primary = l
                            break
                    neighbor_domains[nn_primary] += 1
            unique_domains = len(neighbor_domains)
            total_n = sum(neighbor_domains.values())
            entropy_val = 0.0
            for count in neighbor_domains.values():
                p = count / max(total_n, 1)
                if p > 0:
                    entropy_val -= p * math.log2(p)
            entropy = entropy_val
            own_primary = next((l for l in own_labels if not l.startswith("__")), None)
            foreign_ratio = (total_n - neighbor_domains.get(own_primary or "__none__", 0)) / max(total_n, 1)

        return {
            "node_id": nid,
            "bridge_score": round(bridge_score, 4),
            "unique_domains": unique_domains or 0,
            "domain_distribution": dict(neighbor_domains) if neighbor_domains else {},
            "neighbor_entropy": round(entropy or 0.0, 4),
            "foreign_neighbor_ratio": round(foreign_ratio or 0.0, 4),
            "own_labels": sorted(own_labels) if own_labels else [],
        }

    # ------------------------------------------------------------------
    # 5. Concept Drift Detection
    # ------------------------------------------------------------------

    def track_concept_drift(
        self, node_id: str, position: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Record a placement event and detect concept drift.

        Drift is measured as the Euclidean displacement between the new
        position and the last recorded position for the same primary label.
        A drift event is emitted when displacement exceeds 3 lattice
        spacings.
        """
        labels = self._node_labels.get(node_id, [])
        if not labels:
            return None

        primary = labels[0]
        now = time.time()
        self._placement_timestamps[node_id] = now

        if primary not in self._concept_region_history:
            self._concept_region_history[primary] = [(now, position.copy())]
            return None

        history = self._concept_region_history[primary]
        old_time, old_pos = history[-1]
        displacement = float(np.linalg.norm(position - old_pos))
        time_delta = max(now - old_time, 0.001)

        drift_rate = displacement / time_delta

        history.append((now, position.copy()))
        if len(history) > 100:
            history[:] = history[-100:]

        current_centroid = self._concept_regions.get(primary, position)
        centroid_shift = float(np.linalg.norm(position - current_centroid))

        drift_event = None
        spacing = self._field._spacing
        if displacement > spacing * 3:
            severity = "high" if displacement > spacing * 6 else "medium"
            drift_event = {
                "concept": primary,
                "node_id": node_id,
                "displacement": round(displacement, 4),
                "centroid_shift": round(centroid_shift, 4),
                "drift_rate": round(drift_rate, 6),
                "time_delta": round(time_delta, 2),
                "history_length": len(history),
                "severity": severity,
            }
            self._drift_events.append(drift_event)
            if len(self._drift_events) > 1000:
                self._drift_events = self._drift_events[-1000:]

        return drift_event

    def get_drift_report(self) -> Dict[str, Any]:
        """Summarise concept drift activity across all tracked labels."""
        report: Dict[str, Any] = {
            "total_events": len(self._drift_events),
            "tracked_concepts": len(self._concept_region_history),
            "recent_events": [],
            "concept_stability": {},
        }

        for event in self._drift_events[-20:]:
            report["recent_events"].append({
                "concept": event["concept"],
                "severity": event["severity"],
                "displacement": event["displacement"],
                "centroid_shift": event["centroid_shift"],
            })

        spacing = self._field._spacing
        for concept, history in self._concept_region_history.items():
            if len(history) < 2:
                report["concept_stability"][concept] = {
                    "placements": len(history),
                    "total_drift": 0.0,
                    "status": "stable",
                }
                continue

            total_drift = 0.0
            for i in range(1, len(history)):
                _, prev_pos = history[i - 1]
                _, curr_pos = history[i]
                total_drift += float(np.linalg.norm(curr_pos - prev_pos))

            avg_drift = total_drift / len(history)
            if avg_drift > spacing * 4:
                status = "drifting"
            elif avg_drift > spacing * 2:
                status = "shifting"
            else:
                status = "stable"

            report["concept_stability"][concept] = {
                "placements": len(history),
                "total_drift": round(total_drift, 4),
                "avg_drift": round(avg_drift, 4),
                "status": status,
            }

        return report

    # ------------------------------------------------------------------
    # Composite helpers
    # ------------------------------------------------------------------

    def full_geometric_analysis(self) -> Dict[str, Any]:
        """Run all five enhanced analyses and return a combined report."""
        analogies = self.discover_geometric_analogies(k=5)
        clusters = self.discover_spatial_clusters()
        bridges = self.detect_bridge_nodes(top_k=10)
        drift = self.get_drift_report()

        return {
            "analogies": analogies,
            "clusters": clusters,
            "bridges": bridges,
            "drift_report": drift,
            "indexed_nodes": len(self._node_labels),
            "concept_regions": len(self._concept_regions),
        }
