"""
TetraMesh — Pure tetrahedral memory mesh for TetraMem-XL.

Memory lives on 3-simplices (tetrahedra), connected by shared
faces, edges, and vertices into a dynamic topological mesh.

All operations use pure geometry primitives. No vector embeddings.
"""

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import gudhi

    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


def text_to_geometry(text: str, labels=None) -> np.ndarray:
    """Deterministic text-to-3D-point mapping using multi-scale hash.

    Layer 1 (SHA-256): Base spatial position — deterministic hash
    Layer 2 (Label geometry): Labels rotate the point toward label-specific attractors
    Layer 3 (Content structure): Sentence length and punctuation shift the position

    This creates a 3D space where:
    - Similar labels cluster memories together
    - Similar content lengths/structures occupy nearby regions
    - SHA-256 ensures uniqueness while preserving topology
    """
    h = hashlib.sha256(text.encode()).digest()
    vals = np.frombuffer(h[:12], dtype=np.uint32).copy().astype(np.float32)
    base_max = float(np.abs(vals).max())
    if base_max < 1e-8:
        base_max = 1.0
    base = vals / base_max * 3.0

    if labels and len(labels) > 0:
        label_h = hashlib.sha256("|".join(sorted(labels)).encode()).digest()
        label_vals = np.frombuffer(label_h[:12], dtype=np.uint32).copy().astype(np.float32)
        label_max = float(np.abs(label_vals).max())
        if label_max < 1e-8:
            label_max = 1.0
        label_point = label_vals / label_max * 3.0

        n_labels = min(len(labels), 5)
        blend = 0.15 + 0.05 * n_labels
        base = base * (1.0 - blend) + label_point * blend

    content_len = len(text)
    n_sentences = text.count("。") + text.count("！") + text.count("？") + text.count(".") + text.count("!") + text.count("?")
    n_sentences = max(n_sentences, 1)

    structure_shift = np.array([
        (content_len % 200) / 200.0 * 0.3 - 0.15,
        (n_sentences % 10) / 10.0 * 0.2 - 0.1,
        0.0,
    ], dtype=np.float32)

    base = base + structure_shift

    result = base[:3]
    if not np.all(np.isfinite(result)):
        result = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return result


class MemoryTetrahedron:
    __slots__ = (
        "id",
        "content",
        "vertex_indices",
        "centroid",
        "labels",
        "metadata",
        "weight",
        "creation_time",
        "last_access_time",
        "init_weight",
        "_spatial_alpha",
        "integration_count",
        "access_count",
        "secondary_memories",
    )

    id: str
    content: str
    vertex_indices: Tuple[int, int, int, int]
    centroid: np.ndarray
    labels: List[str]
    metadata: Dict[str, Any]
    weight: float
    creation_time: float
    last_access_time: float
    init_weight: float
    _spatial_alpha: float
    integration_count: int
    access_count: int

    def __init__(
        self,
        id: str,
        content: str,
        vertex_indices: Tuple[int, int, int, int],
        centroid: np.ndarray,
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
        creation_time: float = 0.0,
        last_access_time: float = 0.0,
        init_weight: float = 1.0,
        _spatial_alpha: float = 0.0,
        integration_count: int = 0,
        access_count: int = 0,
    ):
        self.id = id
        self.content = content
        self.vertex_indices = vertex_indices
        self.centroid = centroid.astype(np.float32) if centroid.dtype != np.float32 else centroid
        self.labels = labels if labels is not None else []
        self.metadata = metadata if metadata is not None else {}
        self.weight = weight
        self.creation_time = creation_time
        self.last_access_time = last_access_time
        self.init_weight = init_weight
        self._spatial_alpha = _spatial_alpha
        self.integration_count = integration_count
        self.access_count = access_count
        self.secondary_memories: List[Dict[str, Any]] = []

    def filtration(self, time_lambda: float = 0.001) -> float:
        age = time.time() - self.creation_time
        integration_bonus = 1.0 / (1.0 + 0.1 * self.integration_count)
        return self._spatial_alpha * integration_bonus + time_lambda * age

    def catalyze_integration(self, strength: float = 1.0) -> None:
        self.integration_count += 1
        self.weight = min(10.0, self.weight + strength * 0.05)

    def touch(self) -> None:
        self.last_access_time = time.time()
        self.access_count += 1

    def attach_secondary(self, content: str, labels: Optional[List[str]] = None,
                         weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> int:
        slot = {
            "content": content,
            "labels": labels or [],
            "weight": weight,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        self.secondary_memories.append(slot)
        return len(self.secondary_memories) - 1

    def integrate_secondary(self) -> int:
        if not self.secondary_memories:
            return 0

        n = len(self.secondary_memories)
        all_contents = [self.content] + [s["content"] for s in self.secondary_memories]

        word_freq = {}
        for content in all_contents:
            words = re.findall(r"\b\w{3,}\b", content.lower())
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1

        theme_words = sorted(word_freq.items(), key=lambda x: -x[1])
        top_themes = [w for w, c in theme_words[:5] if c >= 2]

        if top_themes and n >= 2:
            primary_snippet = self.content[:60]
            secondary_snippets = []
            for s in sorted(self.secondary_memories, key=lambda x: -x["weight"])[:3]:
                secondary_snippets.append(s["content"][:60])

            if len(primary_snippet) > 20 and secondary_snippets:
                theme_str = ", ".join(top_themes[:3])
                self.content = (
                    "[abstract:" + theme_str + "] "
                    + primary_snippet
                    + " + " + str(len(secondary_snippets)) + " related"
                )

        label_freq = {}
        for lbl in self.labels:
            label_freq[lbl] = label_freq.get(lbl, 0) + 2
        for s in self.secondary_memories:
            for lbl in s.get("labels", []):
                label_freq[lbl] = label_freq.get(lbl, 0) + 1

        consolidated_labels = [lbl for lbl, cnt in sorted(label_freq.items(), key=lambda x: -x[1])
                               if lbl not in ("__dream__", "__system__")]
        consolidated_labels = consolidated_labels[:10]
        self.labels = consolidated_labels

        total_w = self.weight
        for s in self.secondary_memories:
            total_w += s.get("weight", 1.0)

        fused = total_w / (1 + n)
        integration_boost = 1.0 + 0.1 * min(n, 5)
        self.weight = min(10.0, fused * integration_boost)

        provenance = {
            "reorg_timestamp": time.time(),
            "sources_count": n,
            "source_contents": [s["content"][:80] for s in self.secondary_memories[:5]],
            "source_labels": [s.get("labels", []) for s in self.secondary_memories[:5]],
            "themes_extracted": top_themes[:5],
        }
        existing = self.metadata.get("reorg_history", [])
        existing.append(provenance)
        self.metadata["reorg_history"] = existing[-3:]

        self.catalyze_integration(1.0)
        self.secondary_memories.clear()
        return n


@dataclass
class FaceRecord:
    vertex_indices: Tuple[int, int, int]
    tetrahedra: Set[str] = field(default_factory=set)

    @property
    def is_boundary(self) -> bool:
        return len(self.tetrahedra) <= 1


class TetraMesh:
    """
    Dynamic tetrahedral mesh where each 3-simplex carries a memory.

    Growth: new tetrahedra attach to boundary faces (outward growth).
    Retrieval: topological navigation via shared faces/edges/vertices.
    Time Law: filtration acts as integration catalyst — memories that
    participate in integration cycles become more accessible, never decay.
    """

    def __init__(self, time_lambda: float = 0.001):
        self._vertices: List[np.ndarray] = []
        self._tetrahedra: Dict[str, MemoryTetrahedron] = {}
        self._faces: Dict[Tuple[int, int, int], FaceRecord] = {}
        self._vertex_to_tetra: Dict[int, Set[str]] = defaultdict(set)
        self._label_index: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._time_lambda = time_lambda
        self._centroid_index_dirty = True
        self._centroid_ids: List[str] = []
        self._centroid_matrix: Optional[np.ndarray] = None
        self._boundary_dirty = True
        self._inserts_since_boundary_rebuild: int = 0
        self._boundary_face_keys: List[Tuple[int, int, int]] = []
        self._boundary_centroids: List[np.ndarray] = []
        self._last_tetra_id: Optional[str] = None
        self._query_cache: Dict[str, Any] = {}
        self._query_cache_max = 100
        self._query_cache_dirty = True
        self._vertex_ref_count: Dict[int, int] = defaultdict(int)
        self._vertex_compact_threshold = 256
        self._hub_node_id: Optional[str] = None
        self._hub_node_score: float = -1.0

    def _content_hash(self, content: str) -> str:
        return hashlib.sha256(content.strip().lower().encode()).hexdigest()[:16]

    def store(
        self,
        content: str,
        seed_point: np.ndarray,
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
        dedup: bool = False,
    ) -> str:
        with self._lock:
            if dedup:
                chash = self._content_hash(content)
                for tid, t in self._tetrahedra.items():
                    if self._content_hash(t.content) == chash:
                        return tid
            tetra_id = hashlib.sha256(
                (content + str(time.time()) + str(len(self._tetrahedra))).encode()
            ).hexdigest()[:16]

            if not self._tetrahedra:
                vindices = self._create_seed_tetrahedron(seed_point)
            else:
                vindices = self._attach_to_boundary(seed_point)

            a, b, c, d = (self._vertices[i] for i in vindices)
            centroid = (a + b + c + d) * 0.25

            spatial_alpha = self._compute_spatial_alpha_fast(a, b, c, d, weight)

            tetra = MemoryTetrahedron(
                id=tetra_id,
                content=content,
                vertex_indices=vindices,
                centroid=centroid,
                labels=labels or [],
                metadata=metadata or {},
                weight=weight,
                creation_time=time.time(),
                last_access_time=time.time(),
                init_weight=weight,
                _spatial_alpha=spatial_alpha,
            )

            self._tetrahedra[tetra_id] = tetra
            self._ref_vertices(vindices)
            if labels:
                for label in labels:
                    self._label_index[label].add(tetra_id)

            for fk in self._faces_of_tetra(vindices):
                if fk not in self._faces:
                    self._faces[fk] = FaceRecord(vertex_indices=fk)
                    self._boundary_face_keys.append(fk)
                    va, vb, vc = (self._vertices[i] for i in fk)
                    self._boundary_centroids.append((va + vb + vc) / 3.0)
                self._faces[fk].tetrahedra.add(tetra_id)

            self._last_tetra_id = tetra_id
            self._centroid_index_dirty = True
            self._inserts_since_boundary_rebuild += 1
            if self._inserts_since_boundary_rebuild >= 50:
                self._boundary_dirty = True
                self._inserts_since_boundary_rebuild = 0
            self._query_cache_dirty = True
            self._hub_node_id = None
            return tetra_id

    def _extract_text_tokens(self, text: str) -> set:
        tokens = set()
        for w in re.findall(r"[a-zA-Z0-9]{2,}", text.lower()):
            tokens.add(w)
        for c in re.findall(r"[\u4e00-\u9fff]", text):
            tokens.add(c)
        for bigram in re.findall(r"[\u4e00-\u9fff]{2}", text):
            tokens.add(bigram)
        return tokens

    def _seed_by_text(self, query_text: str) -> Optional[str]:
        qtokens = self._extract_text_tokens(query_text)
        if not qtokens:
            return None
        best_id = None
        best_score = 0
        for tid, t in self._tetrahedra.items():
            ctokens = self._extract_text_tokens(t.content)
            overlap = len(qtokens & ctokens)
            if overlap > best_score:
                best_score = overlap
                best_id = tid
        return best_id if best_score > 0 else None

    def _seed_by_labels_text(self, labels, query_text):
        if labels:
            for lbl in labels:
                for tid in self._label_index.get(lbl, set()):
                    if tid in self._tetrahedra:
                        return tid
        if query_text:
            return self._seed_by_text(query_text)
        return self.seed_by_structure(labels)

    def _rank_by_text_relevance(self, query_text, k):
        if not query_text:
            return []
        qtokens = self._extract_text_tokens(query_text)
        if not qtokens:
            return []
        scored = []
        for tid, t in self._tetrahedra.items():
            ctokens = self._extract_text_tokens(t.content)
            if not ctokens:
                continue
            overlap = len(qtokens & ctokens)
            if overlap == 0:
                continue
            score = overlap / max(len(qtokens), 1)
            scored.append((tid, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def query_topological(self, query_point: np.ndarray, k: int = 5,
                          labels: Optional[List[str]] = None,
                          query_text: Optional[str] = None) -> List[Tuple[str, float]]:
        """Hybrid query — merges text relevance × topology × weight signals.

        Three scoring channels combined with geometric weights:
          TEXT (alpha):  token overlap ratio (char bigram + word level)
          TOPO  (beta):  connection type + volume + label Jaccard + freshness
          WEIGHT (gamma): raw weight as importance signal

        Final score = alpha * text + beta * topo + gamma * weight
        When text query is absent, topology dominates with gamma assist.
        """
        cache_key = f"{query_point.tobytes().hex()}_{k}_{tuple(labels or [])}_{query_text or ''}"

        with self._lock:
            if not self._query_cache_dirty and cache_key in self._query_cache:
                return self._query_cache[cache_key]

            if not self._tetrahedra:
                return []

            alpha = 0.45
            beta = 0.35
            gamma = 0.20

            text_scores: Dict[str, float] = {}
            if query_text:
                text_scores = self._compute_text_scores(query_text)

            topo_scores: Dict[str, float] = {}
            weight_scores: Dict[str, float] = {}

            max_w = max((t.weight for t in self._tetrahedra.values()), default=1.0)
            if max_w < 0.1:
                max_w = 1.0

            seed_id = self._seed_by_labels_text(labels, query_text)
            has_topo = seed_id is not None

            if has_topo:
                self._tetrahedra[seed_id].touch()
                nav = self.navigate_topology(seed_id, max_steps=k * 3)

                type_penalty = {"seed": 0.1, "face": 0.3, "edge": 0.6, "vertex": 0.85}
                seed_t = self._tetrahedra[seed_id]
                v1 = self._tetra_volume(seed_t)

                for tid, conn_type, hop in nav:
                    tetra = self._tetrahedra.get(tid)
                    if tetra is None:
                        continue

                    penalty = type_penalty.get(conn_type, 0.9) + hop * 0.05

                    v_score = 0.0
                    v2 = self._tetra_volume(tetra)
                    if v1 > 0 and v2 > 0:
                        v_score = 0.2 * min(v1, v2) / max(v1, v2)

                    w_sim = 0.15 * (1.0 - abs(seed_t.weight - tetra.weight) / max(seed_t.weight, tetra.weight, 0.1))

                    l_score = 0.0
                    if seed_t.labels and tetra.labels:
                        l_score = 0.15 * len(set(seed_t.labels) & set(tetra.labels)) / len(set(seed_t.labels) | set(tetra.labels))

                    fil = tetra.filtration(self._time_lambda)
                    t_score = 0.2 * 1.0 / (1.0 + fil * 0.5)
                    w_factor = tetra.weight / (tetra.init_weight + 1e-6)
                    t_score *= w_factor

                    topo_scores[tid] = penalty + v_score + w_sim + l_score + t_score

            all_ids = set(text_scores.keys()) | set(topo_scores.keys())
            if not all_ids:
                all_ids = set(self._tetrahedra.keys())

            results = []
            for tid in all_ids:
                tetra = self._tetrahedra.get(tid)
                if tetra is None:
                    continue

                t_s = text_scores.get(tid, 0.0)
                p_s = topo_scores.get(tid, 0.0)
                w_s = tetra.weight / max_w

                if text_scores:
                    score = alpha * t_s + beta * p_s + gamma * w_s
                elif has_topo:
                    score = beta * p_s + gamma * w_s
                else:
                    score = gamma * w_s

                results.append((tid, score))

            results.sort(key=lambda x: -x[1])
            final = results[:k]

            self._query_cache[cache_key] = final
            if len(self._query_cache) > self._query_cache_max:
                oldest = list(self._query_cache.keys())[: self._query_cache_max // 2]
                for ok in oldest:
                    del self._query_cache[ok]
            self._query_cache_dirty = False

            return final

    def _compute_text_scores(self, query_text: str) -> Dict[str, float]:
        qtokens = self._extract_text_tokens(query_text)
        if not qtokens:
            return {}
        scores = {}
        for tid, t in self._tetrahedra.items():
            ctokens = self._extract_text_tokens(t.content)
            if not ctokens:
                continue
            overlap = len(qtokens & ctokens)
            if overlap == 0:
                continue
            scores[tid] = overlap / max(len(qtokens), 1)
        return scores

    def associate_topological(
        self, tetra_id: str, max_depth: int = 2
    ) -> List[Tuple[str, float, str]]:
        with self._lock:
            if tetra_id not in self._tetrahedra:
                return []

            visited = {tetra_id}
            results = []

            if max_depth >= 1:
                for nid in self._face_neighbors(tetra_id):
                    if nid not in visited and nid in self._tetrahedra:
                        self._tetrahedra[nid].touch()
                        score = self._association_score(tetra_id, nid, "face")
                        results.append((nid, score, "shared_face"))
                        visited.add(nid)

            if max_depth >= 2:
                edge_nbs = self._edge_neighbors(tetra_id) - visited
                for nid in edge_nbs:
                    if nid in self._tetrahedra:
                        self._tetrahedra[nid].touch()
                        score = self._association_score(tetra_id, nid, "edge")
                        results.append((nid, score, "shared_edge"))
                        visited.add(nid)

            for nid in self._vertex_neighbors(tetra_id) - visited:
                if nid in self._tetrahedra:
                    score = self._association_score(tetra_id, nid, "vertex")
                    results.append((nid, score, "shared_vertex"))
                    visited.add(nid)

            results.sort(key=lambda x: x[1], reverse=True)
            return results

    def store_secondary(self, tetra_id: str, content: str,
                         labels: Optional[List[str]] = None,
                         weight: float = 1.0,
                         metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """Attach a secondary memory to an existing tetrahedron.

        The memory lives on the same 3-simplex, increasing its density.
        Call integrate_secondary() later to merge accumulated memories.
        """
        with self._lock:
            tetra = self._tetrahedra.get(tetra_id)
            if tetra is None:
                return None
            slot_idx = tetra.attach_secondary(content, labels, weight, metadata)
            if labels:
                for label in labels:
                    self._label_index[label].add(tetra_id)
            self._query_cache_dirty = True
            return slot_idx

    def integrate_tetra(self, tetra_id: str) -> int:
        """Integrate all secondary memories on a tetrahedron into its primary."""
        with self._lock:
            tetra = self._tetrahedra.get(tetra_id)
            if tetra is None:
                return 0
            count = tetra.integrate_secondary()
            if count > 0:
                self._centroid_index_dirty = True
                self._query_cache_dirty = True
            return count


    def abstract_reorganize(self, min_density: int = 2, max_operations: int = 20,
                            fusion_fn: Optional[Any] = None) -> Dict[str, Any]:
        stats = {
            "integrated_count": 0,
            "cross_fusions": 0,
            "tetra_scanned": 0,
            "themes_created": [],
        }

        with self._lock:
            dense_tetras = []
            for tid, tetra in self._tetrahedra.items():
                n_sec = len(tetra.secondary_memories) if tetra.secondary_memories else 0
                if n_sec >= min_density:
                    dense_tetras.append((tid, n_sec))
                stats["tetra_scanned"] += 1

            dense_tetras.sort(key=lambda x: -x[1])

            ops = 0
            for tid, density in dense_tetras:
                if ops >= max_operations:
                    break
                tetra = self._tetrahedra.get(tid)
                if tetra is None or not tetra.secondary_memories:
                    continue
                count = tetra.integrate_secondary()
                if count > 0:
                    stats["integrated_count"] += 1
                    ops += 1

            if len(dense_tetras) >= 2 and ops < max_operations:
                cross_pairs = []
                for i in range(min(len(dense_tetras), 10)):
                    for j in range(i + 1, min(len(dense_tetras), 10)):
                        t1_id = dense_tetras[i][0]
                        t2_id = dense_tetras[j][0]
                        t1_t = self._tetrahedra.get(t1_id)
                        t2_t = self._tetrahedra.get(t2_id)
                        if t1_t is None or t2_t is None:
                            continue
                        shared = self._face_neighbors(t1_id) & self._face_neighbors(t2_id)
                        edge_overlap = len(
                            set(t1_t.labels) &
                            set(t2_t.labels)
                        )
                        if len(shared) > 0 or edge_overlap >= 2:
                            cross_pairs.append((t1_id, t2_id, len(shared) + edge_overlap))

                cross_pairs.sort(key=lambda x: -x[2])

                for t1_id, t2_id, score in cross_pairs:
                    if ops >= max_operations:
                        break
                    t1 = self._tetrahedra.get(t1_id)
                    t2 = self._tetrahedra.get(t2_id)
                    if t1 is None or t2 is None:
                        continue

                    if fusion_fn is not None:
                        fused_content = fusion_fn(t1.content, t2.content)
                    else:
                        c1 = t1.content[:60]
                        c2 = t2.content[:60]
                        shared_lbl = set(t1.labels) & set(t2.labels)
                        lbl_str = ", ".join(shared_lbl - {"__dream__", "__system__"}) if shared_lbl else "general"
                        fused_content = "[concept:" + lbl_str + "] " + c1 + " intersect " + c2

                    bridge = (t1.centroid + t2.centroid) / 2.0
                    bridge += np.random.normal(0, 0.01, size=3)
                    combined_labels = list(set(t1.labels + t2.labels) - {"__dream__", "__system__"})

                    new_id = self.store(
                        content=fused_content,
                        seed_point=bridge,
                        labels=combined_labels[:8],
                        metadata={
                            "type": "cross_fusion",
                            "sources": [t1_id, t2_id],
                            "fusion_score": score,
                        },
                        weight=min(t1.weight, t2.weight) * 0.8,
                    )
                    stats["cross_fusions"] += 1
                    stats["themes_created"].append(new_id[:8])
                    ops += 1

        self._query_cache_dirty = True
        return stats

    def catalyze_integration_batch(
        self, tetra_ids: List[str], strength: float = 1.0
    ) -> Dict[str, Any]:
        catalyzed = 0
        for tid in tetra_ids:
            tetra = self._tetrahedra.get(tid)
            if tetra is not None:
                tetra.catalyze_integration(strength)
                catalyzed += 1
        return {"catalyzed": catalyzed, "total": len(self._tetrahedra)}

    def compute_persistent_entropy(self) -> float:
        if not GUDHI_AVAILABLE or len(self._tetrahedra) < 4:
            return 0.0
        st = self.compute_ph()
        if st is None:
            return 0.0
        from .persistent_entropy import compute_persistent_entropy

        return compute_persistent_entropy(st)

    def edge_contraction(self, tetra_id_1: str, tetra_id_2: str) -> Optional[str]:
        with self._lock:
            t1 = self._tetrahedra.get(tetra_id_1)
            t2 = self._tetrahedra.get(tetra_id_2)
            if t1 is None or t2 is None:
                return None

            shared = set(t1.vertex_indices) & set(t2.vertex_indices)
            if len(shared) < 2:
                return None

            new_centroid = (t1.centroid + t2.centroid) / 2.0
            new_weight = (t1.weight + t2.weight) / 2.0
            primary, secondary = (t1, t2) if t1.weight >= t2.weight else (t2, t1)
            new_content = primary.content
            new_labels = list(set(t1.labels + t2.labels))
            new_meta = {
                **secondary.metadata,
                **primary.metadata,
                "merged_from": [t1.id, t2.id],
                "merged_content": secondary.content[:500],
                "merged_weight": secondary.weight,
            }

            self._remove_tetrahedron(t1.id)
            self._remove_tetrahedron(t2.id)

            new_vi = self._create_seed_tetrahedron(new_centroid)
            new_id = hashlib.sha256((t1.id + t2.id + str(time.time())).encode()).hexdigest()[:16]

            merged_spatial_alpha = self._compute_spatial_alpha(new_vi, new_weight)

            merged = MemoryTetrahedron(
                id=new_id,
                content=new_content,
                vertex_indices=new_vi,
                centroid=np.mean([self._vertices[i] for i in new_vi], axis=0),
                labels=new_labels,
                metadata=new_meta,
                weight=new_weight,
                creation_time=min(t1.creation_time, t2.creation_time),
                last_access_time=time.time(),
                init_weight=new_weight,
                _spatial_alpha=merged_spatial_alpha,
            )

            self._tetrahedra[new_id] = merged
            self._ref_vertices(new_vi)
            for label in new_labels:
                self._label_index[label].add(new_id)
            for fk in self._faces_of_tetra(new_vi):
                if fk not in self._faces:
                    self._faces[fk] = FaceRecord(vertex_indices=fk)
                self._faces[fk].tetrahedra.add(new_id)
            for vi in new_vi:
                self._vertex_to_tetra[vi].add(new_id)

            return new_id

    def compute_ph(self) -> Optional[Any]:
        if not GUDHI_AVAILABLE or len(self._vertices) < 4:
            return None
        with self._lock:
            used = set()
            for tetra in self._tetrahedra.values():
                used.update(tetra.vertex_indices)
            if len(used) < 4:
                return None

            used_sorted = sorted(used)
            pts = [self._vertices[i] for i in used_sorted]

            vertex_time_weights: Dict[int, float] = {}
            for vi in used_sorted:
                tetra_ids = self._vertex_to_tetra.get(vi, set())
                if not tetra_ids:
                    vertex_time_weights[vi] = 1.0
                    continue
                time_weighted = []
                for tid in tetra_ids:
                    t = self._tetrahedra.get(tid)
                    if t:
                        fil = t.filtration(self._time_lambda)
                        time_weighted.append(t.weight / (1.0 + fil))
                vertex_time_weights[vi] = float(np.mean(time_weighted)) if time_weighted else 1.0

            weights = [vertex_time_weights[vi] for vi in used_sorted]

            ac = gudhi.AlphaComplex(points=pts, weights=weights, precision="fast")
            st = ac.create_simplex_tree()
            st.compute_persistence(homology_coeff_field=2, min_persistence=0.01)
            return st

    def get_tetrahedron(self, tetra_id: str) -> Optional[MemoryTetrahedron]:
        return self._tetrahedra.get(tetra_id)

    @property
    def tetrahedra(self) -> Dict[str, MemoryTetrahedron]:
        return self._tetrahedra

    def snapshot_tetrahedra(self) -> Dict[str, MemoryTetrahedron]:
        with self._lock:
            return dict(self._tetrahedra)

    @property
    def vertices(self) -> List[np.ndarray]:
        return self._vertices

    @property
    def faces(self) -> Dict[Tuple[int, int, int], FaceRecord]:
        return self._faces

    @property
    def boundary_faces(self) -> List[Tuple[int, int, int]]:
        return [fk for fk, f in self._faces.items() if f.is_boundary]

    @property
    def label_index(self) -> Dict[str, Set[str]]:
        return self._label_index

    def browse_timeline(
        self,
        direction: str = "newest",
        limit: int = 20,
        label_filter: Optional[List[str]] = None,
        min_weight: float = 0.0,
        exclude_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Navigate the mesh along the temporal dimension via filtration values.

        Filtration = spatial_alpha * integration_bonus + time_lambda * age,
        so sorting by filtration is navigating the 3D mesh through time.

        Args:
            direction: 'newest' (low filtration = recent) or 'oldest'
            limit: max results
            label_filter: only include tetrahedra with these labels
            min_weight: minimum weight threshold
            exclude_labels: exclude tetrahedra with ANY of these labels
        """
        with self._lock:
            exclude_set = set(exclude_labels or [])
            items = []
            for tid, t in self._tetrahedra.items():
                if exclude_set & set(t.labels):
                    continue
                if t.weight < min_weight:
                    continue
                if label_filter:
                    if not set(label_filter) & set(t.labels):
                        continue
                fil = t.filtration(self._time_lambda)
                items.append({
                    "id": tid,
                    "content": t.content,
                    "labels": list(t.labels),
                    "weight": float(t.weight),
                    "filtration": float(fil),
                    "centroid": t.centroid.tolist() if hasattr(t.centroid, 'tolist') else list(t.centroid),
                    "creation_time": t.creation_time,
                    "access_count": t.access_count,
                    "integration_count": t.integration_count,
                    "metadata": {
                        k: v for k, v in t.metadata.items()
                        if k in ("type", "source", "fusion_quality")
                    },
                })
            reverse = direction == "newest"
            items.sort(key=lambda x: x["creation_time"], reverse=reverse)
            return items[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_tetrahedra": len(self._tetrahedra),
            "total_vertices": len(self._vertices),
            "total_faces": len(self._faces),
            "boundary_faces": len(self.boundary_faces),
            "total_labels": len(self._label_index),
        }
        if self._tetrahedra:
            fil_values = [t.filtration(self._time_lambda) for t in self._tetrahedra.values()]
            weights = [t.weight for t in self._tetrahedra.values()]
            int_counts = [t.integration_count for t in self._tetrahedra.values()]
            access_counts = [t.access_count for t in self._tetrahedra.values()]
            stats["avg_filtration"] = float(np.mean(fil_values))
            stats["avg_weight"] = float(np.mean(weights))
            stats["min_weight"] = float(min(weights))
            stats["max_filtration"] = float(max(fil_values))
            stats["total_integrations"] = sum(int_counts)
            stats["total_accesses"] = sum(access_counts)
            stats["persistent_entropy"] = self.compute_persistent_entropy()
        return stats

    # ── internal ──────────────────────────────────────────────

    def _compute_spatial_alpha(self, vindices: Tuple, weight: float) -> float:
        v0, v1, v2, v3 = (self._vertices[i] for i in vindices)
        e01 = float(np.sum((v0 - v1) ** 2))
        e02 = float(np.sum((v0 - v2) ** 2))
        e03 = float(np.sum((v0 - v3) ** 2))
        e12 = float(np.sum((v1 - v2) ** 2))
        e13 = float(np.sum((v1 - v3) ** 2))
        e23 = float(np.sum((v2 - v3) ** 2))
        avg_edge_sq = (e01 + e02 + e03 + e12 + e13 + e23) / 6.0
        if avg_edge_sq < 1e-12:
            return 1e6 / (weight + 1e-6)
        return avg_edge_sq**0.5 / (weight + 1e-6)

    @staticmethod
    def _compute_spatial_alpha_fast(v0, v1, v2, v3, weight: float) -> float:
        avg_sq = (
            ((v0[0] - v1[0]) ** 2 + (v0[1] - v1[1]) ** 2 + (v0[2] - v1[2]) ** 2)
            + ((v0[0] - v2[0]) ** 2 + (v0[1] - v2[1]) ** 2 + (v0[2] - v2[2]) ** 2)
            + ((v0[0] - v3[0]) ** 2 + (v0[1] - v3[1]) ** 2 + (v0[2] - v3[2]) ** 2)
            + ((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2)
            + ((v1[0] - v3[0]) ** 2 + (v1[1] - v3[1]) ** 2 + (v1[2] - v3[2]) ** 2)
            + ((v2[0] - v3[0]) ** 2 + (v2[1] - v3[1]) ** 2 + (v2[2] - v3[2]) ** 2)
        ) / 6.0
        if avg_sq < 1e-12:
            return 1e6 / (weight + 1e-6)
        return (avg_sq**0.5) / (weight + 1e-6)

    def _time_score(self, tetra_id: str, geometric_dist: float, path_penalty: float) -> float:
        tetra = self._tetrahedra[tetra_id]
        fil = tetra.filtration(self._time_lambda)
        time_bonus = 1.0 / (1.0 + fil * 0.5)
        weight_factor = tetra.weight / (tetra.init_weight + 1e-6)
        return geometric_dist * path_penalty * time_bonus / (weight_factor + 0.1)

    def navigate_topology(
        self, seed_id: str, max_steps: int = 30, strategy: str = "bfs"
    ) -> List[Tuple[str, str, int]]:
        """Pure topology navigation — no Euclidean distance used.

        Returns list of (tetra_id, connection_type, hop_distance).
        connection_type: 'face', 'edge', 'vertex'
        hop_distance: number of hops from seed
        """
        with self._lock:
            if seed_id not in self._tetrahedra:
                return []

            visited = {seed_id}
            result = [(seed_id, "seed", 0)]
            frontier_face = [(seed_id, 0)]
            frontier_edge: List[Tuple[str, int]] = []
            frontier_vertex: List[Tuple[str, int]] = []

            for _ in range(max_steps):
                next_frontier = []
                for tid, hop in frontier_face:
                    for nid in self._face_neighbors(tid):
                        if nid not in visited and nid in self._tetrahedra:
                            visited.add(nid)
                            self._tetrahedra[nid].touch()
                            result.append((nid, "face", hop + 1))
                            next_frontier.append((nid, hop + 1))
                    for nid in self._edge_neighbors(tid):
                        if nid not in visited:
                            frontier_edge.append((nid, hop + 1))
                frontier_face = next_frontier

                next_edge = []
                for nid, hop in frontier_edge:
                    if nid not in visited and nid in self._tetrahedra:
                        visited.add(nid)
                        self._tetrahedra[nid].touch()
                        result.append((nid, "edge", hop))
                        next_frontier.append((nid, hop))
                    elif nid not in visited:
                        next_edge.append((nid, hop))
                frontier_face.extend(next_frontier)
                frontier_edge = next_edge

                if not frontier_face and not frontier_edge:
                    break

            for nid, hop in frontier_edge:
                if nid not in visited and nid in self._tetrahedra:
                    visited.add(nid)
                    result.append((nid, "vertex", hop))

            return result

    def seed_by_label(self, labels: List[str]) -> Optional[str]:
        """Pure topology seed — find tetra by label match, no geometry."""
        best_id = None
        best_overlap = 0
        best_access = -1
        for tid, tetra in self._tetrahedra.items():
            overlap = len(set(tetra.labels) & set(labels))
            if overlap > best_overlap or (overlap == best_overlap and tetra.access_count > best_access):
                best_overlap = overlap
                best_access = tetra.access_count
                best_id = tid
        return best_id

    def seed_by_structure(self, query_labels: Optional[List[str]] = None) -> Optional[str]:
        """Pure topology seed — pick highest-face-connectivity tetra.
        No Euclidean distance. Uses structural importance."""
        if query_labels:
            labeled = self.seed_by_label(query_labels)
            if labeled:
                return labeled

        if self._hub_node_id is not None and self._hub_node_id in self._tetrahedra:
            return self._hub_node_id

        best_id = None
        best_score = -1.0
        for tid, tetra in self._tetrahedra.items():
            face_nb = len(self._face_neighbors(tid))
            edge_nb = len(self._edge_neighbors(tid))
            struct_score = face_nb * 3.0 + edge_nb * 1.0 + tetra.weight * 0.5
            if struct_score > best_score:
                best_score = struct_score
                best_id = tid
        self._hub_node_id = best_id
        self._hub_node_score = best_score
        return best_id

    def _add_vertex(self, point: np.ndarray) -> int:
        idx = len(self._vertices)
        self._vertices.append(point.copy())
        return idx

    def _ref_vertices(self, vindices: Tuple[int, int, int, int]) -> None:
        for vi in vindices:
            self._vertex_ref_count[vi] += 1

    def _unref_vertices(self, vindices: Tuple[int, int, int, int]) -> None:
        for vi in vindices:
            self._vertex_ref_count[vi] -= 1
        if len(self._vertices) >= self._vertex_compact_threshold:
            self._try_compact_vertices()

    def _face_key(self, a: int, b: int, c: int) -> Tuple[int, int, int]:
        return tuple(sorted([a, b, c]))

    def _faces_of_tetra(self, vi: Tuple[int, int, int, int]) -> List[Tuple[int, int, int]]:
        a, b, c, d = vi
        return [
            self._face_key(a, b, c),
            self._face_key(a, b, d),
            self._face_key(a, c, d),
            self._face_key(b, c, d),
        ]

    def _create_seed_tetrahedron(self, center: np.ndarray) -> Tuple[int, int, int, int]:
        r = 0.25
        golden = (1 + 5**0.5) / 2
        verts = []
        for i in range(4):
            theta = np.arccos(1 - 2 * (i + 0.5) / 4)
            phi = 2 * np.pi * i / golden
            offset = r * np.array(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            )
            verts.append(self._add_vertex(center + offset))
        return tuple(verts)

    def _attach_to_boundary(self, seed_point: np.ndarray) -> Tuple[int, int, int, int]:
        if self._boundary_dirty and not self._boundary_face_keys:
            self._rebuild_boundary_cache()

        last_id = self._last_tetra_id
        if last_id is not None and last_id in self._tetrahedra:
            best_face = None
            best_dist = float("inf")
            sx, sy, sz = float(seed_point[0]), float(seed_point[1]), float(seed_point[2])
            for fk in self._faces_of_tetra(self._tetrahedra[last_id].vertex_indices):
                face = self._faces.get(fk)
                if face is not None and face.is_boundary:
                    va, vb, vc = (self._vertices[v] for v in fk)
                    cx = (float(va[0]) + float(vb[0]) + float(vc[0])) / 3.0
                    cy = (float(va[1]) + float(vb[1]) + float(vc[1])) / 3.0
                    cz = (float(va[2]) + float(vb[2]) + float(vc[2])) / 3.0
                    d = (cx - sx) ** 2 + (cy - sy) ** 2 + (cz - sz) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_face = fk
            if best_face is not None:
                new_vi = self._add_vertex(seed_point)
                return (best_face[0], best_face[1], best_face[2], new_vi)

        bf_keys = self._boundary_face_keys
        if not bf_keys:
            return self._create_seed_tetrahedron(seed_point)

        bf_centroids = self._boundary_centroids
        if not bf_centroids:
            return self._create_seed_tetrahedron(seed_point)

        n = len(bf_keys)
        if n > 128:
            sample_size = min(128, n)
            indices = np.random.choice(n, size=sample_size, replace=False)
            best_idx = indices[0]
            best_dist = float("inf")
            for si in indices:
                c = bf_centroids[si]
                d = float(np.sum((c - seed_point) ** 2))
                if d < best_dist:
                    best_dist = d
                    best_idx = si
            best_face = bf_keys[best_idx]
        else:
            best_idx = 0
            best_dist = float("inf")
            for i in range(n):
                c = bf_centroids[i]
                d = float(np.sum((c - seed_point) ** 2))
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            best_face = bf_keys[best_idx]

        new_vi = self._add_vertex(seed_point)
        return (best_face[0], best_face[1], best_face[2], new_vi)

    def _rebuild_boundary_cache(self) -> None:
        bf_keys = []
        bf_centroids = []
        for fk, face in self._faces.items():
            if face.is_boundary:
                centroid = np.mean([self._vertices[v] for v in fk], axis=0)
                bf_keys.append(fk)
                bf_centroids.append(centroid)
        self._boundary_face_keys = bf_keys
        self._boundary_centroids = bf_centroids
        self._boundary_dirty = False

    def _prune_boundary_cache(self) -> None:
        pruned_keys = []
        pruned_centroids = []
        for i, fk in enumerate(self._boundary_face_keys):
            face = self._faces.get(fk)
            if face is not None and face.is_boundary:
                pruned_keys.append(fk)
                pruned_centroids.append(self._boundary_centroids[i])
        if len(pruned_keys) == len(self._boundary_face_keys):
            return
        self._boundary_face_keys = pruned_keys
        self._boundary_centroids = pruned_centroids

    def _nearest_tetrahedron(self, point: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._tetrahedra:
            return None, float("inf")
        if len(self._tetrahedra) > 50 and self._centroid_index_dirty:
            self._rebuild_centroid_index()
        if self._centroid_matrix is not None and len(self._centroid_matrix) > 50:
            diffs = self._centroid_matrix - point
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            idx = int(np.argmin(dists))
            return self._centroid_ids[idx], float(dists[idx])
        best_id = None
        best_dist = float("inf")
        for tid, tetra in self._tetrahedra.items():
            d = np.linalg.norm(point - tetra.centroid)
            if d < best_dist:
                best_dist = d
                best_id = tid
        return best_id, best_dist

    def _rebuild_centroid_index(self) -> None:
        if not self._tetrahedra:
            self._centroid_matrix = None
            self._centroid_ids = []
            self._centroid_index_dirty = False
            return
        self._centroid_ids = list(self._tetrahedra.keys())
        self._centroid_matrix = np.array(
            [self._tetrahedra[tid].centroid for tid in self._centroid_ids]
        )
        self._centroid_index_dirty = False

    def _invalidate_centroid_index(self) -> None:
        self._centroid_index_dirty = True

    def _face_neighbors(self, tetra_id: str) -> Set[str]:
        tetra = self._tetrahedra.get(tetra_id)
        if tetra is None:
            return set()
        neighbors = set()
        for fk in self._faces_of_tetra(tetra.vertex_indices):
            face = self._faces.get(fk)
            if face:
                for tid in face.tetrahedra:
                    if tid != tetra_id:
                        neighbors.add(tid)
        return neighbors

    def _edge_neighbors(self, tetra_id: str) -> Set[str]:
        tetra = self._tetrahedra.get(tetra_id)
        if tetra is None:
            return set()
        my_verts = set(tetra.vertex_indices)
        neighbors = set()
        for vi in tetra.vertex_indices:
            for tid in self._vertex_to_tetra.get(vi, set()):
                if tid != tetra_id and tid not in neighbors:
                    other = self._tetrahedra.get(tid)
                    if other:
                        shared = my_verts & set(other.vertex_indices)
                        if len(shared) == 2:
                            neighbors.add(tid)
        return neighbors

    def _vertex_neighbors(
        self, tetra_id: str, face_nb: Optional[Set[str]] = None, edge_nb: Optional[Set[str]] = None
    ) -> Set[str]:
        tetra = self._tetrahedra.get(tetra_id)
        if tetra is None:
            return set()
        my_verts = set(tetra.vertex_indices)
        neighbors = set()
        for vi in tetra.vertex_indices:
            for tid in self._vertex_to_tetra.get(vi, set()):
                if tid != tetra_id:
                    other = self._tetrahedra.get(tid)
                    if other:
                        shared = my_verts & set(other.vertex_indices)
                        if len(shared) >= 1:
                            neighbors.add(tid)
        if face_nb is None:
            face_nb = self._face_neighbors(tetra_id)
        if edge_nb is None:
            edge_nb = self._edge_neighbors(tetra_id)
        return neighbors - face_nb - edge_nb

    def _association_score(self, id1: str, id2: str, conn_type: str) -> float:
        t1 = self._tetrahedra[id1]
        t2 = self._tetrahedra[id2]

        type_w = {"shared_face": 1.0, "shared_edge": 0.6, "shared_vertex": 0.3}
        base = type_w.get(conn_type, 0.1)

        v1 = self._tetra_volume(t1)
        v2 = self._tetra_volume(t2)
        vol_ratio = min(v1, v2) / (max(v1, v2) + 1e-12)

        w_sim = 1.0 - abs(t1.weight - t2.weight) / max(t1.weight, t2.weight, 0.1)

        if t1.labels and t2.labels:
            l_jac = len(set(t1.labels) & set(t2.labels)) / len(set(t1.labels) | set(t2.labels))
        else:
            l_jac = 0.0

        return base * 0.5 + vol_ratio * 0.2 + w_sim * 0.15 + l_jac * 0.15

    def _tetra_volume(self, tetra: MemoryTetrahedron) -> float:
        vs = [self._vertices[i] for i in tetra.vertex_indices]
        mat = np.array([vs[1] - vs[0], vs[2] - vs[0], vs[3] - vs[0]])
        return abs(float(np.linalg.det(mat))) / 6.0

    def _remove_tetrahedron(self, tetra_id: str) -> None:
        tetra = self._tetrahedra.pop(tetra_id, None)
        if tetra is None:
            return
        for label in tetra.labels:
            self._label_index[label].discard(tetra_id)
        for fk in self._faces_of_tetra(tetra.vertex_indices):
            face = self._faces.get(fk)
            if face:
                face.tetrahedra.discard(tetra_id)
                if not face.tetrahedra:
                    del self._faces[fk]
        for vi in tetra.vertex_indices:
            self._vertex_to_tetra[vi].discard(tetra_id)
        self._unref_vertices(tetra.vertex_indices)
        self._invalidate_centroid_index()

    def _try_compact_vertices(self) -> None:
        unused = [i for i in range(len(self._vertices)) if self._vertex_ref_count.get(i, 0) <= 0]
        if not unused:
            return
        unused_set = set(unused)
        if len(unused) < len(self._vertices) * 0.3:
            return

        old_to_new: Dict[int, int] = {}
        new_vertices = []
        for old_idx in range(len(self._vertices)):
            if old_idx not in unused_set:
                new_idx = len(new_vertices)
                old_to_new[old_idx] = new_idx
                new_vertices.append(self._vertices[old_idx])

        if not new_vertices and self._tetrahedra:
            return

        self._vertices = new_vertices

        new_ref_count: Dict[int, int] = defaultdict(int)
        new_v2t: Dict[int, Set[str]] = defaultdict(set)

        for tid, tetra in self._tetrahedra.items():
            new_vi = tuple(old_to_new[vi] for vi in tetra.vertex_indices)
            tetra.vertex_indices = new_vi
            for vi in new_vi:
                new_ref_count[vi] += 1
                new_v2t[vi].add(tid)

        self._vertex_ref_count = new_ref_count
        self._vertex_to_tetra = new_v2t

        new_faces: Dict[Tuple[int, int, int], FaceRecord] = {}
        for old_fk, face in self._faces.items():
            new_fk = tuple(sorted(old_to_new[vi] for vi in old_fk))
            face.vertex_indices = new_fk
            new_faces[new_fk] = face
        self._faces = new_faces

        self._boundary_dirty = True
        self._centroid_index_dirty = True


class TetraMeshStore:
    """SQLite-backed persistence for TetraMesh.

    Replaces JSON replay with incremental SQLite operations:
    - UPSERT on store (no full dump)
    - Streaming SELECT on load (constant memory)
    - Atomic WAL mode for crash safety
    - Full mesh topology preserved: vertices, faces, tetrahedra
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS vertices (
        idx INTEGER PRIMARY KEY,
        x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS tetrahedra (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        vertex_indices TEXT NOT NULL,
        centroid TEXT NOT NULL,
        labels TEXT NOT NULL DEFAULT '[]',
        metadata TEXT NOT NULL DEFAULT '{{}}',
        weight REAL NOT NULL DEFAULT 1.0,
        creation_time REAL NOT NULL DEFAULT 0.0,
        last_access_time REAL NOT NULL DEFAULT 0.0,
        init_weight REAL NOT NULL DEFAULT 1.0,
        spatial_alpha REAL NOT NULL DEFAULT 0.0,
        integration_count INTEGER NOT NULL DEFAULT 0,
        access_count INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS faces (
        face_key TEXT PRIMARY KEY,
        tetrahedra_ids TEXT NOT NULL DEFAULT '[]'
    );
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """

    def __init__(self, db_path: str = "tetramem.db"):
        self._db_path = db_path
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def init_db(self) -> None:
        conn = self._conn()
        conn.executescript(self.SCHEMA)
        conn.commit()

    def save_tetra(self, t: MemoryTetrahedron) -> None:
        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO tetrahedra
            (id, content, vertex_indices, centroid, labels, metadata,
             weight, creation_time, last_access_time, init_weight,
             spatial_alpha, integration_count, access_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                t.id,
                t.content,
                json.dumps(t.vertex_indices),
                json.dumps(t.centroid.tolist() if hasattr(t.centroid, 'tolist') else list(t.centroid)),
                json.dumps(t.labels),
                json.dumps(t.metadata, default=str),
                t.weight,
                t.creation_time,
                t.last_access_time,
                t.init_weight,
                t._spatial_alpha,
                t.integration_count,
                t.access_count,
            ),
        )
        conn.commit()

    def save_face(self, face_key: Tuple[int, int, int], tetra_ids: Set[str]) -> None:
        conn = self._conn()
        fk_str = json.dumps(face_key)
        conn.execute(
            "INSERT OR REPLACE INTO faces (face_key, tetrahedra_ids) VALUES (?, ?)",
            (fk_str, json.dumps(sorted(tetra_ids))),
        )
        conn.commit()

    def save_vertex(self, idx: int, point: np.ndarray) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO vertices (idx, x, y, z) VALUES (?, ?, ?, ?)",
            (idx, float(point[0]), float(point[1]), float(point[2])),
        )
        conn.commit()

    def save_meta(self, key: str, value: str) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    def save_full_mesh(self, mesh: 'TetraMesh') -> None:
        conn = self._conn()
        with mesh._lock:
            conn.execute("BEGIN")
            try:
                conn.execute("DELETE FROM vertices")
                conn.execute("DELETE FROM tetrahedra")
                conn.execute("DELETE FROM faces")

                for idx, v in enumerate(mesh._vertices):
                    conn.execute(
                        "INSERT INTO vertices (idx, x, y, z) VALUES (?, ?, ?, ?)",
                        (idx, float(v[0]), float(v[1]), float(v[2])),
                    )

                for tid, t in mesh._tetrahedra.items():
                    conn.execute(
                        """INSERT INTO tetrahedra
                        (id, content, vertex_indices, centroid, labels, metadata,
                         weight, creation_time, last_access_time, init_weight,
                         spatial_alpha, integration_count, access_count)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            tid,
                            t.content,
                            json.dumps(t.vertex_indices),
                            json.dumps(t.centroid.tolist() if hasattr(t.centroid, 'tolist') else list(t.centroid)),
                            json.dumps(t.labels),
                            json.dumps(t.metadata, default=str),
                            t.weight,
                            t.creation_time,
                            t.last_access_time,
                            t.init_weight,
                            t._spatial_alpha,
                            t.integration_count,
                            t.access_count,
                        ),
                    )

                for fk, face in mesh._faces.items():
                    conn.execute(
                        "INSERT INTO faces (face_key, tetrahedra_ids) VALUES (?, ?)",
                        (json.dumps(fk), json.dumps(sorted(face.tetrahedra))),
                    )

                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    ("time_lambda", str(mesh._time_lambda)),
                )

                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def load_full_mesh(self) -> Optional['TetraMesh']:
        conn = self._conn()
        row = conn.execute("SELECT value FROM meta WHERE key='time_lambda'").fetchone()
        time_lambda = float(row["value"]) if row else 0.001
        mesh = TetraMesh(time_lambda=time_lambda)

        vertex_rows = conn.execute("SELECT idx, x, y, z FROM vertices ORDER BY idx").fetchall()
        vertices = {}
        for r in vertex_rows:
            vertices[r["idx"]] = np.array([r["x"], r["y"], r["z"]], dtype=np.float32)
            if r["idx"] >= len(mesh._vertices):
                mesh._vertices.extend([np.zeros(3)] * (r["idx"] - len(mesh._vertices) + 1))
            mesh._vertices[r["idx"]] = vertices[r["idx"]]

        tetra_rows = conn.execute("SELECT * FROM tetrahedra").fetchall()
        for r in tetra_rows:
            vi = tuple(json.loads(r["vertex_indices"]))
            centroid = np.array(json.loads(r["centroid"]), dtype=np.float32)
            t = MemoryTetrahedron(
                id=r["id"],
                content=r["content"],
                vertex_indices=vi,
                centroid=centroid,
                labels=json.loads(r["labels"]),
                metadata=json.loads(r["metadata"]),
                weight=r["weight"],
                creation_time=r["creation_time"],
                last_access_time=r["last_access_time"],
                init_weight=r["init_weight"],
                _spatial_alpha=r["spatial_alpha"],
                integration_count=r["integration_count"],
                access_count=r["access_count"],
            )
            mesh._tetrahedra[r["id"]] = t
            for lbl in t.labels:
                mesh._label_index[lbl].add(r["id"])
            for v in vi:
                mesh._vertex_to_tetra[v].add(r["id"])
                mesh._vertex_ref_count[v] += 1

        face_rows = conn.execute("SELECT face_key, tetrahedra_ids FROM faces").fetchall()
        for r in face_rows:
            fk = tuple(json.loads(r["face_key"]))
            tids = set(json.loads(r["tetrahedra_ids"]))
            mesh._faces[fk] = FaceRecord(vertex_indices=fk, tetrahedra=tids)

        mesh._centroid_index_dirty = True
        mesh._boundary_dirty = True
        mesh._query_cache_dirty = True
        if mesh._tetrahedra:
            mesh._last_tetra_id = list(mesh._tetrahedra.keys())[-1]

        return mesh

    def incremental_save(self, mesh: 'TetraMesh', dirty_ids: Optional[Set[str]] = None) -> None:
        conn = self._conn()
        with mesh._lock:
            targets = dirty_ids if dirty_ids else set(mesh._tetrahedra.keys())
            for tid in targets:
                t = mesh._tetrahedra.get(tid)
                if t is None:
                    conn.execute("DELETE FROM tetrahedra WHERE id=?", (tid,))
                    continue
                conn.execute(
                    """INSERT OR REPLACE INTO tetrahedra
                    (id, content, vertex_indices, centroid, labels, metadata,
                     weight, creation_time, last_access_time, init_weight,
                     spatial_alpha, integration_count, access_count)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        tid,
                        t.content,
                        json.dumps(t.vertex_indices),
                        json.dumps(t.centroid.tolist() if hasattr(t.centroid, 'tolist') else list(t.centroid)),
                        json.dumps(t.labels),
                        json.dumps(t.metadata, default=str),
                        t.weight,
                        t.creation_time,
                        t.last_access_time,
                        t.init_weight,
                        t._spatial_alpha,
                        t.integration_count,
                        t.access_count,
                    ),
                )
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        conn = self._conn()
        nt = conn.execute("SELECT COUNT(*) FROM tetrahedra").fetchone()[0]
        nv = conn.execute("SELECT COUNT(*) FROM vertices").fetchone()[0]
        nf = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        return {"db_tetrahedra": nt, "db_vertices": nv, "db_faces": nf}

    def close(self) -> None:
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
