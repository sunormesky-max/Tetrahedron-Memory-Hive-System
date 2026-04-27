import re
import math
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .semantic_reasoning import GeometricSemanticReasoner


class GeometricSemanticIndex:
    def __init__(self):
        self._bigram_index: Dict[str, Set[str]] = defaultdict(set)
        self._word_index: Dict[str, Set[str]] = defaultdict(set)
        self._trigram_index: Dict[str, Set[str]] = defaultdict(set)
        self._label_index: Dict[str, Set[str]] = defaultdict(set)
        self._label_graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._domain_centroids: Dict[str, np.ndarray] = {}
        self._node_domains: Dict[str, List[str]] = {}
        self._content_hashes: Dict[str, str] = {}
        self._reasoner: Optional["GeometricSemanticReasoner"] = None

    def index_node(self, node_id: str, content: str, labels: List[str],
                   weight: float, position: np.ndarray):
        if not content:
            return
        for w in self._extract_words(content):
            self._word_index[w].add(node_id)
        for bg in self._extract_bigrams(content):
            self._bigram_index[bg].add(node_id)
        for tg in self._extract_trigrams(content):
            self._trigram_index[tg].add(node_id)
        for lbl in labels:
            self._label_index[lbl].add(node_id)
        for i, l1 in enumerate(labels):
            for l2 in labels[i+1:]:
                self._label_graph[l1][l2] += weight * 0.1
                self._label_graph[l2][l1] += weight * 0.1
        primary = labels[0] if labels else "__default__"
        self._node_domains[node_id] = labels[:3]
        if primary not in self._domain_centroids:
            self._domain_centroids[primary] = position.copy()
        else:
            alpha = 0.1
            self._domain_centroids[primary] = (
                (1 - alpha) * self._domain_centroids[primary] + alpha * position
            )
        self._content_hashes[node_id] = content[:50]

    def remove_node(self, node_id: str):
        for bg, ids in list(self._bigram_index.items()):
            ids.discard(node_id)
            if not ids:
                del self._bigram_index[bg]
        for w, ids in list(self._word_index.items()):
            ids.discard(node_id)
            if not ids:
                del self._word_index[w]
        for tg, ids in list(self._trigram_index.items()):
            ids.discard(node_id)
            if not ids:
                del self._trigram_index[tg]
        for lbl, ids in list(self._label_index.items()):
            ids.discard(node_id)
            if not ids:
                del self._label_index[lbl]
        self._node_domains.pop(node_id, None)
        self._content_hashes.pop(node_id, None)

    def set_reasoner(self, reasoner):
        self._reasoner = reasoner

    def search(self, query: str, k: int = 10,
                labels: Optional[List[str]] = None,
                label_index: Optional[Dict] = None,
                occupied_nodes: Optional[Dict] = None,
                hebbian: Optional[object] = None,
                crystals: Optional[object] = None,
                pulse_accum: Optional[Dict] = None,
                crystal_boost: float = 1.8,
                super_crystal_boost: float = 2.5) -> List[Tuple[str, float]]:
        candidate_scores: Dict[str, float] = defaultdict(float)

        query_words = self._extract_words(query)
        query_bigrams = self._extract_bigrams(query)
        query_trigrams = self._extract_trigrams(query)

        word_hits = set()
        for w in query_words:
            word_hits.update(self._word_index.get(w, set()))

        bigram_hits = set()
        for bg in query_bigrams:
            bigram_hits.update(self._bigram_index.get(bg, set()))

        trigram_hits = set()
        for tg in query_trigrams:
            trigram_hits.update(self._trigram_index.get(tg, set()))

        text_candidates = word_hits | bigram_hits | trigram_hits

        if labels:
            label_candidates = set()
            for lbl in labels:
                label_candidates.update(self._label_index.get(lbl, set()))
            label_candidates.update(self._expand_labels(labels))
            text_candidates = text_candidates | label_candidates

        if not text_candidates and occupied_nodes:
            all_occ = list(occupied_nodes.keys())
            sample_size = min(500, len(all_occ))
            import random
            text_candidates = set(random.sample(all_occ, sample_size)) if all_occ else set()

        for nid in text_candidates:
            node = occupied_nodes.get(nid) if occupied_nodes else None
            if node is None:
                continue

            score = 0.0
            w_overlap = len(query_words & set(self._extract_words(self._content_hashes.get(nid, ""))))
            if query_words:
                score += (w_overlap / max(len(query_words), 1)) * 30.0

            bg_overlap = len(query_bigrams & set(self._extract_bigrams(self._content_hashes.get(nid, ""))))
            if query_bigrams:
                score += (bg_overlap / max(len(query_bigrams), 1)) * 25.0

            tg_overlap = len(query_trigrams & set(self._extract_trigrams(self._content_hashes.get(nid, ""))))
            if query_trigrams:
                score += (tg_overlap / max(len(query_trigrams), 1)) * 20.0

            if labels:
                node_labels = self._node_domains.get(nid, [])
                label_match = len(set(labels) & set(node_labels))
                score += (label_match / max(len(labels), 1)) * 15.0

            if hasattr(node, 'weight'):
                score += min(node.weight, 5.0) * 2.0

            if hasattr(node, 'activation'):
                score += min(node.activation, 5.0) * 1.5

            if pulse_accum and nid in pulse_accum:
                score += min(pulse_accum[nid], 3.0) * 1.0

            if hebbian and hasattr(hebbian, '_edges'):
                for (a, b), edge_data in hebbian._edges.items():
                    related_id = b if a == nid else (a if b == nid else None)
                    if related_id and related_id in text_candidates:
                        ew = edge_data if isinstance(edge_data, (int, float)) else 1.0
                        score += ew * 0.5

            if crystals and hasattr(crystals, '_crystals'):
                for cid, cdata in crystals._crystals.items():
                    if nid in (cdata.get('nodes', []) if isinstance(cdata, dict) else []):
                        cw = cdata.get('weight', 1.8) if isinstance(cdata, dict) else 1.8
                        boost = super_crystal_boost if cw >= 4.0 else crystal_boost
                        score += boost

            candidate_scores[nid] = score

        if self._reasoner and candidate_scores:
            top_ids = sorted(candidate_scores.items(), key=lambda x: -x[1])[:max(k, 5)]
            for top_nid, _ in top_ids:
                expanded = self._reasoner.expand_concept(top_nid, depth=1)
                for exp_nid, strength in expanded.items():
                    if exp_nid in candidate_scores:
                        candidate_scores[exp_nid] += strength * 8.0

            if self._reasoner:
                analogical = self._reasoner.find_analogical_pairs(k=3)
                for a_id, b_id, analogy_score in analogical:
                    if a_id in candidate_scores and b_id not in candidate_scores:
                        candidate_scores[b_id] = candidate_scores[a_id] * analogy_score * 0.3
                    elif b_id in candidate_scores and a_id not in candidate_scores:
                        candidate_scores[a_id] = candidate_scores[b_id] * analogy_score * 0.3

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return sorted_candidates[:k]

    def _expand_labels(self, labels: List[str], depth: int = 1) -> Set[str]:
        expanded = set()
        for lbl in labels:
            for related, weight in self._label_graph.get(lbl, {}).items():
                if weight >= 0.5:
                    expanded.update(self._label_index.get(related, set()))
        return expanded

    @staticmethod
    def _extract_words(text: str) -> Set[str]:
        words = set(re.findall(r'[a-zA-Z]{2,}', text.lower()))
        cn_chars = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        for seg in cn_chars:
            for i in range(len(seg)):
                words.add(seg[i])
            for i in range(len(seg) - 1):
                words.add(seg[i:i+2])
        return words

    @staticmethod
    def _extract_bigrams(text: str) -> Set[str]:
        chars = re.sub(r'\s+', '', text.lower())
        return {chars[i:i+2] for i in range(len(chars) - 1) if len(chars[i:i+2].strip()) == 2}

    @staticmethod
    def _extract_trigrams(text: str) -> Set[str]:
        chars = re.sub(r'\s+', '', text.lower())
        return {chars[i:i+3] for i in range(len(chars) - 2)}

    def stats(self) -> Dict:
        return {
            "word_index_size": len(self._word_index),
            "bigram_index_size": len(self._bigram_index),
            "trigram_index_size": len(self._trigram_index),
            "label_index_size": len(self._label_index),
            "label_graph_edges": sum(len(v) for v in self._label_graph.values()),
            "domain_count": len(self._domain_centroids),
        }
