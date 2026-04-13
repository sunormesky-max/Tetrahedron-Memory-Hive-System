"""
Geometry module for Tetrahedron Memory System.

Provides geometric primitives, precise multi-scale text-to-geometry mapping,
and 4D spatiotemporal embedding for the tetrahedral memory space.
"""

import hashlib
import math
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


class GeometryPrimitives:
    """
    Pure geometric primitives for tetrahedra operations.
    """

    @staticmethod
    def tetrahedron_volume(vertices: np.ndarray) -> float:
        if vertices.shape != (4, 3):
            raise ValueError(f"Expected 4x3 array, got {vertices.shape}")
        v0, v1, v2, v3 = vertices
        mat = np.array([v1 - v0, v2 - v0, v3 - v0])
        return abs(np.linalg.det(mat)) / 6.0

    @staticmethod
    def triangle_area(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        cross = np.cross(p1 - p0, p2 - p0)
        return float(np.linalg.norm(cross) / 2.0)

    @staticmethod
    def shared_faces(tet1: np.ndarray, tet2: np.ndarray, tol: float = 1e-6) -> int:
        def get_faces(vertices: np.ndarray) -> List[np.ndarray]:
            return [np.delete(vertices, i, axis=0) for i in range(4)]

        def faces_match(face1: np.ndarray, face2: np.ndarray, tol: float) -> bool:
            face1_sorted = face1[np.lexsort(face1.T)]
            face2_sorted = face2[np.lexsort(face2.T)]
            return np.allclose(face1_sorted, face2_sorted, atol=tol)

        faces1 = get_faces(tet1)
        faces2 = get_faces(tet2)
        shared = 0
        for f1 in faces1:
            for f2 in faces2:
                if faces_match(f1, f2, tol):
                    shared += 1
                    break
        return shared

    @staticmethod
    def shared_vertices(tet1: np.ndarray, tet2: np.ndarray, tol: float = 1e-6) -> int:
        shared = 0
        for v1 in tet1:
            for v2 in tet2:
                if np.allclose(v1, v2, atol=tol):
                    shared += 1
                    break
        return shared

    @staticmethod
    def jaccard_index(tet1: np.ndarray, tet2: np.ndarray, tol: float = 1e-6) -> float:
        shared = GeometryPrimitives.shared_vertices(tet1, tet2, tol)
        union = 8 - shared
        return shared / union if union > 0 else 0.0

    @staticmethod
    def centroid(vertices: np.ndarray) -> np.ndarray:
        return np.mean(vertices, axis=0)

    @staticmethod
    def circumcenter(vertices: np.ndarray) -> np.ndarray:
        p0 = vertices[0]
        A = 2 * (vertices[1:] - p0)
        b = np.sum(vertices[1:] ** 2, axis=1) - np.sum(p0**2)
        try:
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return c
        except np.linalg.LinAlgError:
            return GeometryPrimitives.centroid(vertices)

    @staticmethod
    def circumradius(vertices: np.ndarray) -> float:
        center = GeometryPrimitives.circumcenter(vertices)
        return float(np.linalg.norm(vertices[0] - center))

    @staticmethod
    def is_point_in_tetrahedron(point: np.ndarray, vertices: np.ndarray) -> bool:
        v0, v1, v2, v3 = vertices
        mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
        try:
            coords = np.linalg.solve(mat, point - v0)
            return all(0 <= c <= 1 for c in coords) and sum(coords) <= 1 + 1e-6
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def tetrahedron_intersection(
        tet1: np.ndarray, tet2: np.ndarray, n_samples: int = 1000
    ) -> float:
        all_points = np.vstack([tet1, tet2])
        min_bounds = np.min(all_points, axis=0)
        max_bounds = np.max(all_points, axis=0)
        box_volume = np.prod(max_bounds - min_bounds)
        rng = np.random.default_rng(42)
        points = rng.uniform(min_bounds, max_bounds, (n_samples, 3))
        in_both = 0
        for p in points:
            if GeometryPrimitives.is_point_in_tetrahedron(
                p, tet1
            ) and GeometryPrimitives.is_point_in_tetrahedron(p, tet2):
                in_both += 1
        return (in_both / n_samples) * box_volume


class _NGramFingerprint:
    """
    Multi-scale text fingerprint for geometric mapping.

    Decomposes text into character n-grams and word n-grams,
    producing a deterministic 3-component vector that captures
    both surface form and semantic topic.
    """

    _GOLDEN_RATIOS = (
        (1.0 / 1.618033988749895, 2.0 / 1.618033988749895),
        (3.0 / 1.618033988749895, 5.0 / 1.618033988749895),
        (7.0 / 1.618033988749895, 11.0 / 1.618033988749895),
    )

    @classmethod
    def fingerprint(cls, text: str) -> np.ndarray:
        char_bigrams = cls._char_ngrams(text, 2)
        word_unigrams = cls._word_ngrams(text, 1)
        word_bigrams = cls._word_ngrams(text, 2)

        c1 = cls._hash_component(char_bigrams, cls._GOLDEN_RATIOS[0])
        c2 = cls._hash_component(word_unigrams, cls._GOLDEN_RATIOS[1])
        c3 = cls._hash_component(word_bigrams, cls._GOLDEN_RATIOS[2])

        combined = np.array([c1, c2, c3])

        norm = np.linalg.norm(combined)
        if norm < 1e-12:
            return cls._fallback(text)
        return combined / norm

    @classmethod
    def _char_ngrams(cls, text: str, n: int) -> List[str]:
        normalized = text.lower().strip()
        if len(normalized) < n:
            return [normalized]
        return [normalized[i : i + n] for i in range(len(normalized) - n + 1)]

    @classmethod
    def _word_ngrams(cls, text: str, n: int) -> List[str]:
        words = text.lower().split()
        if not words:
            return ["_empty_"]
        if len(words) < n:
            return [" ".join(words)]
        return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

    @classmethod
    def _hash_component(cls, ngrams: List[str], golden: Tuple[float, float]) -> float:
        if not ngrams:
            return 0.0
        counter = Counter(ngrams)
        a, b = golden
        value = 0.0
        for gram, count in counter.items():
            h = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16)
            angle = (h * a) % (2 * math.pi)
            value += count * math.cos(angle + b)
        return value

    @classmethod
    def _fallback(cls, text: str) -> np.ndarray:
        h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(h % (2**31))
        theta = rng.uniform(0, 2 * math.pi)
        phi = math.acos(2 * rng.uniform() - 1)
        return np.array(
            [
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            ]
        )


class SemanticEmbedder:
    """Optional neural embedding layer for semantic text-to-geometry mapping.

    Falls back to _NGramFingerprint when sentence-transformers is not installed.
    """

    _model = None
    _model_name = None

    @classmethod
    def load(cls, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(model_name)
            cls._model_name = model_name
        except ImportError:
            pass

    @classmethod
    def is_available(cls) -> bool:
        return cls._model is not None

    @classmethod
    def embed(cls, text: str) -> Optional[np.ndarray]:
        if cls._model is None:
            return None
        vec = cls._model.encode(text, normalize_embeddings=True)
        arr = np.asarray(vec, dtype=np.float32)
        return arr[:3] / (np.linalg.norm(arr[:3]) + 1e-12)


class TextToGeometryMapper:
    """
    Maps text strings to geometric coordinates with multi-scale
    fingerprinting and optional 4D spatiotemporal embedding.

    The mapping uses three layers:
    1. N-gram fingerprint: deterministic hash-based direction vector
    2. Label sector: shared labels attract nodes into overlapping sectors
    3. Temporal dimension: optional 4th coordinate encoding time

    This ensures that semantically similar memories (shared vocabulary,
    shared labels) naturally cluster in geometric space, enabling
    genuine topological self-emergence.
    """

    _LABEL_SECTOR_CACHE: Dict[str, np.ndarray] = {}
    _SECTOR_CACHE_LOCK = __import__("threading").Lock()

    def __init__(
        self,
        seed: Optional[int] = None,
        extended: bool = False,
        dimension: int = 3,
        temporal_decay: float = 0.01,
        label_attraction: float = 0.3,
    ):
        self._seed = seed
        self._extended = extended
        self._dimension = dimension
        self._temporal_decay = temporal_decay
        self._label_attraction = label_attraction

    def map_text(self, text: str) -> np.ndarray:
        direction = _NGramFingerprint.fingerprint(text)

        if self._extended:
            h = hashlib.sha256(text.encode()).hexdigest()[:8]
            rng = np.random.RandomState(int(h, 16) % (2**31))
            scale = 5.0 + rng.uniform(0, 15.0)
            direction = direction * scale

        return direction.astype(np.float64)

    def map_text_4d(
        self,
        text: str,
        timestamp: Optional[float] = None,
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        base = self.map_text(text)

        if timestamp is None:
            timestamp = time.time()

        t_coord = math.log1p(timestamp) * self._temporal_decay

        if labels:
            sector = self._compute_label_sector(labels)
            base = base * (1.0 - self._label_attraction) + sector * self._label_attraction
            norm = np.linalg.norm(base)
            if norm > 1e-12:
                base = base / norm

        if self._extended:
            base_4d = np.zeros(4, dtype=np.float64)
            base_4d[:3] = base
            base_4d[3] = t_coord
        else:
            base_4d = np.zeros(4, dtype=np.float64)
            base_4d[:3] = base
            base_4d[3] = t_coord

        return base_4d

    def map_text_weighted(self, text: str, importance: float = 1.0) -> Tuple[np.ndarray, float]:
        position = self.map_text(text)
        init_weight = 1.5 + importance * 4.0
        return position, init_weight

    def map_tetrahedron(self, text: str, scale: float = 1.0) -> np.ndarray:
        vertices = []
        for i in range(4):
            vertex = self.map_text(f"{text}_vertex_{i}")
            vertices.append(vertex * scale)
        return np.array(vertices)

    def map_batch(self, texts: list) -> np.ndarray:
        return np.array([self.map_text(text) for text in texts])

    def get_distance(self, text1: str, text2: str) -> float:
        p1 = self.map_text(text1)
        p2 = self.map_text(text2)
        dot = np.clip(np.dot(p1, p2), -1.0, 1.0)
        return float(np.arccos(dot))

    def get_tetrahedron_similarity(self, text1: str, text2: str) -> float:
        tet1 = self.map_tetrahedron(text1)
        tet2 = self.map_tetrahedron(text2)
        return GeometryPrimitives.jaccard_index(tet1, tet2)

    def _compute_label_sector(self, labels: List[str]) -> np.ndarray:
        if not labels:
            return np.array([1.0, 0.0, 0.0])

        primary = labels[0]
        with self._SECTOR_CACHE_LOCK:
            if primary not in self._LABEL_SECTOR_CACHE:
                h = int(hashlib.sha256(primary.encode()).hexdigest()[:8], 16)
                rng = np.random.RandomState(h % (2**31))
                theta = rng.uniform(0, 2 * math.pi)
                phi = math.acos(2 * rng.uniform() - 1)
                self._LABEL_SECTOR_CACHE[primary] = np.array(
                    [
                        math.sin(phi) * math.cos(theta),
                        math.sin(phi) * math.sin(theta),
                        math.cos(phi),
                    ]
                )
            return self._LABEL_SECTOR_CACHE[primary]


def weighted_tetra_power_radius(points: np.ndarray, weights: np.ndarray) -> float:
    if len(points) != 4:
        return float("inf")
    A, B, C, D = points
    wA, wB, wC, wD = weights
    b_vec = 0.5 * np.array(
        [
            np.dot(B, B) - np.dot(A, A) - (wB - wA),
            np.dot(C, C) - np.dot(A, A) - (wC - wA),
            np.dot(D, D) - np.dot(A, A) - (wD - wA),
        ]
    )
    M = np.stack([B - A, C - A, D - A])
    try:
        O = np.linalg.solve(M, b_vec)
        return float(np.dot(O - A, O - A) - wA)
    except np.linalg.LinAlgError:
        return float("inf")
