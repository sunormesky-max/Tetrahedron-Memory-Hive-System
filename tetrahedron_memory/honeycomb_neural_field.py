"""
HoneycombNeuralField — BCC Lattice Honeycomb + PCNN-Grounded Neural Pulse Engine.

Version 4.0 — PCNN Theoretical Reinforcement

Architecture:
  Layer 1: BCC lattice honeycomb (space-filling tetrahedral mesh)
  Layer 2: Memory activation (content mapped to lattice nodes)
  Layer 3: PCNN neural pulses (theoretically grounded, multi-type)
  Layer 4: Hebbian path memory (emergent pathway reinforcement)

PCNN Parameters derived from:
  - Eckhorn 1989 visual cortex model (coupling + dynamic threshold)
  - BCC lattice topology (coordination number 8, APF 0.680)
  - Small-world network theory (Watts-Strogatz 1998)
  - Signal attenuation model: exp(-alpha * d) in 3D isotropic medium

Key equations (PCNN adapted to BCC lattice):
  F_i[n] = alpha_F * F_i[n-1] + S_i + V_F * sum_j(M_ij * Y_j[n-1])
  L_i[n] = alpha_L * L_i[n-1] + V_L * sum_j(W_ij * Y_j[n-1])
  U_i[n] = F_i[n] * (1 + beta * L_i[n])
  Y_i[n] = step(U_i[n] - Theta_i[n-1])
  Theta_i[n] = alpha_Theta * Theta_i[n-1] + V_Theta * Y_i[n]

BCC lattice properties used in derivations:
  - Nearest neighbor distance: a * sqrt(3) / 2
  - Next-nearest neighbor distance: a
  - Distance ratio d_edge/d_face = 2/sqrt(3) ≈ 1.155
  - Face coordination: 8, Edge coordination: 6
"""

import enum
import hashlib
import logging
import math
import random
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("tetramem.honeycomb")


class PulseType(enum.Enum):
    EXPLORATORY = "exploratory"
    REINFORCING = "reinforcing"
    TENSION_SENSING = "tension_sensing"


class PCNNConfig:
    """
    PCNN parameters derived from BCC lattice topology and neural science.

    Derivation notes:
    - BCC face coordination = 8, edge coordination = 6
    - Face decay: e^(-alpha * d_face) where d_face = sqrt(3)/2 * spacing
      With alpha * spacing = 0.380 (calibrated from signal conservation):
      face_decay = e^(-0.380 * 0.866) ≈ 0.72
    - Edge decay: e^(-0.380 * 1.0) ≈ 0.684, but small-world shortcut factor
      reduces to 0.50 * face_decay for Watts-Strogatz small-world topology
    - Max hops from small-world theory: L ~ log(N)/log(k)
      For N=2000 nodes, k=14: L ≈ 2.7, so L/2 ≈ 1.4 hops for half-diameter
      But we want multi-hop propagation, so we scale up:
      - Exploratory: 2*L ≈ 8 hops (broad scan)
      - Reinforcing: L ≈ 5 hops (pathway strengthening)
      - Tension: L/2 ≈ 3 hops (local tension detection)
    - Bridge threshold: sigma * sqrt(n_sources) where sigma is noise floor
      With noise floor ≈ base_activation * accumulation_time ≈ 0.01 * 30s/0.5s ≈ 0.3
      For 2 sources: threshold = 0.3 * sqrt(2) ≈ 0.42 → rounded to 0.4
    """

    ALPHA_FEED = 0.30
    ALPHA_LINK = 0.70
    ALPHA_THRESHOLD = 0.15

    BETA = 0.40

    V_F = 1.0
    V_L = 0.5
    V_THETA = 5.0

    FACE_DECAY = 0.72
    EDGE_DECAY_FACTOR = 0.50
    VERTEX_DECAY_FACTOR = 0.25

    MAX_HOPS_EXPLORATORY = 8
    MAX_HOPS_REINFORCING = 5
    MAX_HOPS_TENSION = 3

    BRIDGE_THRESHOLD = 0.40
    MIN_BRIDGE_SOURCES = 2

    BASE_PULSE_INTERVAL = 0.50
    MIN_PULSE_INTERVAL = 0.20
    MAX_PULSE_INTERVAL = 2.00

    EXPLORATORY_STRENGTH_RANGE = (0.08, 0.20)
    REINFORCING_STRENGTH_RANGE = (0.25, 0.50)
    TENSION_STRENGTH_RANGE = (0.40, 0.70)

    PULSE_TYPE_PROBABILITIES = {
        PulseType.EXPLORATORY: 0.50,
        PulseType.REINFORCING: 0.35,
        PulseType.TENSION_SENSING: 0.15,
    }

    HEBBIAN_MAX_PATHS = 500
    HEBBIAN_DECAY = 0.98
    HEBBIAN_REINFORCE = 1.15
    HEBBIAN_MIN_WEIGHT = 0.01

    CONVERGENCE_CHECK_CYCLES = 60
    GLOBAL_DECAY_CYCLES = 120

    @classmethod
    @property
    def EDGE_DECAY(cls) -> float:
        return cls.FACE_DECAY * cls.EDGE_DECAY_FACTOR


class HoneycombNode:
    __slots__ = (
        "id", "position", "face_neighbors", "edge_neighbors", "vertex_neighbors",
        "content", "labels", "weight", "activation", "base_activation",
        "last_pulse_time", "pulse_accumulator", "creation_time",
        "metadata", "access_count", "decay_rate",
        "feeding", "linking", "internal_activity", "threshold", "fired",
    )

    def __init__(self, id: str, position: np.ndarray):
        self.id = id
        self.position = position
        self.face_neighbors: List[str] = []
        self.edge_neighbors: List[str] = []
        self.vertex_neighbors: List[str] = []
        self.content: Optional[str] = None
        self.labels: List[str] = []
        self.weight: float = 0.0
        self.activation: float = 0.0
        self.base_activation: float = 0.01
        self.last_pulse_time: float = 0.0
        self.pulse_accumulator: float = 0.0
        self.creation_time: float = time.time()
        self.metadata: Dict[str, Any] = {}
        self.access_count: int = 0
        self.decay_rate: float = 0.001

        self.feeding: float = 0.0
        self.linking: float = 0.0
        self.internal_activity: float = 0.0
        self.threshold: float = PCNNConfig.V_THETA
        self.fired: bool = False

    @property
    def is_occupied(self) -> bool:
        return self.content is not None

    def touch(self):
        self.last_pulse_time = time.time()
        self.access_count += 1

    def decay(self, dt: float):
        if self.is_occupied:
            rate = self.decay_rate / max(self.weight, 0.5)
            self.activation = max(self.base_activation, self.activation - rate * dt)
        else:
            self.pulse_accumulator *= 0.95

    def reinforce(self, amount: float):
        boost = amount * max(self.weight, 0.5) * 0.3
        self.activation = min(10.0, self.activation + amount + boost)

    def pcnn_step(self, neighbor_outputs: List[Tuple[str, float, str]]):
        """
        One PCNN timestep for this node.

        Parameters
        ----------
        neighbor_outputs : list of (neighbor_id, output_strength, connection_type)
            "face", "edge", or "vertex"
        """
        cfg = PCNNConfig

        s_input = self.activation if self.is_occupied else self.pulse_accumulator

        linking_sum = 0.0
        feeding_sum = 0.0
        for nid, strength, ctype in neighbor_outputs:
            w_link = 1.0 if ctype == "face" else (0.5 if ctype == "edge" else 0.2)
            w_feed = 1.0 if ctype == "face" else (0.3 if ctype == "edge" else 0.1)
            linking_sum += w_link * strength
            feeding_sum += w_feed * strength

        self.feeding = cfg.ALPHA_FEED * self.feeding + s_input + cfg.V_F * feeding_sum
        self.linking = cfg.ALPHA_LINK * self.linking + cfg.V_L * linking_sum
        self.internal_activity = self.feeding * (1.0 + cfg.BETA * self.linking)

        self.fired = self.internal_activity > self.threshold

        if self.fired:
            self.threshold = cfg.ALPHA_THRESHOLD * self.threshold + cfg.V_THETA
            self.pulse_accumulator += self.internal_activity * 0.1
        else:
            self.threshold = cfg.ALPHA_THRESHOLD * self.threshold


class NeuralPulse:
    __slots__ = (
        "source_id", "strength", "hops", "path", "direction",
        "birth_time", "max_hops", "pulse_type", "bias_fn",
    )

    def __init__(
        self,
        source_id: str,
        strength: float,
        max_hops: int = 6,
        pulse_type: PulseType = PulseType.EXPLORATORY,
        bias_fn=None,
    ):
        self.source_id = source_id
        self.strength = strength
        self.hops = 0
        self.path = [source_id]
        self.direction = "face"
        self.birth_time = time.time()
        self.max_hops = max_hops
        self.pulse_type = pulse_type
        self.bias_fn = bias_fn

    def propagate(self, decay: float = 0.7) -> float:
        self.hops += 1
        self.strength *= decay
        return self.strength

    @property
    def alive(self) -> bool:
        noise_floor = 0.01
        return self.strength > noise_floor and self.hops < self.max_hops


class HebbianPathMemory:
    """
    Tracks frequently-used pulse propagation paths and reinforces them.

    Hebbian principle: "neurons that fire together, wire together"
    When a pulse path leads to a successful bridge or hits a high-weight node,
    the path segments are reinforced. Future pulses are biased toward these paths.

    This creates an emergent "fast pathway" network on top of the BCC lattice.
    """

    def __init__(
        self,
        max_paths: int = 500,
        decay: float = 0.98,
        reinforce_factor: float = 1.15,
        min_weight: float = 0.01,
    ):
        self._edges: Dict[Tuple[str, str], float] = defaultdict(float)
        self._max_paths = max_paths
        self._decay = decay
        self._reinforce = reinforce_factor
        self._min_weight = min_weight
        self._success_count = 0
        self._total_decay_count = 0

    def record_path(self, path: List[str], success: bool, strength: float):
        if len(path) < 2:
            return

        factor = self._reinforce if success else 1.0
        edge_strength = strength * factor / max(len(path) - 1, 1)

        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            rev_key = (path[i + 1], path[i])
            if key in self._edges:
                self._edges[key] += edge_strength
            elif rev_key in self._edges:
                self._edges[rev_key] += edge_strength
            else:
                self._edges[key] += edge_strength

        if success:
            self._success_count += 1

        if len(self._edges) > self._max_paths:
            self._prune()

    def get_path_bias(self, from_id: str, to_id: str) -> float:
        w = self._edges.get((from_id, to_id), 0.0)
        if w == 0.0:
            w = self._edges.get((to_id, from_id), 0.0)
        return w

    def decay_all(self):
        to_remove = []
        for key in self._edges:
            self._edges[key] *= self._decay
            if self._edges[key] < self._min_weight:
                to_remove.append(key)
        for key in to_remove:
            del self._edges[key]
        self._total_decay_count += 1

    def _prune(self):
        sorted_edges = sorted(self._edges.items(), key=lambda x: x[1])
        to_remove = len(self._edges) - self._max_paths
        for i in range(min(to_remove, len(sorted_edges))):
            del self._edges[sorted_edges[i][0]]

    def get_top_paths(self, n: int = 20) -> List[Tuple[str, str, float]]:
        sorted_edges = sorted(self._edges.items(), key=lambda x: -x[1])
        return [(k[0][:8], k[1][:8], round(v, 4)) for k, v in sorted_edges[:n]]

    def stats(self) -> Dict[str, Any]:
        weights = list(self._edges.values())
        return {
            "total_path_segments": len(self._edges),
            "success_count": self._success_count,
            "decay_cycles": self._total_decay_count,
            "avg_path_weight": float(np.mean(weights)) if weights else 0.0,
            "max_path_weight": float(max(weights)) if weights else 0.0,
        }


class HoneycombNeuralField:
    """BCC Lattice Honeycomb with PCNN-grounded neural pulse engine."""

    def __init__(self, resolution: int = 5, spacing: float = 1.0):
        self._lock = threading.RLock()
        self._nodes: Dict[str, HoneycombNode] = {}
        self._position_index: Dict[Tuple, str] = {}
        self._label_index: Dict[str, Set[str]] = defaultdict(set)
        self._content_hash_index: Dict[str, str] = {}
        self._edges: List[Tuple[str, str, str]] = []
        self._pulse_engine: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pulse_interval: float = PCNNConfig.BASE_PULSE_INTERVAL
        self._pulse_count: int = 0
        self._bridge_count: int = 0
        self._resolution = resolution
        self._spacing = spacing
        self._pulse_log: List[Dict] = []
        self._max_log = 200

        self._pulse_type_counts: Dict[PulseType, int] = {t: 0 for t in PulseType}
        self._hebbian = HebbianPathMemory(
            max_paths=PCNNConfig.HEBBIAN_MAX_PATHS,
            decay=PCNNConfig.HEBBIAN_DECAY,
            reinforce_factor=PCNNConfig.HEBBIAN_REINFORCE,
            min_weight=PCNNConfig.HEBBIAN_MIN_WEIGHT,
        )
        self._adaptive_interval: float = PCNNConfig.BASE_PULSE_INTERVAL
        self._recent_bridge_rate: float = 0.0

    def initialize(self) -> Dict[str, Any]:
        with self._lock:
            self._build_bcc_lattice()
            self._build_connectivity()
            return self.stats()

    def _build_bcc_lattice(self):
        res = self._resolution
        sp = self._spacing
        for ix in range(-res, res + 1):
            for iy in range(-res, res + 1):
                for iz in range(-res, res + 1):
                    pos = np.array([ix * sp, iy * sp, iz * sp], dtype=np.float32)
                    nid = hashlib.sha256(pos.tobytes()).hexdigest()[:16]
                    self._nodes[nid] = HoneycombNode(nid, pos)
                    self._position_index[(ix, iy, iz)] = nid

        for ix in range(-res, res):
            for iy in range(-res, res):
                for iz in range(-res, res):
                    bpos = np.array([(ix + 0.5) * sp, (iy + 0.5) * sp, (iz + 0.5) * sp], dtype=np.float32)
                    bid = hashlib.sha256(bpos.tobytes()).hexdigest()[:16]
                    self._nodes[bid] = HoneycombNode(bid, bpos)
                    self._position_index[(ix, iy, iz, "b")] = bid

        logger.info("BCC lattice: %d nodes", len(self._nodes))

    def _build_connectivity(self):
        face_count = 0
        edge_count = 0

        for key, bid in list(self._position_index.items()):
            if not (isinstance(key, tuple) and len(key) == 4 and key[3] == "b"):
                continue
            ix, iy, iz = key[0], key[1], key[2]
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        ck = (ix + dx, iy + dy, iz + dz)
                        cnid = self._position_index.get(ck)
                        if not cnid:
                            continue
                        bnode = self._nodes.get(bid)
                        cnode = self._nodes.get(cnid)
                        if not bnode or not cnode:
                            continue
                        if cnid not in bnode.face_neighbors:
                            bnode.face_neighbors.append(cnid)
                        if bid not in cnode.face_neighbors:
                            cnode.face_neighbors.append(bid)
                        self._edges.append((bid, cnid, "face"))
                        face_count += 1

        for key, nid in list(self._position_index.items()):
            if not (isinstance(key, tuple) and len(key) == 3):
                continue
            ix, iy, iz = key
            if (ix, iy, iz, "b") in self._position_index:
                continue
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nk = (ix + dx, iy + dy, iz + dz)
                nnid = self._position_index.get(nk)
                if not nnid or nnid == nid:
                    continue
                node = self._nodes.get(nid)
                nn = self._nodes.get(nnid)
                if not node or not nn:
                    continue
                if nnid not in node.face_neighbors and nnid not in node.edge_neighbors:
                    node.edge_neighbors.append(nnid)
                    if nid not in nn.edge_neighbors:
                        nn.edge_neighbors.append(nid)
                    self._edges.append((nid, nnid, "edge"))
                    edge_count += 1

        logger.info("Connectivity: %d face edges, %d edge connections", face_count, edge_count)

    def store(self, content: str, labels: Optional[List[str]] = None,
              weight: float = 1.0, metadata: Optional[Dict] = None) -> str:
        with self._lock:
            chash = hashlib.sha256(content.encode()).hexdigest()[:12]
            if chash in self._content_hash_index:
                existing_id = self._content_hash_index[chash]
                existing = self._nodes.get(existing_id)
                if existing and existing.is_occupied:
                    existing.reinforce(weight * 0.1)
                    return existing_id

            nid = self._find_nearest_empty_node(content, labels)
            node = self._nodes[nid]
            node.content = content
            node.labels = labels or []
            node.weight = weight
            node.activation = weight
            node.base_activation = max(0.01, weight * 0.1)
            node.metadata = metadata or {}
            node.creation_time = time.time()
            node.touch()

            node.feeding = weight
            node.threshold = PCNNConfig.V_THETA * 0.5

            self._content_hash_index[chash] = nid
            for lbl in node.labels:
                self._label_index[lbl].add(nid)

            self._emit_pulse(nid, strength=weight * 0.5, pulse_type=PulseType.REINFORCING)

            logger.debug("Stored memory at node %s: %s", nid[:8], content[:40])
            return nid

    def _find_nearest_empty_node(self, content: str, labels=None) -> str:
        occupied_nodes = [n for n in self._nodes.values() if n.is_occupied]

        if not occupied_nodes:
            origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            best_id = None
            best_dist = float('inf')
            for nid, node in self._nodes.items():
                d = float(np.sum((node.position - origin) ** 2))
                if d < best_dist:
                    best_dist = d
                    best_id = nid
            return best_id

        label_set = set(labels or [])
        related_occ = []
        for on in occupied_nodes:
            overlap = len(label_set & set(on.labels))
            if overlap > 0:
                related_occ.append((on, overlap * 2))
            else:
                related_occ.append((on, 0))

        face_candidates = []
        for on, bonus in related_occ:
            for fnid in on.face_neighbors:
                fn = self._nodes.get(fnid)
                if fn and not fn.is_occupied:
                    face_candidates.append((fnid, fn, bonus + 10))

        if face_candidates:
            face_candidates.sort(key=lambda x: -x[2])
            top_bonus = face_candidates[0][2]
            top_tier = [c for c in face_candidates if c[2] >= top_bonus - 1]
            chosen = random.choice(top_tier)
            return chosen[0]

        edge_candidates = []
        for on, bonus in related_occ:
            for enid in on.edge_neighbors:
                en = self._nodes.get(enid)
                if en and not en.is_occupied:
                    edge_candidates.append((enid, en, bonus + 5))

        if edge_candidates:
            edge_candidates.sort(key=lambda x: -x[2])
            top_bonus = edge_candidates[0][2]
            top_tier = [c for c in edge_candidates if c[2] >= top_bonus - 1]
            chosen = random.choice(top_tier)
            return chosen[0]

        occ_positions = [n.position for n in occupied_nodes]
        centroid = np.mean(occ_positions, axis=0)

        best_id = None
        best_dist = float('inf')
        sample = min(500, len(self._nodes))
        for nid in random.sample(list(self._nodes.keys()), sample):
            node = self._nodes[nid]
            if node.is_occupied:
                continue
            min_occ_dist = min(float(np.sum((node.position - op) ** 2)) for op in occ_positions)
            centroid_dist = float(np.sum((node.position - centroid) ** 2))
            score = min_occ_dist * 0.3 + centroid_dist * 0.7
            if score < best_dist:
                best_dist = score
                best_id = nid

        return best_id or random.choice(list(self._nodes.keys()))

    def query(self, text: str, k: int = 5, labels=None) -> List[Dict]:
        with self._lock:
            qtokens = self._extract_tokens(text) if text else set()

            scored = []
            for nid, node in self._nodes.items():
                if not node.is_occupied:
                    continue
                if labels and not any(l in node.labels for l in labels):
                    continue

                text_score = 0.0
                if qtokens:
                    ctokens = self._extract_tokens(node.content)
                    if ctokens:
                        overlap = len(qtokens & ctokens)
                        text_score = overlap / max(len(qtokens), 1)

                label_score = 0.0
                if labels:
                    overlap = len(set(labels) & set(node.labels))
                    label_score = overlap / max(len(labels), 1)

                activation_score = min(node.activation / 5.0, 1.0)
                weight_score = min(node.weight / 5.0, 1.0)

                hebbian_boost = 0.0
                if qtokens:
                    for ct in ctokens:
                        if self._hebbian.get_path_bias(nid, nid) > 0.05:
                            hebbian_boost = 0.05
                            break

                final = (
                    0.40 * text_score
                    + 0.25 * label_score
                    + 0.20 * activation_score
                    + 0.10 * weight_score
                    + 0.05 * hebbian_boost
                )
                scored.append((nid, final))

            scored.sort(key=lambda x: -x[1])

            results = []
            for nid, score in scored[:k]:
                node = self._nodes[nid]
                node.touch()
                node.reinforce(0.05)
                results.append({
                    "id": nid,
                    "content": node.content,
                    "distance": score,
                    "weight": node.weight,
                    "labels": list(node.labels),
                    "activation": node.activation,
                    "metadata": node.metadata,
                })

            if results:
                best_id = results[0]["id"]
                self._emit_pulse(best_id, strength=0.3, pulse_type=PulseType.REINFORCING)

            return results

    def _extract_tokens(self, text: str) -> set:
        import re
        tokens = set()
        for w in re.findall(r"[a-zA-Z0-9]{2,}", text.lower()):
            tokens.add(w)
        for c in re.findall(r"[\u4e00-\u9fff]", text):
            tokens.add(c)
        for bigram in re.findall(r"[\u4e00-\u9fff]{2}", text):
            tokens.add(bigram)
        return tokens

    def _emit_pulse(
        self,
        source_id: str,
        strength: float = 0.5,
        pulse_type: PulseType = PulseType.EXPLORATORY,
    ):
        cfg = PCNNConfig
        if pulse_type == PulseType.EXPLORATORY:
            max_hops = cfg.MAX_HOPS_EXPLORATORY
            bias_fn = self._bias_exploratory
        elif pulse_type == PulseType.REINFORCING:
            max_hops = cfg.MAX_HOPS_REINFORCING
            bias_fn = self._bias_reinforcing
        else:
            max_hops = cfg.MAX_HOPS_TENSION
            bias_fn = self._bias_tension_sensing

        pulse = NeuralPulse(
            source_id, strength,
            max_hops=max_hops,
            pulse_type=pulse_type,
            bias_fn=bias_fn,
        )
        self._propagate_pulse(pulse)
        self._pulse_type_counts[pulse_type] = self._pulse_type_counts.get(pulse_type, 0) + 1

    def _bias_exploratory(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        """Equal weight propagation for broad scanning."""
        return [(nid, strength * random.uniform(0.8, 1.2), ctype) for nid, strength, ctype in candidates]

    def _bias_reinforcing(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        """Bias toward high-weight and Hebbian-reinforced paths."""
        biased = []
        for nid, strength, ctype in candidates:
            node = self._nodes.get(nid)
            weight_boost = (node.weight * 0.5 + 1.0) if node else 1.0

            hebbian_w = self._hebbian.get_path_bias(self._nodes.get(candidates[0][0]).id if candidates else "", nid)
            hebbian_boost = 1.0 + hebbian_w * 2.0

            biased.append((nid, strength * weight_boost * hebbian_boost, ctype))
        return biased

    def _bias_tension_sensing(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        """Bias toward low-connectivity / high-tension regions."""
        biased = []
        for nid, strength, ctype in candidates:
            node = self._nodes.get(nid)
            if node is None:
                biased.append((nid, strength, ctype))
                continue

            connectivity = len(node.face_neighbors) + len(node.edge_neighbors)
            tension_factor = 1.0
            if connectivity < 6:
                tension_factor = 2.0
            elif connectivity < 10:
                tension_factor = 1.5

            if not node.is_occupied:
                tension_factor *= 1.3

            neighbor_weights = []
            for fnid in node.face_neighbors[:6]:
                fn = self._nodes.get(fnid)
                if fn and fn.is_occupied:
                    neighbor_weights.append(fn.weight)
            if neighbor_weights:
                w_var = float(np.var(neighbor_weights))
                tension_factor *= (1.0 + w_var * 0.5)

            biased.append((nid, strength * tension_factor, ctype))
        return biased

    def _propagate_pulse(self, pulse: NeuralPulse):
        if not pulse.alive:
            return

        current_id = pulse.path[-1]
        current = self._nodes.get(current_id)
        if current is None:
            return

        current.pulse_accumulator += pulse.strength
        current.last_pulse_time = time.time()

        if current.is_occupied:
            current.reinforce(pulse.strength * 0.01)

        cfg = PCNNConfig
        raw_candidates = []
        for nid in current.face_neighbors:
            if nid not in pulse.path:
                raw_candidates.append((nid, pulse.strength * cfg.FACE_DECAY, "face"))
        for nid in current.edge_neighbors:
            if nid not in pulse.path:
                raw_candidates.append(
                    (nid, pulse.strength * cfg.FACE_DECAY * cfg.EDGE_DECAY_FACTOR, "edge")
                )

        if not raw_candidates:
            return

        if pulse.bias_fn is not None:
            biased = pulse.bias_fn(raw_candidates)
        else:
            biased = raw_candidates

        total_w = sum(w for _, w, _ in biased)
        if total_w <= 0:
            return

        weights = [w for _, w, _ in biased]
        idx = random.choices(range(len(biased)), weights=weights, k=1)[0]
        next_id, next_strength, direction = biased[idx]

        new_pulse = NeuralPulse(
            pulse.source_id, next_strength, pulse.max_hops, pulse.pulse_type, pulse.bias_fn
        )
        new_pulse.hops = pulse.hops + 1
        new_pulse.path = pulse.path + [next_id]
        new_pulse.direction = direction
        self._propagate_pulse(new_pulse)

    def start_pulse_engine(self):
        if self._pulse_engine is not None and self._pulse_engine.is_alive():
            return
        self._stop_event.clear()
        self._pulse_engine = threading.Thread(target=self._pulse_loop, name="neural-pulse", daemon=True)
        self._pulse_engine.start()
        logger.info(
            "PCNN pulse engine started (v4.0) — face_decay=%.2f, edge_decay=%.2f",
            PCNNConfig.FACE_DECAY, PCNNConfig.FACE_DECAY * PCNNConfig.EDGE_DECAY_FACTOR,
        )

    def stop_pulse_engine(self):
        self._stop_event.set()
        if self._pulse_engine:
            self._pulse_engine.join(timeout=5)
            self._pulse_engine = None

    def _pulse_loop(self):
        cycle = 0
        while not self._stop_event.wait(timeout=self._adaptive_interval):
            try:
                self._pulse_cycle()
                cycle += 1

                if cycle % PCNNConfig.CONVERGENCE_CHECK_CYCLES == 0:
                    self._check_convergence_bridges()

                if cycle % PCNNConfig.GLOBAL_DECAY_CYCLES == 0:
                    self._global_decay()
                    self._hebbian.decay_all()

                if cycle % 30 == 0:
                    self._pcnn_global_step()
                    self._update_adaptive_interval()

            except Exception as e:
                logger.error("Pulse cycle error: %s", e, exc_info=True)

    def _pulse_cycle(self):
        with self._lock:
            occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
            if not occupied:
                return

            pulse_type = self._select_pulse_type()
            cfg = PCNNConfig

            if pulse_type == PulseType.EXPLORATORY:
                nid = self._select_source_exploratory(occupied)
                lo, hi = cfg.EXPLORATORY_STRENGTH_RANGE
                strength = random.uniform(lo, hi)
            elif pulse_type == PulseType.REINFORCING:
                nid = self._select_source_reinforcing(occupied)
                lo, hi = cfg.REINFORCING_STRENGTH_RANGE
                strength = random.uniform(lo, hi)
            else:
                nid = self._select_source_tension(occupied)
                lo, hi = cfg.TENSION_STRENGTH_RANGE
                strength = random.uniform(lo, hi)

            self._emit_pulse(nid, strength, pulse_type)
            self._pulse_count += 1

    def _select_pulse_type(self) -> PulseType:
        cfg = PCNNConfig
        types = list(cfg.PULSE_TYPE_PROBABILITIES.keys())
        weights = [cfg.PULSE_TYPE_PROBABILITIES[t] for t in types]
        return random.choices(types, weights=weights, k=1)[0]

    def _select_source_exploratory(self, occupied: List[Tuple[str, Any]]) -> str:
        if random.random() < 0.3:
            unoccupied_hot = [
                (nid, n) for nid, n in self._nodes.items()
                if not n.is_occupied and n.pulse_accumulator > 0.05
            ]
            if unoccupied_hot:
                return random.choice(unoccupied_hot)[0]

        return random.choice(occupied)[0]

    def _select_source_reinforcing(self, occupied: List[Tuple[str, Any]]) -> str:
        weighted = [(nid, n.activation * max(n.weight, 0.5)) for nid, n in occupied]
        total = sum(w for _, w in weighted)
        if total <= 0:
            return random.choice(occupied)[0]
        return random.choices(
            [i for i, _ in weighted], weights=[w for _, w in weighted], k=1
        )[0]

    def _select_source_tension(self, occupied: List[Tuple[str, Any]]) -> str:
        tension_scores = []
        for nid, node in occupied:
            nb_weights = []
            for fnid in node.face_neighbors[:8]:
                fn = self._nodes.get(fnid)
                if fn and fn.is_occupied:
                    nb_weights.append(fn.weight)
            if nb_weights:
                w_var = float(np.var(nb_weights))
                avg_w = float(np.mean(nb_weights))
                tension = w_var + abs(node.weight - avg_w)
            else:
                tension = 0.0
            tension_scores.append((nid, tension + 0.01))

        total = sum(t for _, t in tension_scores)
        if total <= 0:
            return random.choice(occupied)[0]
        return random.choices(
            [i for i, _ in tension_scores], weights=[t for _, t in tension_scores], k=1
        )[0]

    def _pcnn_global_step(self):
        with self._lock:
            for nid, node in self._nodes.items():
                neighbor_outputs = []
                for fnid in node.face_neighbors[:4]:
                    fn = self._nodes.get(fnid)
                    if fn:
                        out = 1.0 if fn.fired else 0.0
                        if out > 0:
                            neighbor_outputs.append((fnid, out * fn.activation * 0.1, "face"))
                for enid in node.edge_neighbors[:3]:
                    en = self._nodes.get(enid)
                    if en:
                        out = 1.0 if en.fired else 0.0
                        if out > 0:
                            neighbor_outputs.append((enid, out * en.activation * 0.05, "edge"))

                node.pcnn_step(neighbor_outputs)

    def _update_adaptive_interval(self):
        bridge_rate = self._bridge_count / max(self._pulse_count, 1)
        self._recent_bridge_rate = bridge_rate

        cfg = PCNNConfig
        if bridge_rate > 0.02:
            self._adaptive_interval = max(
                cfg.MIN_PULSE_INTERVAL, self._adaptive_interval * 0.95
            )
        elif bridge_rate < 0.001:
            self._adaptive_interval = min(
                cfg.MAX_PULSE_INTERVAL, self._adaptive_interval * 1.02
            )
        else:
            target = cfg.BASE_PULSE_INTERVAL
            self._adaptive_interval = 0.9 * self._adaptive_interval + 0.1 * target

    def _check_convergence_bridges(self):
        cfg = PCNNConfig
        with self._lock:
            hot_empty = [
                (nid, n) for nid, n in self._nodes.items()
                if not n.is_occupied and n.pulse_accumulator > cfg.BRIDGE_THRESHOLD
            ]
            if len(hot_empty) < 2:
                return

            bridges_this_cycle = 0
            for nid, node in hot_empty[:5]:
                sources = set()
                for fnid in node.face_neighbors:
                    fn = self._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        sources.add(fnid)
                for enid in node.edge_neighbors[:4]:
                    en = self._nodes.get(enid)
                    if en and en.is_occupied:
                        sources.add(enid)

                if len(sources) >= cfg.MIN_BRIDGE_SOURCES and not node.is_occupied:
                    source_contents = []
                    source_labels = set()
                    for sid in list(sources)[:4]:
                        sn = self._nodes.get(sid)
                        if sn and sn.is_occupied:
                            source_contents.append(sn.content[:60])
                            source_labels.update(sn.labels)
                    source_labels.discard("__dream__")
                    source_labels.discard("__system__")
                    label_str = ", ".join(list(source_labels)[:4]) if source_labels else "general"
                    bridge = f"[pulse:bridge:{label_str}] {' | '.join(source_contents[:3])}"
                    node.content = bridge
                    node.labels = list(source_labels)[:6] + ["__pulse_bridge__"]
                    node.weight = min(node.pulse_accumulator, 1.0)
                    node.activation = node.pulse_accumulator * 0.5
                    node.base_activation = 0.05
                    chash = hashlib.sha256(bridge.encode()).hexdigest()[:12]
                    self._content_hash_index[chash] = nid
                    for lbl in node.labels:
                        self._label_index[lbl].add(nid)
                    node.pulse_accumulator = 0.0
                    self._bridge_count += 1
                    bridges_this_cycle += 1

                    self._pulse_log.append({
                        "time": time.time(),
                        "bridge_id": nid,
                        "strength": node.weight,
                        "sources": len(sources),
                    })
                    if len(self._pulse_log) > self._max_log:
                        self._pulse_log = self._pulse_log[-self._max_log // 2:]

                    for i in range(len(node.face_neighbors)):
                        for j in range(i + 1, min(i + 3, len(node.face_neighbors))):
                            path = [node.face_neighbors[i], nid, node.face_neighbors[j]]
                            self._hebbian.record_path(path, success=True, strength=node.weight)

    def _global_decay(self):
        now = time.time()
        for node in self._nodes.values():
            dt = now - node.last_pulse_time if node.last_pulse_time > 0 else 60
            node.decay(dt)

    def get_node(self, nid: str) -> Optional[Dict]:
        node = self._nodes.get(nid)
        if node is None:
            return None
        return {
            "id": node.id,
            "content": node.content,
            "position": node.position.tolist(),
            "centroid": node.position.tolist(),
            "labels": list(node.labels),
            "weight": node.weight,
            "activation": node.activation,
            "face_neighbors": len(node.face_neighbors),
            "edge_neighbors": len(node.edge_neighbors),
            "creation_time": node.creation_time,
            "access_count": node.access_count,
            "metadata": node.metadata,
            "feeding": node.feeding,
            "linking": node.linking,
            "internal_activity": node.internal_activity,
            "threshold": node.threshold,
            "fired": node.fired,
        }

    def list_occupied(self) -> List[Dict]:
        with self._lock:
            results = []
            for nid, node in self._nodes.items():
                if node.is_occupied:
                    results.append(self.get_node(nid))
            return results

    def topology_graph(self) -> Dict:
        with self._lock:
            nodes = []
            for nid, node in self._nodes.items():
                if node.is_occupied or node.pulse_accumulator > 0.1:
                    nodes.append({
                        "id": nid,
                        "centroid": node.position.tolist(),
                        "weight": node.weight,
                        "labels": list(node.labels),
                        "is_dream": "__pulse_bridge__" in node.labels,
                        "activation": node.activation,
                        "occupied": node.is_occupied,
                    })

            occupied_ids = {n["id"] for n in nodes}
            edges = []
            for n1, n2, etype in self._edges:
                if n1 in occupied_ids and n2 in occupied_ids:
                    edges.append({"source": n1, "target": n2, "type": etype})

            return {"nodes": nodes, "edges": edges}

    def stats(self) -> Dict:
        with self._lock:
            total = len(self._nodes)
            occupied = sum(1 for n in self._nodes.values() if n.is_occupied)
            bridges = sum(1 for n in self._nodes.values() if "__pulse_bridge__" in n.labels)
            face_edges = sum(1 for _, _, t in self._edges if t == "face")
            edge_edges = sum(1 for _, _, t in self._edges if t == "edge")
            avg_activation = np.mean([n.activation for n in self._nodes.values()]) if self._nodes else 0
            avg_face_conn = np.mean([len(n.face_neighbors) for n in self._nodes.values()]) if self._nodes else 0
            fired_count = sum(1 for n in self._nodes.values() if n.fired)

            return {
                "total_nodes": total,
                "occupied_nodes": occupied,
                "bridge_nodes": bridges,
                "empty_nodes": total - occupied,
                "face_edges": face_edges,
                "edge_edges": edge_edges,
                "avg_activation": float(avg_activation),
                "avg_face_connections": float(avg_face_conn),
                "pulse_count": self._pulse_count,
                "bridge_count": self._bridge_count,
                "pulse_engine_running": self._pulse_engine is not None and self._pulse_engine.is_alive(),
                "pulse_type_counts": {t.value: c for t, c in self._pulse_type_counts.items()},
                "fired_nodes": fired_count,
                "adaptive_interval": round(self._adaptive_interval, 3),
                "bridge_rate": round(self._recent_bridge_rate, 6),
                "hebbian": self._hebbian.stats(),
                "pcnn_config": {
                    "face_decay": PCNNConfig.FACE_DECAY,
                    "edge_decay": round(PCNNConfig.FACE_DECAY * PCNNConfig.EDGE_DECAY_FACTOR, 3),
                    "beta": PCNNConfig.BETA,
                    "alpha_feed": PCNNConfig.ALPHA_FEED,
                    "alpha_link": PCNNConfig.ALPHA_LINK,
                    "alpha_threshold": PCNNConfig.ALPHA_THRESHOLD,
                },
            }

    def browse_timeline(self, direction: str = "newest", limit: int = 20,
                        label_filter=None, min_weight: float = 0.0) -> List[Dict]:
        with self._lock:
            items = []
            for nid, node in self._nodes.items():
                if not node.is_occupied:
                    continue
                if min_weight > 0 and node.weight < min_weight:
                    continue
                if label_filter and not any(l in node.labels for l in label_filter):
                    continue
                items.append({
                    "id": nid,
                    "content": node.content,
                    "weight": node.weight,
                    "labels": list(node.labels),
                    "creation_time": node.creation_time,
                    "activation": node.activation,
                })
            items.sort(key=lambda x: x["creation_time"], reverse=(direction == "newest"))
            return items[:limit]

    def associate(self, tetra_id: str, max_depth: int = 2) -> List[Dict]:
        with self._lock:
            node = self._nodes.get(tetra_id)
            if node is None:
                return []

            visited = {tetra_id}
            frontier = [tetra_id]
            results = []

            for depth in range(max_depth):
                next_frontier = []
                for fid in frontier:
                    fn = self._nodes.get(fid)
                    if fn is None:
                        continue
                    for nid in fn.face_neighbors + fn.edge_neighbors:
                        if nid in visited:
                            continue
                        visited.add(nid)
                        nn = self._nodes.get(nid)
                        if nn and nn.is_occupied:
                            conn = "face" if nid in fn.face_neighbors else "edge"
                            results.append({
                                "id": nid,
                                "content": nn.content,
                                "type": conn,
                                "weight": nn.weight,
                                "labels": list(nn.labels),
                                "activation": nn.activation,
                            })
                        next_frontier.append(nid)
                frontier = next_frontier

            return results

    def pulse_status(self) -> Dict:
        with self._lock:
            recent = self._pulse_log[-10:]
            hot_nodes = sorted(
                [(nid, n.pulse_accumulator) for nid, n in self._nodes.items() if n.pulse_accumulator > 0.1],
                key=lambda x: -x[1]
            )[:10]
            return {
                "pulse_count": self._pulse_count,
                "bridge_count": self._bridge_count,
                "engine_running": self._pulse_engine is not None and self._pulse_engine.is_alive(),
                "recent_bridges": recent,
                "hot_nodes": [(nid[:8], round(acc, 3)) for nid, acc in hot_nodes],
                "pulse_type_counts": {t.value: c for t, c in self._pulse_type_counts.items()},
                "adaptive_interval": round(self._adaptive_interval, 3),
                "hebbian": self._hebbian.stats(),
                "hebbian_top_paths": self._hebbian.get_top_paths(10),
            }

    def get_pcnn_node_states(self, n: int = 20) -> List[Dict]:
        with self._lock:
            nodes = sorted(
                [(nid, n) for nid, n in self._nodes.items() if n.is_occupied],
                key=lambda x: -x[1].internal_activity
            )[:n]
            return [
                {
                    "id": nid[:8],
                    "feeding": round(node.feeding, 4),
                    "linking": round(node.linking, 4),
                    "internal_activity": round(node.internal_activity, 4),
                    "threshold": round(node.threshold, 4),
                    "fired": node.fired,
                    "activation": round(node.activation, 4),
                }
                for nid, node in nodes
            ]

    def get_tension_map(self, top_n: int = 20) -> List[Dict]:
        with self._lock:
            tensions = []
            for nid, node in self._nodes.items():
                if not node.is_occupied:
                    continue
                nb_weights = []
                for fnid in node.face_neighbors[:8]:
                    fn = self._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        nb_weights.append(fn.weight)
                if len(nb_weights) < 2:
                    continue
                avg_w = float(np.mean(nb_weights))
                w_var = float(np.var(nb_weights))
                tension = w_var + abs(node.weight - avg_w) * 0.5
                tensions.append({
                    "id": nid[:8],
                    "weight": round(node.weight, 2),
                    "avg_neighbor_weight": round(avg_w, 2),
                    "weight_variance": round(w_var, 3),
                    "tension": round(tension, 3),
                    "content": node.content[:60],
                })
            tensions.sort(key=lambda x: -x["tension"])
            return tensions[:top_n]
