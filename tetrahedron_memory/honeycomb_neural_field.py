"""
HoneycombNeuralField - BCC Lattice Honeycomb + PCNN-Grounded Neural Pulse Engine.

Version 5.0 - Structural Cascade + Lattice Integrity + Crystallized Pathways

This module contains the main HoneycombNeuralField orchestrator class.
Supporting classes have been extracted into separate modules for maintainability.
All classes are re-exported here for backward compatibility.
"""

import hashlib
import logging
import math
import random
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .pcnn_types import PulseType, PCNNConfig, NeuralPulse
from .honeycomb_node import HoneycombNode
from .crystallized_pathway import CrystallizedPathway
from .hebbian_memory import HebbianPathMemory
from .spatial_reflection import SpatialReflectionField
from .lattice_integrity import LatticeIntegrityReport, LatticeIntegrityChecker
from .self_check import SelfCheckResult, SelfCheckEngine
from .tetrahedral_cell import TetrahedralCell, HoneycombCellMap
from .self_organize import SemanticCluster, OrganizeResult, SelfOrganizeEngine
from .dream_engine import DreamCycleResult, DreamEngine
from .agent_driver import AgentMemoryDriver
from .feedback import FeedbackRecord, FeedbackLoop
from .session import SessionRecord, Session, SessionManager
from .insight_aggregator import InsightAggregator
from .rw_lock import ReadWriteLock
from .self_regulation import SelfRegulationEngine
from .geometry import TextToGeometryMapper

logger = logging.getLogger("tetramem.honeycomb")

__all__ = [
    "PulseType", "PCNNConfig", "NeuralPulse",
    "HoneycombNode",
    "CrystallizedPathway",
    "HebbianPathMemory",
    "SpatialReflectionField",
    "LatticeIntegrityReport", "LatticeIntegrityChecker",
    "SelfCheckResult", "SelfCheckEngine",
    "TetrahedralCell", "HoneycombCellMap",
    "SemanticCluster", "OrganizeResult", "SelfOrganizeEngine",
    "DreamCycleResult", "DreamEngine",
    "AgentMemoryDriver",
    "FeedbackRecord", "FeedbackLoop",
    "SessionRecord", "Session", "SessionManager",
    "InsightAggregator",
    "HoneycombNeuralField",
]

class HoneycombNeuralField:
    """BCC Lattice Honeycomb with PCNN-grounded neural pulse engine."""

    def __init__(self, resolution: int = 5, spacing: float = 1.0):
        self._lock = threading.RLock()
        self._rw_lock = ReadWriteLock()
        self._nodes: Dict[str, HoneycombNode] = {}
        self._position_index: Dict[Tuple, str] = {}
        self._label_index: Dict[str, Set[str]] = defaultdict(set)
        self._content_hash_index: Dict[str, str] = {}
        self._content_token_index: Dict[str, Set[str]] = defaultdict(set)
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
        self._self_check: Optional[SelfCheckEngine] = None
        self._crystallized: CrystallizedPathway = CrystallizedPathway(
            max_crystals=PCNNConfig.CRYSTALLIZE_MAX_PATHS,
            weight_floor=PCNNConfig.CRYSTAL_WEIGHT_FLOOR,
        )
        self._lattice_checker: Optional[LatticeIntegrityChecker] = None
        self._cascade_count: int = 0
        self._crystal_maintenance_cycle: int = 0
        self._self_organize: Optional[SelfOrganizeEngine] = None
        self._cell_map: HoneycombCellMap = HoneycombCellMap()
        self._dream_engine: Optional[DreamEngine] = None
        self._agent_driver: Optional[AgentMemoryDriver] = None
        self._feedback_loop: Optional[FeedbackLoop] = None
        self._session_manager: Optional[SessionManager] = None
        self._reflection_field: Optional[SpatialReflectionField] = None
        self._bcc_unit_index: Dict[str, List[str]] = defaultdict(list)
        self._spatial_autocorrelation: float = 0.0
        self._autocorrelation_history: List[float] = []
        self._current_phase: str = "fluid"
        self._resonance_events: List[Dict] = []
        self._propagation_source: str = ""
        self._stats_cache = None
        self._stats_cache_time = 0
        self._position_rounded_index: Dict[Tuple[int, int, int], str] = {}
        self._occupied_count: int = 0
        self._occupied_ids: Set[str] = set()
        self._centroid: Optional[np.ndarray] = None
        self._frontier_empty: Dict[str, float] = {}
        self._frontier_rebuild_counter: int = 0
        self._temporal_edges: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self._recent_stores: List[Tuple[str, float]] = []
        self._max_recent_stores: int = 50
        self._lifecycle_thresholds = {
            "fresh_seconds": 3600,
            "consolidating_seconds": 86400,
            "crystallized_reinforcement": 5,
            "ancient_days": 7,
        }
        self._dark_plane_config = {
            "surface": {"pulse_weight": 3.0, "query_priority": 1.0, "dream_focus": 0.3},
            "shallow": {"pulse_weight": 2.0, "query_priority": 0.8, "dream_focus": 1.0},
            "deep": {"pulse_weight": 0.8, "query_priority": 0.5, "dream_focus": 0.6},
            "abyss": {"pulse_weight": 0.2, "query_priority": 0.2, "dream_focus": 0.1},
        }
        self._dark_plane_transitions = 0
        self._dark_plane_reawakenings = 0
        self._attention_foci: List[Dict[str, Any]] = []
        self._attention_mask: Dict[str, float] = {}
        self._attention_decay_rate: float = 0.05
        self._attention_diffusion_rate: float = 0.3
        self._attention_max_foci: int = 3
        self._attention_stats: Dict[str, int] = {
            "focus_set": 0,
            "queries_in_focus": 0,
            "pulses_focused": 0,
            "diffusions": 0,
        }
        self._emergence_history: List[Dict[str, Any]] = []
        self._emergence_max_history: int = 200
        self._last_emergence_snapshot_cycle: int = 0
        self._self_regulation: Optional[SelfRegulationEngine] = None
        self._query_emergence_stats: Dict[str, Any] = {
            "hebbian_paths_from_queries": 0,
            "bridge_validations": 0,
            "bridge_hits": 0,
            "resonance_dreams_triggered": 0,
            "query_path_reinforcements": 0,
        }
        self._resonance_dream_cooldown: float = 0.0
        self._geo_mapper = TextToGeometryMapper(label_attraction=0.3)

    def initialize(self) -> Dict[str, Any]:
        with self._lock:
            self._stats_cache = None
            self._stats_cache_time = 0
            self._build_bcc_lattice()
            self._build_connectivity()
            self._cell_map.build(self._nodes, self._position_index, self._spacing)
            self._build_bcc_unit_index()
            self._self_regulation = SelfRegulationEngine(self)
            logger.info(
                "Honeycomb cells: %d tetrahedral cells in %d BCC units",
                len(self._cell_map._cells), len(self._cell_map._bcc_cell_index),
            )
            return self.stats(force=True)

    def _bcc_direction_factor(self, from_pos: np.ndarray, to_pos: np.ndarray) -> float:
        delta = to_pos - from_pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-10:
            return 1.0
        direction = delta / dist
        abs_dir = np.abs(direction)
        abs_dir_sum = float(np.sum(abs_dir))
        if abs_dir_sum < 1e-10:
            return 1.0
        normalized = abs_dir / abs_dir_sum
        ideal_111 = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        alignment_111 = float(1.0 - np.linalg.norm(normalized - ideal_111) / math.sqrt(2.0/3.0))
        ideal_100 = np.array([1.0, 0.0, 0.0])
        alignment_100 = float(1.0 - np.linalg.norm(normalized - ideal_100) / math.sqrt(2.0))
        factor = 1.0 + 0.20 * alignment_111 - 0.12 * alignment_100
        nn_dist = self._spacing * math.sqrt(3) / 2.0
        ideal_step_ratio = nn_dist / max(dist, 1e-10)
        if 0.8 < ideal_step_ratio < 1.3:
            factor += 0.15
        return max(0.7, min(1.4, factor))

    def _compute_node_geometric_quality(self, nid: str) -> float:
        """
        Compute the geometric quality of a memory node based on its
        tetrahedral cell volume, quality, skewness, and jacobian.
        Higher = better geometric harmony = more reliable memory.
        """
        node = self._nodes.get(nid)
        if node is None or not node.is_occupied:
            return 0.0
        cells = self._cell_map.get_cells_for_node(nid)
        if not cells:
            return 0.5
        total_quality = sum(c.quality for c in cells)
        avg_quality = total_quality / len(cells)
        best_volume = max((c.volume for c in cells), default=0)
        volume_factor = min(1.0, best_volume / (self._spacing ** 3 * 0.1178 + 1e-10))
        avg_jacobian = sum(c.jacobian for c in cells) / len(cells)
        avg_skew = sum(c.skewness for c in cells) / len(cells)
        geo_quality = (
            0.35 * avg_quality
            + 0.25 * volume_factor
            + 0.25 * avg_jacobian
            + 0.15 * (1.0 - avg_skew)
        )
        return round(min(1.0, max(0.0, geo_quality)), 4)

    def _compute_geometric_topo_divergence(self, nid: str) -> float:
        """
        Divergence between geometric proximity and topological (label) proximity.
        High divergence = geometric neighbors share few labels = spatial tension.
        This is the core metric for geometric memory understanding.
        """
        node = self._nodes.get(nid)
        if node is None or not node.is_occupied:
            return 0.0
        node_labels = set(l for l in node.labels if not l.startswith("__"))
        if not node_labels:
            return 0.0
        total_divergence = 0.0
        count = 0
        for fnid in node.face_neighbors[:8]:
            fn = self._nodes.get(fnid)
            if fn and fn.is_occupied:
                fn_labels = set(l for l in fn.labels if not l.startswith("__"))
                if fn_labels:
                    jaccard = len(node_labels & fn_labels) / max(len(node_labels | fn_labels), 1)
                    geometric_dist = float(np.linalg.norm(node.position - fn.position))
                    geo_proximity = 1.0 / (1.0 + geometric_dist)
                    divergence = abs(geo_proximity - jaccard)
                    total_divergence += divergence
                    count += 1
        if count == 0:
            return 0.0
        return total_divergence / count

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

        self._position_rounded_index.clear()
        for nid, node in self._nodes.items():
            rp = (int(round(float(node.position[0]))),
                  int(round(float(node.position[1]))),
                  int(round(float(node.position[2]))))
            self._position_rounded_index[rp] = nid

    def _expand_lattice(self):
        old_res = self._resolution
        new_res = old_res + 1
        self._resolution = new_res
        sp = self._spacing
        new_nodes = 0
        new_body = 0
        new_corner_ids = set()
        new_body_ids = set()

        for ix in range(-new_res, new_res + 1):
            for iy in range(-new_res, new_res + 1):
                for iz in range(-new_res, new_res + 1):
                    if abs(ix) <= old_res and abs(iy) <= old_res and abs(iz) <= old_res:
                        continue
                    pos = np.array([ix * sp, iy * sp, iz * sp], dtype=np.float32)
                    nid = hashlib.sha256(pos.tobytes()).hexdigest()[:16]
                    if nid not in self._nodes:
                        self._nodes[nid] = HoneycombNode(nid, pos)
                        self._position_index[(ix, iy, iz)] = nid
                        new_nodes += 1
                        new_corner_ids.add(nid)

        for ix in range(-new_res, new_res):
            for iy in range(-new_res, new_res):
                for iz in range(-new_res, new_res):
                    if abs(ix) < old_res and abs(iy) < old_res and abs(iz) < old_res:
                        continue
                    bpos = np.array([(ix + 0.5) * sp, (iy + 0.5) * sp, (iz + 0.5) * sp], dtype=np.float32)
                    bid = hashlib.sha256(bpos.tobytes()).hexdigest()[:16]
                    key = (ix, iy, iz, "b")
                    if bid not in self._nodes:
                        self._nodes[bid] = HoneycombNode(bid, bpos)
                        self._position_index[key] = bid
                        new_body += 1
                        new_body_ids.add(bid)
                    elif key not in self._position_index:
                        self._position_index[key] = bid

        self._build_shell_connectivity(new_corner_ids, new_body_ids)
        self._cell_map.build(self._nodes, self._position_index, self._spacing)
        self._build_bcc_unit_index()
        self._position_rounded_index.clear()
        for nid, node in self._nodes.items():
            rp = (int(round(float(node.position[0]))),
                  int(round(float(node.position[1]))),
                  int(round(float(node.position[2]))))
            self._position_rounded_index[rp] = nid
        logger.info("Lattice expanded: res %d->%d, +%d corners +%d body, total %d nodes",
                     old_res, new_res, new_nodes, new_body, len(self._nodes))
        self._rebuild_frontier()

    def _build_shell_connectivity(self, new_corner_ids: Set[str], new_body_ids: Set[str]):
        all_new = new_corner_ids | new_body_ids
        face_count = 0
        edge_count = 0
        vertex_count = 0
        for key, bid in list(self._position_index.items()):
            if not (isinstance(key, tuple) and len(key) == 4 and key[3] == "b"):
                continue
            if bid not in all_new:
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
            if nid not in all_new:
                continue
            ix, iy, iz = key
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
            for dx, dy, dz in [(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
                                (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
                                (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]:
                nk = (ix + dx, iy + dy, iz + dz)
                nnid = self._position_index.get(nk)
                if not nnid or nnid == nid:
                    continue
                node = self._nodes.get(nid)
                nn = self._nodes.get(nnid)
                if not node or not nn:
                    continue
                if (nnid not in node.face_neighbors and
                    nnid not in node.edge_neighbors and
                    nnid not in node.vertex_neighbors):
                    node.vertex_neighbors.append(nnid)
                    if nid not in nn.vertex_neighbors:
                        nn.vertex_neighbors.append(nid)
                    self._edges.append((nid, nnid, "vertex"))
                    vertex_count += 1
        logger.info("Shell connectivity: %d face, %d edge, %d vertex", face_count, edge_count, vertex_count)

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

        vertex_count = 0
        for key, nid in list(self._position_index.items()):
            if not (isinstance(key, tuple) and len(key) == 3):
                continue
            ix, iy, iz = key
            for dx, dy, dz in [(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
                                (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
                                (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]:
                nk = (ix + dx, iy + dy, iz + dz)
                nnid = self._position_index.get(nk)
                if not nnid or nnid == nid:
                    continue
                node = self._nodes.get(nid)
                nn = self._nodes.get(nnid)
                if not node or not nn:
                    continue
                if (nnid not in node.face_neighbors and
                    nnid not in node.edge_neighbors and
                    nnid not in node.vertex_neighbors):
                    node.vertex_neighbors.append(nnid)
                    if nid not in nn.vertex_neighbors:
                        nn.vertex_neighbors.append(nid)
                    self._edges.append((nid, nnid, "vertex"))
                    vertex_count += 1

        logger.info("Connectivity: %d face, %d edge, %d vertex", face_count, edge_count, vertex_count)

        seen = set()
        deduped = []
        for e in self._edges:
            key = (min(e[0], e[1]), max(e[0], e[1]), e[2])
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        self._edges = deduped

    def _build_bcc_unit_index(self):
        """
        Build BCC unit cell containment index.
        For each BCC unit cell (body-center + 8 corners), track which nodes
        belong to the same unit. This provides natural geometric neighborhoods
        beyond face/edge/vertex neighbor types.
        """
        self._bcc_unit_index.clear()
        for key, bid in self._position_index.items():
            if not (isinstance(key, tuple) and len(key) == 4 and key[3] == "b"):
                continue
            ix, iy, iz = key[0], key[1], key[2]
            unit_members = [bid]
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        ck = (ix + dx, iy + dy, iz + dz)
                        cnid = self._position_index.get(ck)
                        if cnid:
                            unit_members.append(cnid)
            for nid in unit_members:
                self._bcc_unit_index[nid].append(bid)
        logger.info("BCC unit index: %d nodes mapped to %d unit cells",
                     len(self._bcc_unit_index), len(set(uid for uids in self._bcc_unit_index.values() for uid in uids)))

    def _get_bcc_cellmates(self, nid: str) -> List[str]:
        """Get all nodes sharing at least one BCC unit cell with the given node."""
        unit_ids = self._bcc_unit_index.get(nid, [])
        if not unit_ids:
            return []
        cellmates = set()
        for uid in unit_ids:
            bc_node = self._nodes.get(uid)
            if bc_node is None:
                continue
            for fnid in bc_node.face_neighbors:
                if fnid != nid:
                    cellmates.add(fnid)
            if uid != nid:
                cellmates.add(uid)
        return list(cellmates)

    def _bcc_cell_coherence(self, nid: str) -> float:
        """
        Measure label coherence within the BCC unit cells containing this node.
        High coherence = same-label memories are concentrated in the same unit cell,
        which is geometrically efficient for BCC topology.
        """
        node = self._nodes.get(nid)
        if node is None or not node.is_occupied or not node.labels:
            return 0.5
        node_labels = set(l for l in node.labels if not l.startswith("__"))
        if not node_labels:
            return 0.5
        unit_ids = self._bcc_unit_index.get(nid, [])
        if not unit_ids:
            return 0.5
        coherent = 0
        total = 0
        for uid in unit_ids:
            bc_node = self._nodes.get(uid)
            if bc_node is None:
                continue
            for fnid in bc_node.face_neighbors:
                if fnid == nid:
                    continue
                fn = self._nodes.get(fnid)
                if fn and fn.is_occupied and fn.labels:
                    fn_labels = set(l for l in fn.labels if not l.startswith("__"))
                    if fn_labels:
                        overlap = len(node_labels & fn_labels) / max(len(node_labels | fn_labels), 1)
                        coherent += overlap
                        total += 1
        if total == 0:
            return 0.5
        return coherent / total

    def compute_spatial_autocorrelation(self) -> float:
        with self._lock:
            if hasattr(self, '_spatial_cache_ts') and time.time() - self._spatial_cache_ts < 30:
                return self._spatial_autocorrelation
            occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
            if len(occupied) < 3:
                self._spatial_autocorrelation = 0.0
                self._spatial_cache_ts = time.time()
                return 0.0
            weights = np.array([n.weight for _, n in occupied], dtype=np.float64)
            nid_list = [nid for nid, _ in occupied]
            nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}
            n = len(weights)
            w_mean = float(np.mean(weights))
            deviations = weights - w_mean
            num_sum = 0.0
            den_sum = float(np.sum(deviations ** 2))
            if den_sum < 1e-12:
                self._spatial_autocorrelation = 0.0
                self._spatial_cache_ts = time.time()
                return 0.0
            w_total = 0.0
            for i, nid in enumerate(nid_list):
                node = self._nodes.get(nid)
                if node is None:
                    continue
                for fnid in node.face_neighbors:
                    j = nid_to_idx.get(fnid)
                    if j is not None and j > i:
                        w_ij = 1.0
                        num_sum += w_ij * deviations[i] * deviations[j]
                        w_total += w_ij
                for enid in node.edge_neighbors:
                    j = nid_to_idx.get(enid)
                    if j is not None and j > i:
                        w_ij = 0.5
                        num_sum += w_ij * deviations[i] * deviations[j]
                        w_total += w_ij
            if w_total < 1e-12:
                self._spatial_autocorrelation = 0.0
                self._spatial_cache_ts = time.time()
                return 0.0
            morans_i = (n / w_total) * (num_sum / den_sum)
            morans_i = max(-1.0, min(1.0, morans_i))
            self._spatial_autocorrelation = morans_i
            self._autocorrelation_history.append(morans_i)
            if len(self._autocorrelation_history) > 50:
                self._autocorrelation_history = self._autocorrelation_history[-25:]
            self._spatial_cache_ts = time.time()
            return morans_i

    def _vacancy_attraction(self, nid: str) -> float:
        node = self._nodes.get(nid)
        if node is None or node.is_occupied:
            return 0.0
        counted = set()
        attraction = 0.0
        for fnid in node.face_neighbors:
            fn = self._nodes.get(fnid)
            if fn and fn.is_occupied:
                attraction += fn.weight * 1.0
                counted.add(fnid)
        for enid in node.edge_neighbors:
            en = self._nodes.get(enid)
            if en and en.is_occupied and enid not in counted:
                attraction += en.weight * 0.5
                counted.add(enid)
        for vnid in node.vertex_neighbors:
            vn = self._nodes.get(vnid)
            if vn and vn.is_occupied and vnid not in counted:
                attraction += vn.weight * 0.25
                counted.add(vnid)
        for cmid in self._get_bcc_cellmates(nid):
            cm = self._nodes.get(cmid)
            if cm and cm.is_occupied and cmid not in counted:
                attraction += cm.weight * 0.3
        return attraction

    def _vacancy_attraction_fast(self, nid: str) -> float:
        node = self._nodes.get(nid)
        if node is None or node.is_occupied:
            return 0.0
        attraction = 0.0
        occ = self._occupied_ids
        for fnid in node.face_neighbors[:6]:
            if fnid in occ:
                attraction += self._nodes[fnid].weight
        for enid in node.edge_neighbors[:4]:
            if enid in occ:
                attraction += self._nodes[enid].weight * 0.5
        return attraction

    def _update_frontier_add(self, nid: str):
        node = self._nodes.get(nid)
        if node is None:
            return
        self._frontier_empty.pop(nid, None)
        for fnid in node.face_neighbors[:6]:
            fn = self._nodes.get(fnid)
            if fn and not fn.is_occupied:
                self._frontier_empty[fnid] = self._frontier_empty.get(fnid, 0) + 1.0
        for enid in node.edge_neighbors[:4]:
            en = self._nodes.get(enid)
            if en and not en.is_occupied:
                self._frontier_empty[enid] = self._frontier_empty.get(enid, 0) + 0.5

    def _rebuild_frontier(self):
        self._frontier_empty.clear()
        occ = self._occupied_ids
        for oid in occ:
            node = self._nodes.get(oid)
            if not node:
                continue
            for fnid in node.face_neighbors[:6]:
                fn = self._nodes.get(fnid)
                if fn and not fn.is_occupied:
                    self._frontier_empty[fnid] = self._frontier_empty.get(fnid, 0) + 1.0
            for enid in node.edge_neighbors[:4]:
                en = self._nodes.get(enid)
                if en and not en.is_occupied:
                    self._frontier_empty[enid] = self._frontier_empty.get(enid, 0) + 0.5

    def _evict_for_space(self) -> str:
        best_bridge = None
        best_dream = None
        best_low = None
        best_any = None
        for nid, n in self._nodes.items():
            if not n.is_occupied:
                continue
            if any(l.startswith("__pulse_bridge__") for l in n.labels):
                if best_bridge is None or n.weight < best_bridge[1].weight:
                    best_bridge = (nid, n)
            elif "__dream__" in n.labels and n.weight < 0.5:
                if best_dream is None or n.weight < best_dream[1].weight:
                    best_dream = (nid, n)
            elif n.weight < 0.3:
                if best_low is None or n.weight < best_low[1].weight:
                    best_low = (nid, n)
            if best_any is None or n.weight < best_any[1].weight:
                best_any = (nid, n)
        for category, label in [(best_bridge, "bridge"), (best_dream, "dream"), (best_low, "low-weight")]:
            if category is not None:
                evict_id, evict_node = category
                self._clear_node(evict_id, evict_node)
                logger.info("Evicted %s node %s (w=%.2f) for new memory", label, evict_id[:8], evict_node.weight)
                return evict_id
        if best_any is not None:
            evict_id, evict_node = best_any
            self._clear_node(evict_id, evict_node)
            logger.warning("Evicted lowest-weight node %s (w=%.2f) — no better candidate", evict_id[:8], evict_node.weight)
            return evict_id
        return random.choice(list(self._nodes.keys()))

    def _clear_node(self, nid, node):
        self._stats_cache = None
        chash = hashlib.sha256((node.content or "").encode()).hexdigest()[:12]
        self._content_hash_index.pop(chash, None)
        for lbl in node.labels:
            self._label_index[lbl].discard(nid)
        for tok in self._extract_tokens(node.content or ""):
            self._content_token_index[tok].discard(nid)
        if node.is_occupied:
            self._occupied_ids.discard(nid)
            old_count = self._occupied_count
            self._occupied_count -= 1
            if self._occupied_count <= 0:
                self._centroid = None
                self._occupied_count = 0
            elif self._centroid is not None:
                self._centroid = (self._centroid * old_count - node.position) / max(self._occupied_count, 1)
        node.content = None
        node.labels = []
        node.weight = 0.0
        node.activation = 0.0
        node.base_activation = 0.01
        node.metadata = {}
        node.crystal_channels.clear()
        node.creation_time = 0.0
        node.domain_affinity.clear()

    def store(self, content: str, labels: Optional[List[str]] = None,
              weight: float = 1.0, metadata: Optional[Dict] = None,
              creation_time_override: Optional[float] = None) -> str:
        with self._lock:
            self._stats_cache = None
            chash = hashlib.sha256(content.encode()).hexdigest()[:12]
            if chash in self._content_hash_index:
                existing_id = self._content_hash_index[chash]
                existing = self._nodes.get(existing_id)
                if existing and existing.is_occupied:
                    existing.reinforce(weight * 0.1)
                    if labels:
                        new_labels = set(existing.labels) | set(labels)
                        for lbl in new_labels - set(existing.labels):
                            existing.labels.append(lbl)
                            self._label_index[lbl].add(existing_id)
                    if metadata:
                        merged_meta = {**existing.metadata, **metadata}
                        for k, v in metadata.items():
                            if k in existing.metadata and isinstance(existing.metadata[k], list) and isinstance(v, list):
                                merged_meta[k] = list(set(existing.metadata[k] + v))
                        existing.metadata = merged_meta
                    if weight > existing.weight:
                        existing.weight = min(10.0, existing.weight + (weight - existing.weight) * 0.3)
                    return existing_id

            total = len(self._nodes)
            occupied_count = self._occupied_count
            expand_needed = False
            dense_count = 0
            if total > 0 and occupied_count / total > 0.85:
                expand_needed = True
            elif total > 0 and occupied_count / total > 0.60:
                sample_size = min(200, len(self._occupied_ids))
                sample_ids = random.sample(list(self._occupied_ids), sample_size) if self._occupied_ids else []
                for sid in sample_ids:
                    n = self._nodes.get(sid)
                    if not n:
                        continue
                    occ_nb = sum(
                        1 for fnid in n.face_neighbors[:6]
                        if fnid in self._occupied_ids
                    )
                    if occ_nb >= 5:
                        dense_count += 1
                dense_ratio = dense_count / max(sample_size, 1)
                if dense_ratio > 0.3:
                    expand_needed = True
            if expand_needed:
                logger.info("Lattice occupancy %.1f%% (dense_nodes=%d) — auto-expanding (res %d->%d)",
                            occupied_count / total * 100,
                            dense_count,
                            self._resolution, self._resolution + 1)
                self._expand_lattice()

            nid = self._find_nearest_empty_node(content, labels)
            node = self._nodes[nid]
            if node.is_occupied:
                nid = self._evict_for_space()
                node = self._nodes[nid]
            node.content = content
            node.labels = labels or []
            node.weight = weight
            node.activation = weight
            node.base_activation = max(0.01, weight * 0.1)
            node.metadata = metadata or {}
            node.metadata["geometric_quality"] = float(self._compute_node_geometric_quality(nid))
            node.metadata["geo_topo_divergence"] = float(self._compute_geometric_topo_divergence(nid))
            node.metadata["bcc_cell_coherence"] = float(self._bcc_cell_coherence(nid))
            node.creation_time = creation_time_override if creation_time_override is not None else time.time()
            node.touch()

            node.feeding = weight
            node.threshold = PCNNConfig.V_THETA * 0.5

            self._occupied_ids.add(nid)
            self._occupied_count += 1
            if self._centroid is None:
                self._centroid = node.position.copy()
            else:
                self._centroid = (self._centroid * (self._occupied_count - 1) + node.position) / self._occupied_count
            self._update_frontier_add(nid)

            self._content_hash_index[chash] = nid
            for lbl in node.labels:
                self._label_index[lbl].add(nid)
            for tok in self._extract_tokens(content):
                self._content_token_index[tok].add(nid)

            self._emit_pulse(nid, strength=weight * 0.5, pulse_type=PulseType.REINFORCING)

            self._connect_temporal(nid, weight)

            logger.debug("Stored memory at node %s: %s", nid[:8], content[:40])
            return nid

    def restore_at_position(self, content: str, centroid: list,
                            labels=None, weight: float = 1.0,
                            metadata=None, creation_time_override=None) -> str:
        with self._lock:
            chash = hashlib.sha256(content.encode()).hexdigest()[:12]
            target_pos = np.array(centroid, dtype=np.float32)
            best_id = None
            best_dist = float('inf')

            rp = (int(round(float(centroid[0]))),
                  int(round(float(centroid[1]))),
                  int(round(float(centroid[2]))))
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    for dz in range(-2, 3):
                        candidate_key = (rp[0] + dx, rp[1] + dy, rp[2] + dz)
                        nid = self._position_rounded_index.get(candidate_key)
                        if nid is None:
                            continue
                        node = self._nodes.get(nid)
                        if node is None or node.is_occupied:
                            continue
                        d = float(np.linalg.norm(node.position - target_pos))
                        if d < best_dist:
                            best_dist = d
                            best_id = nid

            if best_id is None:
                nid = self._find_nearest_empty_node(content, labels)
            else:
                nid = best_id
            node = self._nodes[nid]
            node.content = content
            node.labels = labels or []
            node.weight = weight
            node.activation = weight
            node.base_activation = max(0.01, weight * 0.1)
            node.metadata = metadata or {}
            node.metadata.setdefault("quality_factor", self._cell_quality_factor(nid))
            node.creation_time = creation_time_override if creation_time_override is not None else time.time()
            node.touch()
            node.feeding = weight
            node.threshold = PCNNConfig.V_THETA * 0.5
            self._occupied_ids.add(nid)
            self._occupied_count += 1
            if self._centroid is None:
                self._centroid = node.position.copy()
            else:
                self._centroid = (self._centroid * (self._occupied_count - 1) + node.position) / self._occupied_count
            self._update_frontier_add(nid)
            self._content_hash_index[chash] = nid
            for lbl in node.labels:
                self._label_index[lbl].add(nid)
            for tok in self._extract_tokens(content):
                self._content_token_index[tok].add(nid)
            return nid

    def _find_nearest_empty_node(self, content: str, labels=None) -> str:
        if self._occupied_count == 0:
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
        geo_dir = self._geo_mapper.map_text(content)
        if label_set:
            geo_4d = self._geo_mapper.map_text_4d(content, labels=list(label_set))
            geo_dir = geo_4d[:3]
        geo_norm = np.linalg.norm(geo_dir)
        if geo_norm > 1e-12:
            geo_dir = geo_dir / geo_norm
        centroid = self._centroid if self._centroid is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ideal_pos = centroid + geo_dir * self._spacing * 3.0

        if self._frontier_empty:
            if label_set:
                label_empty = set()
                for lbl in label_set:
                    for rid in self._label_index.get(lbl, set()):
                        rn = self._nodes.get(rid)
                        if not rn:
                            continue
                        for fnid in rn.face_neighbors[:6]:
                            if fnid in self._frontier_empty:
                                label_empty.add(fnid)
                        for enid in rn.edge_neighbors[:4]:
                            if enid in self._frontier_empty:
                                label_empty.add(enid)
                if label_empty:
                    scored = []
                    for nid in label_empty:
                        node = self._nodes.get(nid)
                        if not node or node.is_occupied:
                            continue
                        frontier_score = self._frontier_empty[nid]
                        pos = node.position
                        to_node = pos - centroid
                        tn_norm = np.linalg.norm(to_node)
                        if tn_norm > 1e-10:
                            to_node_dir = to_node / tn_norm
                        else:
                            to_node_dir = np.array([0.0, 0.0, 0.0])
                        direction_alignment = float(np.dot(to_node_dir, geo_dir))
                        geo_distance = float(np.linalg.norm(pos - ideal_pos))
                        geo_score = 1.0 / (1.0 + geo_distance * 0.5)
                        combined = frontier_score * 0.3 + direction_alignment * 0.4 + geo_score * 0.3
                        scored.append((nid, combined))
                    if scored:
                        scored.sort(key=lambda x: -x[1])
                        top = scored[0][1]
                        top_tier = [s for s in scored if s[1] >= top * 0.7]
                        chosen = random.choice(top_tier)
                        nid = chosen[0]
                        node = self._nodes.get(nid)
                        if node and not node.is_occupied:
                            return nid

            frontier_sample_size = min(100, len(self._frontier_empty))
            frontier_items = list(self._frontier_empty.items())
            sampled = random.sample(frontier_items, frontier_sample_size) if len(frontier_items) > frontier_sample_size else frontier_items
            valid = []
            for nid, fscore in sampled:
                node = self._nodes.get(nid)
                if not node or node.is_occupied:
                    continue
                pos = node.position
                geo_distance = float(np.linalg.norm(pos - ideal_pos))
                geo_score = 1.0 / (1.0 + geo_distance * 0.5)
                combined = fscore * 0.3 + geo_score * 0.7
                valid.append((nid, combined))
            if valid:
                valid.sort(key=lambda x: -x[1])
                top = valid[0][1]
                top_tier = [v for v in valid if v[1] >= top * 0.7]
                return random.choice(top_tier)[0]

        best_id = None
        best_score = float('inf')
        sample = min(300, len(self._nodes))
        for nid in random.sample(list(self._nodes.keys()), sample):
            node = self._nodes[nid]
            if node.is_occupied:
                continue
            geo_distance = float(np.linalg.norm(node.position - ideal_pos))
            centroid_dist = float(np.sum((node.position - centroid) ** 2))
            score = geo_distance * 0.5 + centroid_dist * 0.3
            if score < best_score:
                best_score = score
                best_id = nid

        return best_id or random.choice(list(self._nodes.keys()))

    def query(self, text: str, k: int = 5, labels=None) -> List[Dict]:
        with self._lock:
            qtokens = self._extract_tokens(text) if text else set()
            qtrigrams = self._extract_ngrams(text, 3) if text else set()

            pre_hit_ids = set()
            if qtokens:
                for t in qtokens:
                    if t in self._label_index:
                        pre_hit_ids.update(self._label_index[t])
                    if t in self._content_token_index:
                        pre_hit_ids.update(self._content_token_index[t])

            q_geo_dir = None
            if text:
                q_geo_dir = self._geo_mapper.map_text(text)
                qgn = np.linalg.norm(q_geo_dir)
                if qgn > 1e-12:
                    q_geo_dir = q_geo_dir / qgn
                else:
                    q_geo_dir = None
            centroid = self._centroid if self._centroid is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)

            scored = []
            checked = 0
            candidate_ids = list(pre_hit_ids) if pre_hit_ids else list(self._occupied_ids)
            if len(candidate_ids) < k:
                extra_set = set(candidate_ids)
                extra = [nid for nid in self._occupied_ids if nid not in extra_set]
                candidate_ids.extend(extra[:k * 3])

            for nid in candidate_ids:
                node = self._nodes.get(nid)
                if not node or not node.is_occupied:
                    continue
                if labels and not any(l in node.labels for l in labels):
                    continue

                text_score = 0.0
                ctokens = set()
                if qtokens:
                    ctokens = self._extract_tokens(node.content)
                    if ctokens:
                        overlap = len(qtokens & ctokens)
                        text_score = overlap / max(len(qtokens | ctokens), 1)
                        if text_score > 0:
                            content_lower = node.content.lower()
                            content_words = content_lower.split()
                            for qt in qtokens & ctokens:
                                tf = sum(1 for w in content_words if w == qt or (len(qt) >= 2 and qt in w))
                                if tf > 1:
                                    text_score += 0.05 * min(tf, 3)

                trigram_score = 0.0
                if qtrigrams:
                    ctrigrams = self._extract_ngrams(node.content, 3)
                    if ctrigrams:
                        trigram_score = len(qtrigrams & ctrigrams) / max(len(qtrigrams | ctrigrams), 1)

                label_score = 0.0
                if labels:
                    overlap = len(set(labels) & set(node.labels))
                    label_score = overlap / max(len(set(labels) | set(node.labels)), 1)

                activation_score = min(node.activation / 5.0, 1.0)
                weight_score = min(node.weight / 5.0, 1.0)

                hebbian_boost = 0.0
                if qtokens:
                    hebb_total = 0.0
                    for nnid in node.face_neighbors[:8] + node.edge_neighbors[:6]:
                        bias = self._hebbian.get_path_bias(nid, nnid)
                        if bias > 0.05:
                            hebb_total += bias
                    if hebb_total > 0:
                        hebbian_boost = min(0.20, hebb_total * 0.3)

                crystal_boost = 0.0
                if node.crystal_channels:
                    cw_sum = sum(float(v) for v in node.crystal_channels.values()) if node.crystal_channels else 0
                    if cw_sum > 0:
                        crystal_boost = min(0.15, cw_sum * 0.03)
                    else:
                        crystal_boost = min(0.10, len(node.crystal_channels) * 0.02)

                pulse_boost = 0.0
                if node.pulse_accumulator > 0.1:
                    pulse_boost = min(0.05, node.pulse_accumulator * 0.1)

                recency_boost = 0.0
                temporal_boost = 0.0
                if hasattr(node, 'creation_time') and node.creation_time > 0:
                    age = time.time() - node.creation_time
                    if age < 3600:
                        recency_boost = 0.08 * max(0, 1.0 - age / 3600.0)
                    stage = node.lifecycle_stage()
                    if stage == "fresh":
                        recency_boost += 0.03
                    elif stage == "crystallized":
                        recency_boost += 0.02
                    t_edges = self._temporal_edges.get(nid, [])
                    if t_edges:
                        max_prox = max(p for _, p in t_edges)
                        temporal_boost = min(0.06, max_prox * 0.06)

                pcnn_activity = 0.0
                if node.fired:
                    pcnn_activity = 0.04
                elif node.internal_activity > 0.5:
                    pcnn_activity = 0.02

                resonance_bonus = 0.0
                for ev in self._resonance_events:
                    if nid[:8] in ev.get("node_ids", []):
                        resonance_bonus = min(0.04, ev.get("strength", 0) * 0.05)
                        break

                spatial_quality = 0.0
                if self._reflection_field:
                    sq = self._reflection_field.get_spatial_quality(self, nid)
                    spatial_quality = sq

                geometric_quality = node.metadata.get("geometric_quality", 0.5) if node.metadata else 0.5

                cell_effective = 0.0
                node_cells = self._cell_map.get_cells_for_node(nid)
                if node_cells:
                    cell_effective = sum(c.effective_quality for c in node_cells) / len(node_cells)

                geo_topo_divergence = node.metadata.get("geo_topo_divergence", 0.0) if node.metadata else 0.0

                neighbor_density_score = 0.0
                occ_neighbors = sum(1 for fnid in node.face_neighbors[:8] if fnid in self._occupied_ids)
                if occ_neighbors > 0:
                    neighbor_density_score = min(1.0, occ_neighbors / 6.0)

                bcc_coherence = node.metadata.get("bcc_cell_coherence", 0.5) if node.metadata else 0.5

                autocorr_bonus = 0.0
                if self._spatial_autocorrelation > 0.1:
                    autocorr_bonus = self._spatial_autocorrelation * neighbor_density_score * 0.03

                dream_bonus = 0.0
                if "__dream__" in node.labels:
                    dream_bonus = 0.03

                low_priority_penalty = 0.0
                if "__low_priority__" in node.labels:
                    low_priority_penalty = 0.05

                shortcut_boost = 0.0
                if self._self_organize:
                    sc_list = self._self_organize._shortcut_by_node.get(nid, []) if hasattr(self._self_organize, '_shortcut_by_node') else []
                    if sc_list:
                        shortcut_boost = min(0.08, max(s[1] for s in sc_list) * 0.3)

                spatial_proximity = 0.0
                if q_geo_dir is not None:
                    to_node = node.position - centroid
                    tn_norm = float(np.linalg.norm(to_node))
                    if tn_norm > 1e-10:
                        to_node_dir = to_node / tn_norm
                        alignment = float(np.dot(to_node_dir, q_geo_dir))
                        if alignment > 0:
                            spatial_proximity = alignment * min(1.0, 3.0 / (tn_norm + 1.0))

                plane_priority = self._dark_plane_config.get(
                    node.dark_plane(), {}
                ).get("query_priority", 0.5)

                attention_boost = 0.0
                if self._attention_mask:
                    attn = self._attention_mask.get(nid, 0.0)
                    if attn > 0.1:
                        attention_boost = attn * 0.15

                final = (
                    0.28 * text_score
                    + 0.10 * trigram_score
                    + 0.12 * label_score
                    + 0.08 * activation_score
                    + 0.05 * weight_score
                    + 0.06 * hebbian_boost
                    + 0.06 * crystal_boost
                    + 0.02 * pulse_boost
                    + 0.05 * spatial_quality
                    + 0.04 * geometric_quality
                    + 0.03 * cell_effective
                    + 0.03 * neighbor_density_score
                    + 0.02 * geo_topo_divergence
                    + 0.02 * bcc_coherence
                    + 0.01 * autocorr_bonus
                    + 0.01 * dream_bonus
                    + 0.06 * recency_boost
                    + 0.04 * temporal_boost
                    + 0.05 * spatial_proximity
                    + 0.04 * plane_priority
                    + attention_boost
                    + shortcut_boost
                    - low_priority_penalty
                    + pcnn_activity
                    + resonance_bonus
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
                    "weight": float(node.weight),
                    "labels": list(node.labels),
                    "activation": node.activation,
                    "metadata": node.metadata,
                })

            if results:
                best_id = results[0]["id"]
                best_node = self._nodes.get(best_id)
                self._emit_pulse(best_id, strength=0.3, pulse_type=PulseType.REINFORCING)
                self._query_emergence_feedback(results)
                if best_node and self._attention_foci:
                    for r in results[:3]:
                        attn = self._attention_mask.get(r["id"], 0.0)
                        if attn > 0.1:
                            self._attention_stats["queries_in_focus"] += 1
                            break
                if self._self_regulation:
                    self._self_regulation.notify_query_result(
                        results[0].get("distance", 0), k, len(results)
                    )

            return results

    def query_spatial(
        self,
        center: Optional[list] = None,
        radius: float = 3.0,
        k: int = 20,
        labels=None,
        sort_by: str = "distance",
    ) -> List[Dict]:
        with self._lock:
            if center is not None:
                center_pos = np.array(center, dtype=np.float32)
            elif self._centroid is not None:
                center_pos = self._centroid
            else:
                center_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            radius_sq = radius * radius
            results = []
            for nid in self._occupied_ids:
                node = self._nodes.get(nid)
                if not node or not node.is_occupied:
                    continue
                if labels and not any(l in node.labels for l in labels):
                    continue
                dist_sq = float(np.sum((node.position - center_pos) ** 2))
                if dist_sq > radius_sq:
                    continue
                dist = math.sqrt(dist_sq)
                proximity = 1.0 / (1.0 + dist)
                density = sum(1 for fnid in node.face_neighbors[:8] if fnid in self._occupied_ids)
                direction = node.position - center_pos
                dn = float(np.linalg.norm(direction))
                direction_info = {}
                if dn > 1e-10:
                    d = direction / dn
                    direction_info = {
                        "dx": round(float(d[0]), 3),
                        "dy": round(float(d[1]), 3),
                        "dz": round(float(d[2]), 3),
                        "alignment_111": round(float(max(abs(d[0]), abs(d[1]), abs(d[2]))), 3),
                    }
                results.append({
                    "id": nid,
                    "content": node.content,
                    "distance": round(dist, 4),
                    "proximity_score": round(proximity, 4),
                    "weight": float(node.weight),
                    "activation": node.activation,
                    "labels": list(node.labels),
                    "metadata": node.metadata,
                    "position": node.position.tolist(),
                    "neighbor_density": density,
                    "direction": direction_info,
                })

            if sort_by == "distance":
                results.sort(key=lambda x: x["distance"])
            elif sort_by == "weight":
                results.sort(key=lambda x: -x["weight"])
            elif sort_by == "proximity":
                results.sort(key=lambda x: -x["proximity_score"])
            elif sort_by == "density":
                results.sort(key=lambda x: -x["neighbor_density"])

            return results[:k]

    def query_direction(
        self,
        direction: list,
        from_center: bool = True,
        max_angle: float = 0.5,
        k: int = 20,
        labels=None,
    ) -> List[Dict]:
        with self._lock:
            d_vec = np.array(direction, dtype=np.float32)
            dn = np.linalg.norm(d_vec)
            if dn < 1e-10:
                return []
            d_vec = d_vec / dn
            origin = self._centroid if (from_center and self._centroid is not None) else np.array([0.0, 0.0, 0.0], dtype=np.float32)

            results = []
            for nid in self._occupied_ids:
                node = self._nodes.get(nid)
                if not node or not node.is_occupied:
                    continue
                if labels and not any(l in node.labels for l in labels):
                    continue
                to_node = node.position - origin
                tn = float(np.linalg.norm(to_node))
                if tn < 1e-10:
                    continue
                to_dir = to_node / tn
                cos_angle = float(np.dot(to_dir, d_vec))
                angle = math.acos(max(-1.0, min(1.0, cos_angle)))
                if angle > max_angle:
                    continue
                alignment = (1.0 - angle / max_angle) * min(1.0, 3.0 / (tn + 1.0))
                results.append({
                    "id": nid,
                    "content": node.content,
                    "angle": round(angle, 4),
                    "alignment_score": round(alignment, 4),
                    "distance_from_center": round(tn, 4),
                    "weight": float(node.weight),
                    "labels": list(node.labels),
                    "position": node.position.tolist(),
                })

            results.sort(key=lambda x: -x["alignment_score"])
            return results[:k]

    def _query_emergence_feedback(self, results: List[Dict]):
        if len(results) < 2:
            return
        result_ids = [r["id"] for r in results]
        scores = [r["distance"] for r in results]

        path = self._find_query_path(result_ids)
        if path and len(path) >= 2:
            avg_score = sum(scores) / len(scores)
            strength = 0.3 + min(avg_score, 0.7)
            self._hebbian.record_path(path, success=True, strength=strength)
            self._query_emergence_stats["hebbian_paths_from_queries"] += 1
            self._query_emergence_stats["query_path_reinforcements"] += len(path) - 1

        for r in results:
            nid = r["id"]
            node = self._nodes.get(nid)
            if not node or not node.is_occupied:
                continue
            if "__pulse_bridge__" in node.labels:
                node.reinforce(0.15)
                self._query_emergence_stats["bridge_validations"] += 1
                self._query_emergence_stats["bridge_hits"] += 1
                neighbors = node.face_neighbors[:8] + node.edge_neighbors[:6]
                for nnid in neighbors:
                    if nnid in result_ids:
                        self._hebbian.record_path([nid, nnid], success=True, strength=0.8)

        for i in range(min(len(results), 5)):
            nid_i = results[i]["id"]
            node_i = self._nodes.get(nid_i)
            if not node_i:
                continue
            for j in range(i + 1, min(len(results), 5)):
                nid_j = results[j]["id"]
                hebbian_w = self._hebbian.get_path_bias(nid_i, nid_j)
                if hebbian_w < 0.1:
                    for fnid in node_i.face_neighbors[:8]:
                        if fnid == nid_j:
                            self._hebbian.record_path([nid_i, nid_j], success=True, strength=0.4)
                            break
                    else:
                        for enid in node_i.edge_neighbors[:6]:
                            if enid == nid_j:
                                self._hebbian.record_path([nid_i, nid_j], success=True, strength=0.25)
                                break

        self._try_resonance_triggered_dream(results)

    def _find_query_path(self, result_ids: List[str]) -> List[str]:
        if len(result_ids) < 2:
            return []
        path = [result_ids[0]]
        visited = {result_ids[0]}
        remaining = set(result_ids[1:])

        for _ in range(len(result_ids) - 1):
            current = path[-1]
            current_node = self._nodes.get(current)
            if not current_node:
                break
            best_next = None
            best_dist = float('inf')
            for rid in remaining:
                rnode = self._nodes.get(rid)
                if not rnode:
                    continue
                d = float(np.linalg.norm(current_node.position - rnode.position))
                hebbian_w = self._hebbian.get_path_bias(current, rid)
                d = d / (1.0 + hebbian_w * 2.0)
                if d < best_dist:
                    best_dist = d
                    best_next = rid
            if best_next is None:
                break

            step = [current, best_next]
            current_c = current_node
            for _ in range(8):
                best_nbor = None
                best_nbor_dist = best_dist
                target = self._nodes.get(best_next)
                if not target:
                    break
                for fnid in current_c.face_neighbors[:8]:
                    if fnid in visited or fnid not in self._nodes:
                        continue
                    fn = self._nodes[fnid]
                    d = float(np.linalg.norm(fn.position - target.position))
                    if d < best_nbor_dist:
                        best_nbor_dist = d
                        best_nbor = fnid
                for enid in current_c.edge_neighbors[:6]:
                    if enid in visited or enid not in self._nodes:
                        continue
                    en = self._nodes[enid]
                    d = float(np.linalg.norm(en.position - target.position))
                    if d < best_nbor_dist:
                        best_nbor_dist = d
                        best_nbor = enid
                if best_nbor and best_nbor not in remaining:
                    path.append(best_nbor)
                    visited.add(best_nbor)
                    step = [best_nbor]
                    current_c = self._nodes[best_nbor]
                else:
                    break

            path.append(best_next)
            visited.add(best_next)
            remaining.discard(best_next)

        return path

    def _try_resonance_triggered_dream(self, query_results: List[Dict]):
        now = time.time()
        if now - self._resonance_dream_cooldown < 120:
            return
        if not self._resonance_events:
            return
        if not query_results:
            return

        query_labels = set()
        for r in query_results[:5]:
            node = self._nodes.get(r["id"])
            if node and node.labels:
                query_labels.update(node.labels)
        if not query_labels:
            return

        relevant_resonances = []
        for ev in self._resonance_events:
            ev_labels = set(ev.get("labels", []))
            overlap = len(query_labels & ev_labels)
            if overlap > 0:
                relevant_resonances.append((overlap, ev))
        if not relevant_resonances:
            return

        relevant_resonances.sort(key=lambda x: -x[0])
        best_resonance = relevant_resonances[0][1]
        node_ids = best_resonance.get("node_ids", [])
        if len(node_ids) < 2:
            return

        source_nodes = []
        for nid_prefix in node_ids[:8]:
            for full_nid in self._occupied_ids:
                if full_nid.startswith(nid_prefix):
                    node = self._nodes.get(full_nid)
                    if node and node.is_occupied and node.weight >= 1.5:
                        source_nodes.append(full_nid)
                    break
        if len(source_nodes) < 2:
            return

        source_a = source_nodes[0]
        source_b = source_nodes[-1]
        cascade_result = self._dream_cascade_pulse(source_a, source_b)
        if cascade_result and cascade_result.get("cascade_strength", 0) > 0.15:
            resonance_nodes = cascade_result.get("resonance_nodes", {})
            if resonance_nodes:
                best_resonance_nid = max(resonance_nodes, key=resonance_nodes.get)
                rn = self._nodes.get(best_resonance_nid)
                if rn and rn.is_occupied:
                    strength = resonance_nodes[best_resonance_nid]
                    dream_content = (
                        f"[resonance-dream] {rn.content[:60]}"
                        f" | resonance_strength={strength:.3f}"
                        f" | query_labels={','.join(list(query_labels)[:4])}"
                    )
                    self.store(dream_content, labels=["__dream__", "__resonance__"],
                               weight=1.5, metadata={"triggered_by": "resonance_query_feedback"})
                    self._query_emergence_stats["resonance_dreams_triggered"] += 1
                    self._resonance_dream_cooldown = now

    def _connect_temporal(self, nid: str, weight: float):
        now = time.time()
        node = self._nodes.get(nid)
        if not node:
            return

        temporal_window = 300.0
        recent_in_window = [
            (rid, ts) for rid, ts in self._recent_stores
            if now - ts < temporal_window and rid != nid
        ]

        for rid, ts in recent_in_window:
            time_gap = now - ts
            proximity = 1.0 / (1.0 + time_gap / 60.0)
            strength = proximity * min(weight, 2.0) * 0.5

            self._temporal_edges[nid].append((rid, proximity))
            self._temporal_edges[rid].append((nid, proximity))

            self._hebbian.record_path([nid, rid], success=True, strength=strength)

            rn = self._nodes.get(rid)
            if rn and rn.is_occupied:
                shared_labels = set(node.labels) & set(rn.labels)
                shared_labels.discard("__system__")
                if shared_labels:
                    label_overlap = len(shared_labels) / max(len(set(node.labels) | set(rn.labels)), 1)
                    self._hebbian.record_path(
                        [nid, rid], success=True,
                        strength=strength * (1.0 + label_overlap),
                    )

        self._recent_stores.append((nid, now))
        if len(self._recent_stores) > self._max_recent_stores:
            cutoff = now - temporal_window * 2
            self._recent_stores = [
                (rid, ts) for rid, ts in self._recent_stores
                if ts > cutoff
            ]

    def query_temporal(
        self,
        time_range: Optional[Tuple[float, float]] = None,
        direction: str = "newest",
        k: int = 20,
        labels=None,
        min_weight: float = 0.0,
        lifecycle_stage: Optional[str] = None,
    ) -> List[Dict]:
        with self._lock:
            now = time.time()
            results = []
            for nid in self._occupied_ids:
                node = self._nodes.get(nid)
                if not node or not node.is_occupied:
                    continue
                if min_weight > 0 and node.weight < min_weight:
                    continue
                if labels and not any(l in node.labels for l in labels):
                    continue
                if lifecycle_stage and node.lifecycle_stage() != lifecycle_stage:
                    continue
                if time_range:
                    t_start, t_end = time_range
                    if not (t_start <= node.creation_time <= t_end):
                        continue
                age = now - node.creation_time
                temporal_neighbors = self._temporal_edges.get(nid, [])
                results.append({
                    "id": nid,
                    "content": node.content,
                    "weight": float(node.weight),
                    "labels": list(node.labels),
                    "creation_time": float(node.creation_time),
                    "age_seconds": round(age, 1),
                    "lifecycle_stage": node.lifecycle_stage(),
                    "activation": float(node.activation),
                    "reinforcement_count": node.reinforcement_count,
                    "temporal_connections": len(temporal_neighbors),
                })

            if direction == "newest":
                results.sort(key=lambda x: -x["creation_time"])
            elif direction == "oldest":
                results.sort(key=lambda x: x["creation_time"])
            elif direction == "age":
                results.sort(key=lambda x: x["age_seconds"])

            return results[:k]

    def query_temporal_sequence(
        self,
        source_id: str,
        direction: str = "forward",
        max_depth: int = 10,
    ) -> List[Dict]:
        with self._lock:
            visited = {source_id}
            chain = []
            current_edges = self._temporal_edges.get(source_id, [])
            edges_to_follow = sorted(current_edges, key=lambda x: -x[1])

            for eid, proximity in edges_to_follow:
                if len(chain) >= max_depth:
                    break
                if eid in visited:
                    continue
                visited.add(eid)
                node = self._nodes.get(eid)
                if not node or not node.is_occupied:
                    continue
                chain.append({
                    "id": eid,
                    "content": node.content,
                    "labels": list(node.labels),
                    "creation_time": float(node.creation_time),
                    "temporal_proximity": round(proximity, 4),
                    "lifecycle_stage": node.lifecycle_stage(),
                    "weight": float(node.weight),
                })
                next_edges = self._temporal_edges.get(eid, [])
                for neid, nprox in sorted(next_edges, key=lambda x: -x[1]):
                    if neid not in visited and len(chain) < max_depth:
                        edges_to_follow.append((neid, nprox * proximity * 0.8))

            return chain

    def get_lifecycle_stats(self) -> Dict[str, Any]:
        with self._lock:
            counts = {"fresh": 0, "consolidating": 0, "crystallized": 0, "ancient": 0}
            plane_counts = {"surface": 0, "shallow": 0, "deep": 0, "abyss": 0}
            weights = {"fresh": [], "consolidating": [], "crystallized": [], "ancient": []}
            plane_energy = {"surface": 0.0, "shallow": 0.0, "deep": 0.0, "abyss": 0.0}
            for nid in self._occupied_ids:
                node = self._nodes.get(nid)
                if not node or not node.is_occupied:
                    continue
                stage = node.lifecycle_stage()
                plane = node.dark_plane()
                counts[stage] = counts.get(stage, 0) + 1
                plane_counts[plane] = plane_counts.get(plane, 0) + 1
                weights[stage].append(float(node.weight))
                plane_energy[plane] += node.plane_energy_cost()
            avg_weights = {}
            for stage, ws in weights.items():
                avg_weights[stage] = round(sum(ws) / max(len(ws), 1), 3)
            return {
                "counts": counts,
                "plane_counts": plane_counts,
                "avg_weights": avg_weights,
                "plane_energy": {k: round(v, 2) for k, v in plane_energy.items()},
                "total_temporal_edges": sum(len(v) for v in self._temporal_edges.values()) // 2,
                "thresholds": self._lifecycle_thresholds,
                "dark_plane_transitions": self._dark_plane_transitions,
                "dark_plane_reawakenings": self._dark_plane_reawakenings,
            }

    def dark_plane_flow(self):
        with self._lock:
            now = time.time()
            transitions = 0
            reawakenings = 0

            surface_ids = []
            shallow_ids = []
            deep_ids = []
            abyss_ids = []

            for nid in list(self._occupied_ids):
                node = self._nodes.get(nid)
                if not node or not node.is_occupied:
                    continue
                plane = node.dark_plane()
                if plane == "surface":
                    surface_ids.append(nid)
                elif plane == "shallow":
                    shallow_ids.append(nid)
                elif plane == "deep":
                    deep_ids.append(nid)
                elif plane == "abyss":
                    abyss_ids.append(nid)

            if len(surface_ids) > len(self._occupied_ids) * 0.6:
                surface_ids.sort(key=lambda nid: self._nodes[nid].creation_time)
                overflow_count = len(surface_ids) - int(len(self._occupied_ids) * 0.5)
                for nid in surface_ids[:overflow_count]:
                    node = self._nodes.get(nid)
                    if node and node.lifecycle_stage() != "fresh":
                        node.base_activation *= 0.8
                        transitions += 1

            for nid in abyss_ids:
                node = self._nodes.get(nid)
                if not node:
                    continue
                if node.access_count > 3 or node.pulse_accumulator > 0.3:
                    node.activation = min(node.activation + 0.5, node.weight)
                    node.base_activation = max(0.01, node.weight * 0.1)
                    self._emit_pulse(nid, strength=0.4, pulse_type=PulseType.REINFORCING)
                    reawakenings += 1

            if len(shallow_ids) > 0 and len(deep_ids) > 0:
                for nid in shallow_ids[:5]:
                    node = self._nodes.get(nid)
                    if not node:
                        continue
                    if node.reinforcement_count >= self._lifecycle_thresholds["crystallized_reinforcement"]:
                        crystal_boost = sum(node.crystal_channels.values()) if node.crystal_channels else 0
                        if crystal_boost > 0:
                            node.base_activation = min(node.base_activation + 0.05, node.weight * 0.3)
                            transitions += 1

            self._dark_plane_transitions += transitions
            self._dark_plane_reawakenings += reawakenings

            return {
                "transitions": transitions,
                "reawakenings": reawakenings,
                "plane_distribution": {
                    "surface": len(surface_ids),
                    "shallow": len(shallow_ids),
                    "deep": len(deep_ids),
                    "abyss": len(abyss_ids),
                },
            }

    def attention_set_focus(
        self,
        center: Optional[list] = None,
        radius: float = 5.0,
        strength: float = 1.0,
        labels: Optional[List[str]] = None,
        query_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            focus_center = np.array(center, dtype=np.float32) if center is not None else None
            if focus_center is None and query_text:
                geo_dir = self._geo_mapper.map_text(query_text)
                gn = np.linalg.norm(geo_dir)
                if gn > 1e-12:
                    geo_dir = geo_dir / gn
                centroid = self._centroid if self._centroid is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
                focus_center = centroid + geo_dir * self._spacing * 3.0
            if focus_center is None:
                focus_center = self._centroid.copy() if self._centroid is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)

            focus = {
                "center": focus_center.tolist(),
                "radius": radius,
                "strength": strength,
                "labels": set(labels or []),
                "created_at": time.time(),
            }
            self._attention_foci.append(focus)
            if len(self._attention_foci) > self._attention_max_foci:
                self._attention_foci = self._attention_foci[-self._attention_max_foci:]
            self._rebuild_attention_mask()
            self._attention_stats["focus_set"] += 1
            return {
                "center": focus["center"],
                "radius": radius,
                "nodes_in_focus": sum(1 for v in self._attention_mask.values() if v > 0.1),
                "active_foci": len(self._attention_foci),
            }

    def attention_clear(self):
        with self._lock:
            self._attention_foci.clear()
            self._attention_mask.clear()

    def attention_get_mask(self, nid: str) -> float:
        return self._attention_mask.get(nid, 0.0)

    def _rebuild_attention_mask(self):
        self._attention_mask.clear()
        if not self._attention_foci:
            return
        for nid in self._occupied_ids:
            node = self._nodes.get(nid)
            if not node or not node.is_occupied:
                continue
            max_attention = 0.0
            for focus in self._attention_foci:
                fc = np.array(focus["center"], dtype=np.float32)
                dist = float(np.linalg.norm(node.position - fc))
                if dist > focus["radius"] * 2:
                    continue
                spatial_factor = max(0.0, 1.0 - dist / focus["radius"])
                label_factor = 0.0
                if focus["labels"]:
                    node_labels = set(node.labels)
                    overlap = len(focus["labels"] & node_labels)
                    if overlap > 0:
                        label_factor = overlap / max(len(focus["labels"] | node_labels), 1)
                attention = focus["strength"] * (spatial_factor * 0.6 + label_factor * 0.4)
                if not focus["labels"]:
                    attention = focus["strength"] * spatial_factor
                max_attention = max(max_attention, attention)
            if max_attention > 0.01:
                self._attention_mask[nid] = min(1.0, max_attention)

    def _attention_diffuse(self):
        if not self._attention_mask:
            return
        new_mask = dict(self._attention_mask)
        for nid, attn in list(self._attention_mask.items()):
            node = self._nodes.get(nid)
            if not node:
                continue
            spread_amount = attn * self._attention_diffusion_rate
            neighbors = node.face_neighbors[:6] + node.edge_neighbors[:4]
            n_count = max(len(neighbors), 1)
            per_neighbor = spread_amount / n_count
            for nnid in neighbors:
                if nnid in self._occupied_ids:
                    current = new_mask.get(nnid, 0.0)
                    new_mask[nnid] = min(1.0, current + per_neighbor)
            hebbian_boost = 0.0
            for nnid in node.face_neighbors[:6] + node.edge_neighbors[:4]:
                hw = self._hebbian.get_path_bias(nid, nnid)
                if hw > 0.5:
                    hebbian_boost += hw * 0.05
            if hebbian_boost > 0:
                for nnid in node.face_neighbors[:6] + node.edge_neighbors[:4]:
                    if nnid in self._occupied_ids:
                        current = new_mask.get(nnid, 0.0)
                        new_mask[nnid] = min(1.0, current + hebbian_boost * 0.1)
            temporal_edges = self._temporal_edges.get(nid, [])
            for eid, prox in temporal_edges[:3]:
                if eid in self._occupied_ids:
                    current = new_mask.get(eid, 0.0)
                    new_mask[eid] = min(1.0, current + attn * prox * 0.1)
        self._attention_mask = new_mask
        self._attention_stats["diffusions"] += 1

    def _attention_decay(self):
        if not self._attention_mask:
            return
        to_remove = []
        for nid in self._attention_mask:
            self._attention_mask[nid] *= (1.0 - self._attention_decay_rate)
            if self._attention_mask[nid] < 0.01:
                to_remove.append(nid)
        for nid in to_remove:
            del self._attention_mask[nid]
        now = time.time()
        self._attention_foci = [
            f for f in self._attention_foci
            if now - f["created_at"] < 600
        ]

    def attention_status(self) -> Dict[str, Any]:
        with self._lock:
            if not self._attention_mask:
                return {
                    "active": False,
                    "foci_count": 0,
                    "nodes_in_focus": 0,
                    "avg_attention": 0.0,
                    "stats": dict(self._attention_stats),
                }
            vals = list(self._attention_mask.values())
            return {
                "active": True,
                "foci_count": len(self._attention_foci),
                "nodes_in_focus": len(vals),
                "avg_attention": round(sum(vals) / max(len(vals), 1), 4),
                "max_attention": round(max(vals), 4),
                "foci": [
                    {"center": f["center"], "radius": f["radius"], "strength": f["strength"]}
                    for f in self._attention_foci
                ],
                "stats": dict(self._attention_stats),
            }

    def compute_emergence_quality(self) -> Dict[str, Any]:
        with self._lock:
            metrics = {}
            occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
            n_occ = len(occupied)
            if n_occ < 5:
                return {"emergence_level": "dormant", "overall_score": 0.0, "insufficient_data": True}

            # === 1. CLUSTERING STABILITY ===
            clusters_raw = []
            if self._self_organize and hasattr(self._self_organize, '_clusters'):
                clusters_raw = self._self_organize._clusters

            n_clusters = len(clusters_raw)
            cluster_sizes = [len(c.get("node_ids", [])) for c in clusters_raw] if clusters_raw else []
            cluster_label_diversity = 0.0
            if clusters_raw:
                all_cluster_labels = []
                for c in clusters_raw:
                    labels = set()
                    for nid in c.get("node_ids", [])[:20]:
                        node = self._nodes.get(nid)
                        if node:
                            labels.update(node.labels)
                    all_cluster_labels.append(labels)
                if len(all_cluster_labels) > 1:
                    pairwise_jaccard = []
                    for i in range(len(all_cluster_labels)):
                        for j in range(i + 1, len(all_cluster_labels)):
                            a, b = all_cluster_labels[i], all_cluster_labels[j]
                            if a and b:
                                jaccard = len(a & b) / max(len(a | b), 1)
                                pairwise_jaccard.append(jaccard)
                    cluster_label_diversity = 1.0 - (sum(pairwise_jaccard) / max(len(pairwise_jaccard), 1))

            nodes_in_clusters = sum(cluster_sizes)
            cluster_coverage = nodes_in_clusters / max(n_occ, 1)
            cluster_balance = 0.0
            if cluster_sizes and len(cluster_sizes) > 1:
                mean_size = sum(cluster_sizes) / len(cluster_sizes)
                variance = sum((s - mean_size) ** 2 for s in cluster_sizes) / len(cluster_sizes)
                cv = (variance ** 0.5) / max(mean_size, 1)
                cluster_balance = max(0.0, 1.0 - cv)

            morans_i = self._spatial_autocorrelation
            spatial_clustering = max(0.0, morans_i)

            recent_autocorr = self._autocorrelation_history[-10:] if self._autocorrelation_history else []
            clustering_stability = 0.0
            if len(recent_autocorr) >= 3:
                diffs = [abs(recent_autocorr[i] - recent_autocorr[i - 1]) for i in range(1, len(recent_autocorr))]
                avg_diff = sum(diffs) / len(diffs)
                clustering_stability = max(0.0, 1.0 - avg_diff * 10)

            metrics["clustering"] = {
                "n_clusters": n_clusters,
                "coverage": round(cluster_coverage, 4),
                "balance": round(cluster_balance, 4),
                "label_diversity": round(cluster_label_diversity, 4),
                "spatial_autocorrelation": round(spatial_clustering, 4),
                "stability": round(clustering_stability, 4),
                "score": round(
                    cluster_coverage * 0.3 +
                    cluster_balance * 0.2 +
                    cluster_label_diversity * 0.2 +
                    spatial_clustering * 0.15 +
                    clustering_stability * 0.15, 4
                ),
            }

            # === 2. BRIDGE EFFECTIVENESS ===
            bridge_nodes = [(nid, n) for nid, n in occupied if "__pulse_bridge__" in n.labels]
            n_bridges = len(bridge_nodes)
            bridge_rate = self._bridge_count / max(self._pulse_count, 1)

            bridge_avg_semantic = 0.0
            bridge_avg_sources = 0.0
            bridge_avg_weight = 0.0
            if bridge_nodes:
                bridge_avg_semantic = float(np.mean([
                    n.metadata.get("semantic_overlap", 0.0)
                    for _, n in bridge_nodes if n.metadata
                ]))
                bridge_avg_sources = float(np.mean([
                    n.metadata.get("bridge_sources", 0)
                    for _, n in bridge_nodes if n.metadata
                ]))
                bridge_avg_weight = float(np.mean([n.weight for _, n in bridge_nodes]))

            bridge_connectivity = 0.0
            if bridge_nodes:
                connected_clusters = set()
                for _, bn in bridge_nodes:
                    for fnid in bn.face_neighbors:
                        fn = self._nodes.get(fnid)
                        if fn and fn.is_occupied:
                            for c in clusters_raw:
                                if fnid in c.get("node_ids", []):
                                    connected_clusters.add(c.get("id", fnid[:8]))
                if n_clusters > 1:
                    bridge_connectivity = len(connected_clusters) / max(n_clusters, 1)

            recent_bridge_log = [e for e in self._pulse_log if "bridge_id" in e][-20:]
            bridge_freshness = 0.0
            if recent_bridge_log:
                now = time.time()
                avg_age = now - float(np.mean([e["time"] for e in recent_bridge_log]))
                bridge_freshness = max(0.0, 1.0 - avg_age / 3600)

            metrics["bridges"] = {
                "count": n_bridges,
                "total_created": self._bridge_count,
                "creation_rate": round(bridge_rate, 6),
                "avg_semantic_overlap": round(bridge_avg_semantic, 4),
                "avg_sources": round(bridge_avg_sources, 2),
                "avg_weight": round(bridge_avg_weight, 4),
                "cluster_connectivity": round(bridge_connectivity, 4),
                "freshness": round(bridge_freshness, 4),
                "score": round(
                    min(bridge_rate * 500, 1.0) * 0.2 +
                    bridge_avg_semantic * 0.25 +
                    min(bridge_avg_sources / 4.0, 1.0) * 0.15 +
                    bridge_connectivity * 0.25 +
                    bridge_freshness * 0.15, 4
                ),
            }

            # === 3. CRYSTAL ACTIVITY ===
            crystal_paths = 0
            crystal_avg_weight = 0.0
            if self._crystallized and hasattr(self._crystallized, '_crystals'):
                crystal_paths = len(self._crystallized._crystals)
                if crystal_paths > 0:
                    weights = []
                    for key, c in self._crystallized._crystals.items():
                        if isinstance(c, dict):
                            weights.append(c.get("weight", 1.0))
                        elif hasattr(c, 'weight'):
                            weights.append(c.weight)
                    crystal_avg_weight = float(np.mean(weights)) if weights else 0.0
            hebbian_edges = len(self._hebbian._edges) if self._hebbian else 0
            hebbian_strong = sum(1 for w in self._hebbian._edges.values() if w > 2.0) if self._hebbian else 0

            lifecycle_dist = {"fresh": 0, "consolidating": 0, "crystallized": 0, "ancient": 0}
            for _, n in occupied:
                stage = n.lifecycle_stage()
                lifecycle_dist[stage] = lifecycle_dist.get(stage, 0) + 1

            crystal_ratio = lifecycle_dist.get("crystallized", 0) / max(n_occ, 1)
            ancient_ratio = lifecycle_dist.get("ancient", 0) / max(n_occ, 1)
            mature_ratio = crystal_ratio + ancient_ratio

            avg_activation = float(np.mean([n.activation for _, n in occupied]))
            high_activation = sum(1 for _, n in occupied if n.activation > 0.5)
            activation_ratio = high_activation / max(n_occ, 1)

            metrics["crystal"] = {
                "crystallized_paths": crystal_paths,
                "crystal_avg_weight": round(crystal_avg_weight, 4),
                "hebbian_edges": hebbian_edges,
                "hebbian_strong_edges": hebbian_strong,
                "lifecycle_distribution": lifecycle_dist,
                "crystal_ratio": round(crystal_ratio, 4),
                "ancient_ratio": round(ancient_ratio, 4),
                "mature_ratio": round(mature_ratio, 4),
                "avg_activation": round(avg_activation, 4),
                "high_activation_ratio": round(activation_ratio, 4),
                "maintenance_cycles": self._crystal_maintenance_cycle,
                "score": round(
                    min(crystal_paths / max(n_occ * 0.1, 1), 1.0) * 0.2 +
                    mature_ratio * 0.25 +
                    activation_ratio * 0.25 +
                    min(hebbian_strong / max(hebbian_edges * 0.1, 1), 1.0) * 0.15 +
                    min(crystal_avg_weight / 3.0, 1.0) * 0.15, 4
                ),
            }

            # === 4. PHASE TRANSITION QUALITY ===
            phase_transitions = 0
            current_phase = self._current_phase
            field_entropy = 1.0
            energy_trend = 0.0
            if self._reflection_field:
                phase_transitions = int(self._reflection_field._phase_transitions)
                field_entropy = float(self._reflection_field._field_entropy)
                eh = self._reflection_field._energy_history
                if len(eh) >= 5:
                    recent = eh[-5:]
                    energy_trend = float(recent[-1] - recent[0]) / max(abs(recent[0]), 1e-6)

            resonance_count = len(self._resonance_events)
            resonance_avg_strength = 0.0
            if self._resonance_events:
                resonance_avg_strength = float(np.mean([e.get("strength", 0) for e in self._resonance_events]))

            transition_rate = phase_transitions / max(self._pulse_count / 1000, 1)

            entropy_score = 1.0 - abs(field_entropy - 0.5) * 2
            entropy_score = max(0.0, entropy_score)

            phase_health = 0.0
            if current_phase in ("ordered", "crystal"):
                phase_health = 0.8
            elif current_phase == "fluid":
                phase_health = 0.4 + entropy_score * 0.3
            elif current_phase == "turbulent":
                phase_health = 0.3

            metrics["phase"] = {
                "current_phase": current_phase,
                "total_transitions": phase_transitions,
                "transition_rate_per_1k_pulses": round(transition_rate, 4),
                "field_entropy": round(field_entropy, 4),
                "entropy_score": round(entropy_score, 4),
                "energy_trend": round(energy_trend, 4),
                "resonance_events": resonance_count,
                "resonance_avg_strength": round(resonance_avg_strength, 4),
                "phase_health": round(phase_health, 4),
                "score": round(
                    min(transition_rate * 2, 1.0) * 0.2 +
                    entropy_score * 0.25 +
                    phase_health * 0.25 +
                    min(resonance_count / 5, 1.0) * 0.15 +
                    min(resonance_avg_strength / 3, 1.0) * 0.15, 4
                ),
            }

            # === OVERALL EMERGENCE SCORE ===
            overall = (
                metrics["clustering"]["score"] * 0.3 +
                metrics["bridges"]["score"] * 0.25 +
                metrics["crystal"]["score"] * 0.25 +
                metrics["phase"]["score"] * 0.2
            )

            if overall >= 0.7:
                level = "thriving"
            elif overall >= 0.5:
                level = "emerging"
            elif overall >= 0.3:
                level = "developing"
            else:
                level = "dormant"

            metrics["overall_score"] = round(overall, 4)
            metrics["emergence_level"] = level
            metrics["pulse_count"] = self._pulse_count
            metrics["occupied_nodes"] = n_occ
            metrics["snapshot_time"] = time.time()

            self._emergence_history.append(metrics)
            if len(self._emergence_history) > self._emergence_max_history:
                self._emergence_history = self._emergence_history[-self._emergence_max_history // 2:]

            return metrics

    def _extract_ngrams(self, text: str, n: int = 3) -> set:
        ngrams = set()
        clean = re.sub(r'[^\w\s]', '', text.lower())
        words = clean.split()
        if len(words) >= n:
            for i in range(len(words) - n + 1):
                ng = ' '.join(words[i:i+n])
                ngrams.add(ng)
        for char_group in re.findall(r'[\u4e00-\u9fff]+', text):
            for i in range(max(1, len(char_group) - n + 1)):
                ngrams.add(char_group[i:i+n])
        return ngrams

    def _extract_tokens(self, text: str) -> set:
        tokens = set()
        for w in re.findall(r"[a-zA-Z0-9]{2,}", text.lower()):
            tokens.add(w)
        for bigram in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            for i in range(len(bigram) - 1):
                tokens.add(bigram[i:i+2])
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
        elif pulse_type == PulseType.SELF_CHECK:
            max_hops = cfg.SELF_CHECK_MAX_HOPS
            bias_fn = self._bias_self_check
        elif pulse_type == PulseType.CASCADE:
            max_hops = cfg.CASCADE_MAX_HOPS
            bias_fn = self._bias_cascade
        elif pulse_type == PulseType.STRUCTURE:
            max_hops = cfg.STRUCTURE_MAX_HOPS
            bias_fn = self._bias_structure
        elif pulse_type == PulseType.DEEP_REINFORCEMENT:
            max_hops = cfg.DEEP_REINFORCEMENT_MAX_HOPS
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

        if pulse_type == PulseType.CASCADE:
            self._propagate_cascade(pulse)
        else:
            self._propagate_pulse(pulse)
        self._pulse_type_counts[pulse_type] = self._pulse_type_counts.get(pulse_type, 0) + 1

    def _bias_exploratory(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
        biased = []
        for nid, strength, ctype in candidates:
            h = min(self._hebbian.get_path_bias(src, nid), 3.0)
            biased.append((nid, strength * random.uniform(0.8, 1.2) * (1.0 + h * 0.5), ctype))
        return biased

    def _bias_reinforcing(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
        biased = []
        for nid, strength, ctype in candidates:
            node = self._nodes.get(nid)
            weight_boost = min((node.weight * 0.5 + 1.0) if node else 1.0, 5.0)
            hebbian_w = min(self._hebbian.get_path_bias(src, nid), 3.0)
            hebbian_boost = 1.0 + hebbian_w * 2.0
            quality_factor = 1.0
            if node and node.metadata:
                gq = node.metadata.get("geometric_quality", 0.5)
                quality_factor = 0.8 + 0.4 * gq
            biased.append((nid, strength * weight_boost * hebbian_boost * quality_factor, ctype))
        return biased

    def _bias_tension_sensing(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
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
            h = min(self._hebbian.get_path_bias(src, nid), 3.0)
            tension_factor *= (1.0 + h * 0.3)
            biased.append((nid, strength * tension_factor, ctype))
        return biased

    def _bias_self_check(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
        biased = []
        for nid, strength, ctype in candidates:
            node = self._nodes.get(nid)
            if node is None:
                biased.append((nid, strength, ctype))
                continue
            h = min(self._hebbian.get_path_bias(src, nid), 3.0)
            h_factor = 1.0 + h * 0.3
            if not node.is_occupied:
                biased.append((nid, strength * 1.5 * h_factor, ctype))
            else:
                occupied_neighbors = 0
                for fnid in node.face_neighbors[:8]:
                    fn = self._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        occupied_neighbors += 1
                if occupied_neighbors == 0:
                    biased.append((nid, strength * 2.0 * h_factor, ctype))
                else:
                    biased.append((nid, strength * 0.5 * h_factor, ctype))
        return biased

    def _bias_cascade(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
        biased = []
        for nid, strength, ctype in candidates:
            node = self._nodes.get(nid)
            if node is None:
                biased.append((nid, strength, ctype))
                continue
            crystal_boost = 1.0
            source_node = self._nodes.get(src)
            if source_node:
                crystal_boost = self._crystallized.get_boost(source_node.id, nid)
            weight_factor = 1.0 + (node.weight * 0.3 if node.is_occupied else 0.1)
            h = min(self._hebbian.get_path_bias(src, nid), 3.0)
            biased.append((nid, strength * weight_factor * crystal_boost * (1.0 + h * 0.5), ctype))
        return biased

    def _bias_structure(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        src = getattr(self, '_propagation_source', '')
        biased = []
        for nid, strength, ctype in candidates:
            node = self._nodes.get(nid)
            if node is None:
                biased.append((nid, strength * 0.5, ctype))
                continue
            total_neighbors = len(node.face_neighbors) + len(node.edge_neighbors)
            h = min(self._hebbian.get_path_bias(src, nid), 3.0)
            h_factor = 1.0 + h * 0.3
            if total_neighbors < 6:
                biased.append((nid, strength * 2.5 * h_factor, ctype))
            elif total_neighbors < 10:
                biased.append((nid, strength * 1.5 * h_factor, ctype))
            else:
                biased.append((nid, strength * 0.8 * h_factor, ctype))
        return biased

    def _cell_quality_factor(self, nid: str) -> float:
        cells = self._cell_map.get_cells_for_node(nid)
        if not cells:
            return 1.0
        avg_quality = sum(c.quality for c in cells) / len(cells)
        avg_eff = sum(c.effective_quality for c in cells) / len(cells)
        return 0.7 + 0.3 * (0.6 * avg_quality + 0.4 * avg_eff)

    def _propagate_pulse(self, pulse: NeuralPulse):
        queue = deque([pulse])
        while queue:
            cur = queue.popleft()
            if not cur.alive:
                continue

            current_id = cur.path[-1]
            current = self._nodes.get(current_id)
            if current is None:
                continue

            receptivity = current.plane_pulse_receptivity() if current.is_occupied else 1.0
            effective_strength = cur.strength * receptivity
            current.pulse_accumulator = min(PCNNConfig.MAX_PULSE_ACCUMULATOR, current.pulse_accumulator + effective_strength)
            current.last_pulse_time = time.time()

            if current.is_occupied:
                current.reinforce(effective_strength * 0.01)
                if cur.pulse_type == PulseType.SELF_CHECK:
                    self._detect_empty_associations(current_id, current)
                if cur.pulse_type == PulseType.REINFORCING and len(cur.path) >= 2 and self._pulse_engine is not None and self._pulse_engine.is_alive():
                    self._hebbian.record_path(cur.path[-3:] if len(cur.path) >= 3 else cur.path, success=True, strength=cur.strength * 0.6)

            cfg = PCNNConfig
            raw_candidates = []
            cell_q = self._cell_quality_factor(current_id)
            for nid in current.face_neighbors:
                if nid in cur.path_set:
                    continue
                base_strength = cur.strength * cfg.FACE_DECAY * cell_q
                crystal_boost = self._crystallized.get_boost(current_id, nid)
                spatial_bias = self._reflection_field.get_pulse_direction_bias(self, current_id, nid) if self._reflection_field else 1.0
                bcc_dir = self._bcc_direction_factor(current.position, self._nodes[nid].position) if nid in self._nodes else 1.0
                raw_candidates.append((nid, base_strength * crystal_boost * spatial_bias * bcc_dir, "face"))
            for nid in current.edge_neighbors:
                if nid in cur.path_set:
                    continue
                base_strength = cur.strength * cfg.FACE_DECAY * cfg.EDGE_DECAY_FACTOR * cell_q
                crystal_boost = self._crystallized.get_boost(current_id, nid)
                spatial_bias = self._reflection_field.get_pulse_direction_bias(self, current_id, nid) if self._reflection_field else 1.0
                bcc_dir = self._bcc_direction_factor(current.position, self._nodes[nid].position) if nid in self._nodes else 1.0
                raw_candidates.append((nid, base_strength * crystal_boost * spatial_bias * bcc_dir, "edge"))
            for nid in current.vertex_neighbors:
                if nid in cur.path_set:
                    continue
                nn = self._nodes.get(nid)
                if nn and nn.is_occupied:
                    base_strength = cur.strength * cfg.FACE_DECAY * cfg.EDGE_DECAY_FACTOR * 0.3
                    raw_candidates.append((nid, base_strength, "vertex"))

            if not raw_candidates:
                continue

            self._propagation_source = current_id
            if cur.bias_fn is not None:
                biased = cur.bias_fn(raw_candidates)
            else:
                biased = raw_candidates

            total_w = sum(w for _, w, _ in biased)
            if total_w <= 0:
                continue

            weights = [w for _, w, _ in biased]
            if not all(math.isfinite(w) and w > 0 for w in weights):
                continue

            fanout = 2 if cur.pulse_type == PulseType.EXPLORATORY and len(biased) >= 2 else 1
            chosen_indices = random.choices(range(len(biased)), weights=weights, k=min(fanout, len(biased)))
            for idx in chosen_indices:
                next_id, next_strength, direction = biased[idx]

                new_pulse = NeuralPulse(
                    cur.source_id, next_strength, cur.max_hops, cur.pulse_type, cur.bias_fn
                )
                new_pulse.hops = cur.hops + 1
                new_pulse.path = cur.path + [next_id]
                new_pulse.path_set = cur.path_set | {next_id}
                new_pulse.direction = direction
                queue.append(new_pulse)

    def _propagate_cascade(self, pulse: NeuralPulse):
        queue = deque([pulse])
        while queue:
            cur = queue.popleft()
            if not cur.alive:
                continue
            if cur.cascade_depth >= PCNNConfig.CASCADE_MAX_DEPTH:
                self._propagate_pulse(cur)
                continue

            current_id = cur.path[-1]
            current = self._nodes.get(current_id)
            if current is None:
                continue

            current.pulse_accumulator = min(PCNNConfig.MAX_PULSE_ACCUMULATOR, current.pulse_accumulator + cur.strength)
            current.last_pulse_time = time.time()
            if current.is_occupied:
                current.reinforce(cur.strength * 0.02)

            cfg = PCNNConfig
            raw_candidates = []
            for nid in current.face_neighbors:
                if nid in cur.path_set:
                    continue
                base_strength = cur.strength * cfg.FACE_DECAY
                crystal_boost = self._crystallized.get_boost(current_id, nid)
                nn = self._nodes.get(nid)
                spatial_bias = 1.0
                bcc_dir = 1.0
                if nn and self._reflection_field:
                    spatial_bias = self._reflection_field.get_pulse_direction_bias(self, current_id, nid)
                if nn:
                    bcc_dir = self._bcc_direction_factor(current.position, nn.position)
                raw_candidates.append((nid, base_strength * crystal_boost * spatial_bias * bcc_dir, "face"))
            for nid in current.edge_neighbors:
                if nid in cur.path_set:
                    continue
                base_strength = cur.strength * cfg.FACE_DECAY * cfg.EDGE_DECAY_FACTOR
                crystal_boost = self._crystallized.get_boost(current_id, nid)
                nn = self._nodes.get(nid)
                spatial_bias = 1.0
                bcc_dir = 1.0
                if nn and self._reflection_field:
                    spatial_bias = self._reflection_field.get_pulse_direction_bias(self, current_id, nid)
                if nn:
                    bcc_dir = self._bcc_direction_factor(current.position, nn.position)
                raw_candidates.append((nid, base_strength * crystal_boost * spatial_bias * bcc_dir, "edge"))
            for nid in current.vertex_neighbors[:4]:
                if nid in cur.path_set:
                    continue
                nn = self._nodes.get(nid)
                if nn and nn.is_occupied:
                    base_strength = cur.strength * cfg.FACE_DECAY * cfg.EDGE_DECAY_FACTOR * 0.3
                    crystal_boost = self._crystallized.get_boost(current_id, nid)
                    raw_candidates.append((nid, base_strength * crystal_boost, "vertex"))

            if not raw_candidates:
                continue

            if cur.bias_fn is not None:
                biased = cur.bias_fn(raw_candidates)
            else:
                biased = raw_candidates

            total_energy = sum(w for _, w, _ in biased)
            if total_energy <= 0:
                continue

            k = min(cfg.CASCADE_BRANCHING_FACTOR, len(biased))
            weights = [w for _, w, _ in biased]
            if not all(math.isfinite(w) and w > 0 for w in weights):
                continue
            selected_indices = random.choices(range(len(biased)), weights=weights, k=k)

            child_energy_budget = cur.strength * cfg.CASCADE_ENERGY_CONSERVATION * cfg.CASCADE_BRANCHING_DECAY
            energy_per_child = child_energy_budget / k

            for idx in selected_indices:
                next_id, next_strength, direction = biased[idx]
                child = cur.clone(energy_per_child)
                child.hops = cur.hops + 1
                child.path = cur.path[-8:] + [next_id]
                child.path_set = cur.path_set | {next_id}
                child.direction = direction
                self._cascade_count += 1

                queue.append(child)

            self._hebbian.record_path(cur.path[-3:] if len(cur.path) >= 3 else cur.path, success=True, strength=cur.strength * 0.8)

    def _adaptive_pulse_throttle(self):
        occupied = sum(1 for n in self._nodes.values() if n.is_occupied)
        total = len(self._nodes)
        if total == 0:
            return self._adaptive_interval, PCNNConfig.PULSE_BUDGET_PER_CYCLE
        occupancy = occupied / total
        pulse_budget = PCNNConfig.PULSE_BUDGET_PER_CYCLE
        if occupancy > 0.8:
            interval = PCNNConfig.ADAPTIVE_MAX_INTERVAL
            pulse_budget = max(10, pulse_budget // 2)
        elif occupancy > 0.6:
            interval = PCNNConfig.BASE_PULSE_INTERVAL
            pulse_budget = max(20, int(pulse_budget * 0.75))
        elif occupancy < 0.2:
            interval = PCNNConfig.ADAPTIVE_MIN_INTERVAL
            pulse_budget = int(pulse_budget * 1.5)
        else:
            interval = self._adaptive_interval
        self._adaptive_interval = max(
            PCNNConfig.ADAPTIVE_MIN_INTERVAL,
            min(PCNNConfig.ADAPTIVE_MAX_INTERVAL, interval),
        )
        return self._adaptive_interval, pulse_budget

    def start_pulse_engine(self):
        if self._pulse_engine is not None and self._pulse_engine.is_alive():
            return
        self._stop_event.clear()
        self._self_check = SelfCheckEngine(self)
        self._self_check.start()
        self._lattice_checker = LatticeIntegrityChecker(self)
        self._self_organize = SelfOrganizeEngine(self)
        self._dream_engine = DreamEngine(self)
        self._agent_driver = AgentMemoryDriver(self)
        self._feedback_loop = FeedbackLoop(self)
        self._session_manager = SessionManager(self)
        self._reflection_field = SpatialReflectionField()
        self._phase_transition = None
        self._pulse_engine = threading.Thread(target=self._pulse_loop, name="neural-pulse", daemon=True)
        self._pulse_engine.start()
        logger.info(
            "PCNN pulse engine started (v6.2) — face_decay=%.2f, edge_decay=%.2f, cascade=on, dream=on, reflection=on, vertex=on, autocorr=on",
            PCNNConfig.FACE_DECAY, PCNNConfig.FACE_DECAY * PCNNConfig.EDGE_DECAY_FACTOR,
        )

    def stop_pulse_engine(self):
        if self._self_check:
            self._self_check.stop()
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
                    with self._lock:
                        self._global_decay()
                        self._hebbian.decay_all()

                if cycle % 30 == 0:
                    self._pcnn_global_step()
                    self._update_adaptive_interval()

                if cycle % 60 == 0:
                    self._crystal_maintenance()

                if cycle % 600 == 0 and self._lattice_checker:
                    self._lattice_checker.run_full_check()

                if cycle % PCNNConfig.SELF_ORGANIZE_INTERVAL == 0 and self._self_organize:
                    self._self_organize.run_cycle()

                if cycle % PCNNConfig.DREAM_CYCLE_INTERVAL == 0 and self._dream_engine:
                    self._dream_engine.run_dream_cycle()

                if cycle % 120 == 0:
                    with self._lock:
                        self._cell_map.update_all_densities(self._nodes)

                if cycle % 150 == 0 and self._reflection_field:
                    self._reflection_field.run_reflection_cycle(self)
                    self._apply_phase_behavior()

                if cycle % 200 == 0:
                    self.dark_plane_flow()

                if cycle % 50 == 0 and self._attention_mask:
                    self._attention_diffuse()
                    self._attention_decay()
                    self._rebuild_attention_mask()

                if cycle % 300 == 0:
                    self.compute_spatial_autocorrelation()
                    self._detect_resonance()
                    if self._phase_transition and hasattr(self, '_phase_transition'):
                        try:
                            with self._lock:
                                gt, tensions = self._phase_transition.compute_global_tension(self)
                                if self._phase_transition.should_trigger(gt):
                                    clusters = self._phase_transition.identify_tension_clusters(tensions, self)
                                    if clusters:
                                        self._phase_transition.execute_transition(self, tensions, clusters)
                                        logger.info("Phase transition triggered: global_tension=%.1f, clusters=%d", gt, len(clusters))
                        except Exception as e:
                            logger.error("Phase transition error: %s", e)

                if cycle % 500 == 0 and cycle > 0:
                    try:
                        self.compute_emergence_quality()
                        if self._self_regulation:
                            self._self_regulation.regulate()
                    except Exception as e:
                        logger.error("Emergence/regulation error: %s", e)

            except Exception as e:
                logger.error("Pulse cycle error: %s", e, exc_info=True)

    def _pulse_cycle(self):
        with self._lock:
            occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
            if not occupied:
                return

            pulse_type = self._select_pulse_type()
            cfg = PCNNConfig

            plane_targets = {
                PulseType.EXPLORATORY: ("surface", "shallow"),
                PulseType.REINFORCING: ("shallow", "deep"),
                PulseType.CASCADE: ("surface", "shallow"),
                PulseType.SELF_CHECK: ("abyss", "deep"),
                PulseType.STRUCTURE: ("deep", "shallow"),
                PulseType.DEEP_REINFORCEMENT: ("deep", "abyss"),
            }
            preferred_planes = plane_targets.get(pulse_type, ("surface",))

            plane_weighted = []
            for nid, n in occupied:
                plane = n.dark_plane()
                pw = self._dark_plane_config.get(plane, {}).get("pulse_weight", 1.0)
                if plane in preferred_planes:
                    pw *= 2.0
                attn = self._attention_mask.get(nid, 0.0)
                if attn > 0.1:
                    pw *= (1.0 + attn * 2.0)
                    self._attention_stats["pulses_focused"] += 1
                elif self._attention_mask and attn < 0.01:
                    pw *= 0.3
                plane_weighted.append((nid, n, pw))

            total_pw = sum(pw for _, _, pw in plane_weighted)
            if total_pw > 0 and random.random() < 0.7:
                r = random.uniform(0, total_pw)
                cumul = 0
                chosen_nid = plane_weighted[0][0]
                for nid, n, pw in plane_weighted:
                    cumul += pw
                    if cumul >= r:
                        chosen_nid = nid
                        break
                nid = chosen_nid
            else:
                if pulse_type == PulseType.EXPLORATORY:
                    nid = self._select_source_exploratory(occupied)
                elif pulse_type == PulseType.REINFORCING:
                    nid = self._select_source_reinforcing(occupied)
                elif pulse_type == PulseType.SELF_CHECK:
                    nid = self._select_source_self_check(occupied)
                elif pulse_type == PulseType.CASCADE:
                    nid = self._select_source_cascade(occupied)
                elif pulse_type == PulseType.STRUCTURE:
                    nid = self._select_source_structure(occupied)
                elif pulse_type == PulseType.DEEP_REINFORCEMENT:
                    nid = self._select_source_deep_reinforcement(occupied)
                else:
                    nid = self._select_source_tension(occupied)

            if pulse_type == PulseType.EXPLORATORY:
                lo, hi = cfg.EXPLORATORY_STRENGTH_RANGE
                strength = random.uniform(lo, hi)
            elif pulse_type == PulseType.REINFORCING:
                lo, hi = cfg.REINFORCING_STRENGTH_RANGE
                strength = random.uniform(lo, hi)
            elif pulse_type == PulseType.SELF_CHECK:
                strength = cfg.SELF_CHECK_STRENGTH
            elif pulse_type == PulseType.CASCADE:
                lo, hi = cfg.CASCADE_STRENGTH_RANGE
                strength = random.uniform(lo, hi)
            elif pulse_type == PulseType.STRUCTURE:
                strength = cfg.STRUCTURE_STRENGTH
            elif pulse_type == PulseType.DEEP_REINFORCEMENT:
                lo, hi = cfg.DEEP_REINFORCEMENT_STRENGTH_RANGE
                strength = random.uniform(lo, hi)
            else:
                lo, hi = cfg.TENSION_STRENGTH_RANGE
                strength = random.uniform(lo, hi)

            src_node = self._nodes.get(nid)
            if src_node:
                receptivity = src_node.plane_pulse_receptivity()
                strength *= receptivity

            self._emit_pulse(nid, strength, pulse_type)
            self._pulse_count += 1

    def _select_pulse_type(self) -> PulseType:
        cfg = PCNNConfig
        types = list(cfg.PULSE_TYPE_PROBABILITIES.keys())
        weights = list(cfg.PULSE_TYPE_PROBABILITIES.values())
        phase = self._current_phase
        if phase == "turbulent":
            for i, t in enumerate(types):
                if t == PulseType.EXPLORATORY:
                    weights[i] *= 1.5
                elif t == PulseType.REINFORCING:
                    weights[i] *= 0.7
        elif phase == "crystalline":
            for i, t in enumerate(types):
                if t == PulseType.STRUCTURE:
                    weights[i] *= 1.5
                elif t == PulseType.EXPLORATORY:
                    weights[i] *= 0.7
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
        filtered = [(nid, n) for nid, n in occupied if "__low_priority__" not in n.labels]
        if not filtered:
            filtered = occupied
        weighted = [(nid, n.activation * max(n.weight, 0.5)) for nid, n in filtered]
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

    def _select_source_self_check(self, occupied: List[Tuple[str, Any]]) -> str:
        isolation_scores = []
        for nid, node in occupied:
            occupied_neighbors = 0
            for fnid in node.face_neighbors[:8]:
                fn = self._nodes.get(fnid)
                if fn and fn.is_occupied:
                    occupied_neighbors += 1
            for enid in node.edge_neighbors[:4]:
                en = self._nodes.get(enid)
                if en and en.is_occupied:
                    occupied_neighbors += 1
            isolation = 1.0 / (occupied_neighbors + 1)
            vitality_penalty = 0.0
            if node.weight >= 2.0 and node.activation < 0.1:
                vitality_penalty = 2.0
            isolation_scores.append((nid, isolation + vitality_penalty + 0.01))

        total = sum(s for _, s in isolation_scores)
        if total <= 0:
            return random.choice(occupied)[0]
        return random.choices(
            [i for i, _ in isolation_scores], weights=[s for _, s in isolation_scores], k=1
        )[0]

    def _select_source_cascade(self, occupied: List[Tuple[str, Any]]) -> str:
        filtered = [(nid, n) for nid, n in occupied if "__low_priority__" not in n.labels]
        if not filtered:
            filtered = occupied
        dream_sources = [(nid, n) for nid, n in filtered if "__dream__" in n.labels and n.weight >= 2.0]
        if dream_sources and random.random() < 0.3:
            return random.choice(dream_sources)[0]
        hebbian_nodes = set()
        for (a, b), w in self._hebbian._edges.items():
            if w > 1.0:
                hebbian_nodes.add(a)
                hebbian_nodes.add(b)

        hebbian_occupied = [(nid, n) for nid, n in filtered if nid in hebbian_nodes]
        if hebbian_occupied:
            weighted = [(nid, n.activation * max(n.weight, 0.5)) for nid, n in hebbian_occupied]
            total = sum(w for _, w in weighted)
            if total > 0:
                return random.choices(
                    [i for i, _ in weighted], weights=[w for _, w in weighted], k=1
                )[0]

        high_weight = [(nid, n.weight * n.activation) for nid, n in occupied if n.weight >= 2.0]
        if high_weight:
            total = sum(w for _, w in high_weight)
            if total > 0:
                return random.choices(
                    [i for i, _ in high_weight], weights=[w for _, w in high_weight], k=1
                )[0]

        return self._select_source_reinforcing(occupied)

    def _select_source_deep_reinforcement(self, occupied: List[Tuple[str, Any]]) -> str:
        crystal_nodes = [(nid, n) for nid, n in occupied if n.crystal_channels]
        if crystal_nodes:
            weighted = [(nid, n.weight * len(n.crystal_channels) * n.activation) for nid, n in crystal_nodes]
            total = sum(w for _, w in weighted)
            if total > 0:
                return random.choices(
                    [i for i, _ in weighted], weights=[w for _, w in weighted], k=1
                )[0]
        high_weight = [(nid, n) for nid, n in occupied if n.weight >= 3.0]
        if high_weight:
            return random.choice(high_weight)[0]
        return self._select_source_reinforcing(occupied)

    def _select_source_structure(self, occupied: List[Tuple[str, Any]]) -> str:
        structure_scores = []
        for nid, node in occupied:
            total_neighbors = len(node.face_neighbors) + len(node.edge_neighbors)
            neighbor_weight_var = 0.0
            nb_weights = []
            for fnid in node.face_neighbors[:8]:
                fn = self._nodes.get(fnid)
                if fn and fn.is_occupied:
                    nb_weights.append(fn.weight)
            if nb_weights:
                neighbor_weight_var = float(np.var(nb_weights))
            crystal_count = len(node.crystal_channels)
            structure_score = neighbor_weight_var + crystal_count * 0.5 + (1.0 if total_neighbors < 10 else 0.0)
            structure_scores.append((nid, structure_score + 0.01))

        total = sum(s for _, s in structure_scores)
        if total <= 0:
            return random.choice(occupied)[0]
        return random.choices(
            [i for i, _ in structure_scores], weights=[s for _, s in structure_scores], k=1
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
                for vnid in node.vertex_neighbors[:2]:
                    vn = self._nodes.get(vnid)
                    if vn:
                        out = 1.0 if vn.fired else 0.0
                        if out > 0:
                            neighbor_outputs.append((vnid, out * vn.activation * 0.02, "vertex"))

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

    def _apply_phase_behavior(self):
        if not self._reflection_field:
            return
        phase = self._reflection_field._phase_state
        with self._lock:
            if phase == "crystalline":
                self._adaptive_interval = min(
                    PCNNConfig.MAX_PULSE_INTERVAL,
                    self._adaptive_interval * 1.1
                )
            elif phase == "turbulent":
                self._adaptive_interval = max(
                    PCNNConfig.MIN_PULSE_INTERVAL,
                    self._adaptive_interval * 0.85
                )
            self._current_phase = phase

    def _crystal_maintenance(self):
        with self._lock:
            new_crystals = self._crystallized.scan_and_crystallize(self._hebbian._edges)

            self._crystal_maintenance_cycle += 1
            if self._crystal_maintenance_cycle % 5 == 0:
                occupied_set = {nid for nid, n in self._nodes.items() if n.is_occupied}
                dead = self._crystallized.health_check(occupied_set)
                if dead > 0:
                    logger.info("Crystal health check: removed %d dead crystals", dead)

            for node in self._nodes.values():
                node.crystal_channels.clear()

            for key, crystal in self._crystallized._crystals.items():
                a, b = crystal["nodes"]
                node_a = self._nodes.get(a)
                node_b = self._nodes.get(b)
                if node_a:
                    node_a.crystal_channels[b] = crystal["crystal_weight"]
                if node_b:
                    node_b.crystal_channels[a] = crystal["crystal_weight"]

            if new_crystals > 0:
                logger.info("Crystal maintenance: %d new crystallized pathways", new_crystals)

    def _detect_resonance(self):
        with self._lock:
            occupied = [(nid, n) for nid, n in self._nodes.items()
                         if n.is_occupied and n.fired]
            if len(occupied) < 3:
                self._resonance_events = []
                return

            directions = {
                (1,1,1): [], (1,1,-1): [], (1,-1,1): [], (1,-1,-1): [],
                (-1,1,1): [], (-1,1,-1): [], (-1,-1,1): [], (-1,-1,-1): [],
            }
            for nid, node in occupied:
                pos = node.position
                for dkey in directions:
                    proj = float(np.dot(pos, np.array(dkey, dtype=np.float32)))
                    directions[dkey].append((nid, proj))

            resonance_events = []
            for dkey, nodes_proj in directions.items():
                if len(nodes_proj) < 2:
                    continue
                nodes_proj.sort(key=lambda x: x[1])
                proj_vals = [p for _, p in nodes_proj]
                proj_range = max(proj_vals) - min(proj_vals)
                if proj_range <= 0:
                    continue
                n_bins = max(10, min(int(proj_range / self._spacing), 100))
                bin_width = proj_range / n_bins
                bins = [0] * n_bins
                bin_nodes = [[] for _ in range(n_bins)]
                for nid, proj in nodes_proj:
                    bi = min(int((proj - proj_vals[0]) / bin_width), n_bins - 1)
                    bins[bi] += 1
                    bin_nodes[bi].append(nid)
                if max(bins) == 0:
                    continue
                avg_density = len(nodes_proj) / n_bins
                if avg_density < 1:
                    continue
                threshold_density = avg_density * 2.0
                dense_indices = [i for i, c in enumerate(bins) if c >= threshold_density]
                if not dense_indices:
                    continue
                groups = []
                current_group = [dense_indices[0]]
                for i in range(1, len(dense_indices)):
                    if dense_indices[i] - dense_indices[i-1] <= 1:
                        current_group.append(dense_indices[i])
                    else:
                        groups.append(current_group)
                        current_group = [dense_indices[i]]
                groups.append(current_group)
                clusters = []
                for group in groups:
                    cluster_nodes = []
                    for bi in group:
                        cluster_nodes.extend(bin_nodes[bi])
                    if len(cluster_nodes) >= 3:
                        clusters.append(cluster_nodes)

                for cluster in clusters:
                    ids = cluster
                    avg_weight = np.mean([self._nodes[nid].weight for nid in ids])
                    resonance_strength = len(cluster) * avg_weight / 10.0
                    if resonance_strength > 0.5:
                        resonance_events.append({
                            "direction": list(dkey),
                            "node_count": len(cluster),
                            "strength": round(float(resonance_strength), 3),
                            "node_ids": [nid[:8] for nid in ids[:5]],
                        })
                        for i in range(len(ids) - 1):
                            self._hebbian.record_path(
                                [ids[i], ids[i+1]],
                                success=True,
                                strength=resonance_strength * 0.3,
                            )

            self._resonance_events = resonance_events[-10:]

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
                    source_nodes = []
                    source_labels = set()
                    for sid in list(sources)[:4]:
                        sn = self._nodes.get(sid)
                        if sn and sn.is_occupied:
                            source_nodes.append((sid, sn))
                            source_labels.update(sn.labels)
                    source_labels.discard("__dream__")
                    source_labels.discard("__system__")
                    source_labels.discard("__pulse_bridge__")

                    non_system_labels = [l for l in source_labels if not l.startswith("__")]
                    if len(non_system_labels) < 1:
                        continue

                    src_token_sets = [self._extract_tokens(sn.content) for _, sn in source_nodes]
                    pairwise_overlap = 0
                    pair_count = 0
                    for ti in range(len(src_token_sets)):
                        for tj in range(ti + 1, len(src_token_sets)):
                            if src_token_sets[ti] and src_token_sets[tj]:
                                inter = len(src_token_sets[ti] & src_token_sets[tj])
                                union = len(src_token_sets[ti] | src_token_sets[tj])
                                if union > 0:
                                    pairwise_overlap += inter / union
                                pair_count += 1
                    avg_semantic = pairwise_overlap / max(pair_count, 1)

                    bridge_weight = node.pulse_accumulator * (0.5 + 0.5 * avg_semantic)
                    if bridge_weight < 0.3 and avg_semantic < 0.15:
                        continue

                    label_str = ", ".join(non_system_labels[:4])
                    summaries = []
                    for _, sn in source_nodes:
                        clean = sn.content.lstrip("[").split("] ", 1)
                        text = clean[-1] if len(clean) > 1 else clean[0]
                        summaries.append(text.strip()[:50])
                    bridge = f"[bridge] {label_str}: {' ; '.join(summaries[:3])}"

                    node.content = bridge
                    node.labels = non_system_labels[:6] + ["__pulse_bridge__"]
                    node.weight = min(max(bridge_weight, 0.3), 2.0)
                    node.activation = bridge_weight * 0.5
                    node.base_activation = 0.05
                    node.metadata = {
                        "bridge_sources": len(source_nodes),
                        "semantic_overlap": round(avg_semantic, 3),
                        "pulse_accumulated": round(node.pulse_accumulator, 3),
                    }
                    chash = hashlib.sha256(bridge.encode()).hexdigest()[:12]
                    self._content_hash_index[chash] = nid
                    for lbl in node.labels:
                        self._label_index[lbl].add(nid)
                    for tok in self._extract_tokens(bridge):
                        self._content_token_index[tok].add(nid)
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
        with self._lock:
            node = self._nodes.get(nid)
            if node is None:
                return None
            return {
                "id": node.id,
                "content": node.content,
                "position": node.position.tolist(),
                "centroid": node.position.tolist(),
                "labels": list(node.labels),
                "weight": float(node.weight),
                "activation": float(node.activation),
                "face_neighbors": len(node.face_neighbors),
                "edge_neighbors": len(node.edge_neighbors),
                "creation_time": float(node.creation_time),
                "access_count": int(node.access_count),
                "metadata": {k: float(v) if isinstance(v, (np.floating, np.integer)) else bool(v) if isinstance(v, np.bool_) else v for k, v in node.metadata.items()},
                "feeding": float(node.feeding),
                "linking": float(node.linking),
                "internal_activity": float(node.internal_activity),
                "threshold": float(node.threshold),
                "fired": bool(node.fired),
                "crystal_channels": {k[:8]: round(float(v), 3) for k, v in node.crystal_channels.items()},
            }

    def list_occupied(self) -> List[Dict]:
        with self._rw_lock.read_locked():
            results = []
            for nid, node in self._nodes.items():
                if node.is_occupied:
                    results.append({
                        "id": node.id,
                        "content": node.content,
                        "position": node.position.tolist(),
                        "centroid": node.position.tolist(),
                        "labels": list(node.labels),
                        "weight": float(node.weight),
                        "activation": float(node.activation),
                        "face_neighbors": len(node.face_neighbors),
                        "edge_neighbors": len(node.edge_neighbors),
                        "creation_time": float(node.creation_time),
                        "access_count": int(node.access_count),
                        "metadata": {k: float(v) if isinstance(v, (np.floating, np.integer)) else bool(v) if isinstance(v, np.bool_) else v for k, v in node.metadata.items()},
                        "feeding": float(node.feeding),
                        "linking": float(node.linking),
                        "internal_activity": float(node.internal_activity),
                        "threshold": float(node.threshold),
                        "fired": bool(node.fired),
                        "crystal_channels": {k[:8]: round(float(v), 3) for k, v in node.crystal_channels.items()},
                    })
            return results

    def topology_graph(self) -> Dict:
        with self._rw_lock.read_locked():
            nodes = []
            for nid, node in self._nodes.items():
                if node.is_occupied or node.pulse_accumulator > 0.1:
                    nodes.append({
                        "id": nid,
                        "centroid": node.position.tolist(),
                        "weight": node.weight,
                        "labels": list(node.labels),
                        "is_dream": "__dream__" in node.labels,
                        "activation": float(node.activation),
                        "occupied": node.is_occupied,
                    })

            occupied_ids = {n["id"] for n in nodes}
            edges = []
            for n1, n2, etype in self._edges:
                if n1 in occupied_ids and n2 in occupied_ids:
                    edges.append({"source": n1, "target": n2, "type": etype})

            return {"nodes": nodes, "edges": edges}

    def stats(self, force: bool = False) -> Dict:
        now = time.time()
        with self._lock:
            if not force and self._stats_cache and now - self._stats_cache_time < 10:
                return self._stats_cache
            total = len(self._nodes)
            occupied = self._occupied_count
            bridges = sum(1 for n in self._nodes.values() if "__pulse_bridge__" in n.labels)
            face_edges = sum(1 for _, _, t in self._edges if t == "face")
            edge_edges = sum(1 for _, _, t in self._edges if t == "edge")
            vertex_edges = sum(1 for _, _, t in self._edges if t == "vertex")
            all_nodes = list(self._nodes.values())
            occupied_nodes = [n for n in all_nodes if n.is_occupied]
            avg_activation = float(np.mean([n.activation for n in occupied_nodes])) if occupied_nodes else 0
            avg_face_conn = float(np.mean([len(n.face_neighbors) for n in all_nodes])) if all_nodes else 0
            avg_vertex_conn = float(np.mean([len(n.vertex_neighbors) for n in all_nodes])) if all_nodes else 0
            fired_count = sum(1 for n in all_nodes if n.fired)
            crystal_nodes = sum(1 for n in all_nodes if n.crystal_channels)

            result = {
                "total_nodes": total,
                "occupied_nodes": occupied,
                "bridge_nodes": bridges,
                "empty_nodes": total - occupied,
                "face_edges": face_edges,
                "edge_edges": edge_edges,
                "vertex_edges": vertex_edges,
                "avg_vertex_connections": avg_vertex_conn,
                "avg_activation": avg_activation,
                "avg_face_connections": avg_face_conn,
                "pulse_count": self._pulse_count,
                "bridge_count": self._bridge_count,
                "cascade_count": self._cascade_count,
                "crystal_nodes": crystal_nodes,
                "pulse_engine_running": self._pulse_engine is not None and self._pulse_engine.is_alive(),
                "pulse_type_counts": {t.value: c for t, c in self._pulse_type_counts.items()},
                "fired_nodes": fired_count,
                "adaptive_interval": round(self._adaptive_interval, 3),
                "bridge_rate": round(self._recent_bridge_rate, 6),
                "hebbian": self._hebbian.stats(),
                "crystallized": self._crystallized.stats(),
                "reflection_field": self._reflection_field.stats() if self._reflection_field else {"phase_state": "uninitialized"},
                "lattice_integrity": self._lattice_checker.get_latest() if self._lattice_checker else None,
                "self_check": self.self_check_status() if self._self_check else {"engine_running": False},
                "self_organize": self._self_organize.stats() if self._self_organize else {"engine_active": False},
                "honeycomb_cells": self._cell_map.structural_analysis(self._nodes),
                "spatial_autocorrelation": {
                    "morans_i": round(self._spatial_autocorrelation, 4),
                    "history_len": len(self._autocorrelation_history),
                },
                "bcc_unit_index_size": len(self._bcc_unit_index),
                "current_phase": self._current_phase,
                "resonance_events": self._resonance_events,
                "query_emergence": dict(self._query_emergence_stats),
                "lifecycle": self.get_lifecycle_stats(),
                "attention": self.attention_status(),
                "emergence_summary": self._emergence_history[-1] if self._emergence_history else None,
                "emergence_history_count": len(self._emergence_history),
                "self_regulation": self._self_regulation.status() if self._self_regulation else None,
                "pcnn_config": {
                    "face_decay": PCNNConfig.FACE_DECAY,
                    "edge_decay": round(PCNNConfig.FACE_DECAY * PCNNConfig.EDGE_DECAY_FACTOR, 3),
                    "beta": PCNNConfig.BETA,
                    "alpha_feed": PCNNConfig.ALPHA_FEED,
                    "alpha_link": PCNNConfig.ALPHA_LINK,
                    "alpha_threshold": PCNNConfig.ALPHA_THRESHOLD,
                    "cascade_branching": PCNNConfig.CASCADE_BRANCHING_FACTOR,
                    "cascade_max_depth": PCNNConfig.CASCADE_MAX_DEPTH,
                    "crystallize_threshold": PCNNConfig.CRYSTALLIZE_THRESHOLD,
                },
            }
            self._stats_cache = result
            self._stats_cache_time = now
            return result

    def _detect_empty_associations(self, nid: str, node: "HoneycombNode"):
        all_neighbors = node.face_neighbors + node.edge_neighbors
        occupied_count = 0
        empty_count = 0
        for nnid in all_neighbors:
            nn = self._nodes.get(nnid)
            if nn is None:
                empty_count += 1
            elif not nn.is_occupied:
                empty_count += 1
            else:
                occupied_count += 1

        total = occupied_count + empty_count
        if total > 0 and occupied_count == 0:
            if "__isolated__" not in node.labels:
                node.labels.append("__isolated__")
                node.metadata["isolation_detected"] = time.time()
                logger.debug("Empty association detected at node %s", nid[:8])

    def run_self_check(self) -> Dict[str, Any]:
        if self._self_check is None:
            return {"error": "self_check engine not initialized"}
        result = self._self_check.run_full_check()
        with self._lock:
            self._self_check._history.append(result)
            if len(self._self_check._history) > self._self_check._max_history:
                self._self_check._history = self._self_check._history[-self._self_check._max_history // 2:]
        return result.to_dict()

    def self_check_status(self) -> Dict[str, Any]:
        if self._self_check is None:
            return {"engine_running": False}
        status = self._self_check.stats()
        latest = self._self_check.get_latest()
        status["latest_check"] = latest
        return status

    def self_check_history(self, n: int = 10) -> List[Dict]:
        if self._self_check is None:
            return []
        return self._self_check.get_history(n)

    def detect_duplicates(self) -> List[Dict[str, Any]]:
        with self._lock:
            occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
            if len(occupied) < 2:
                return []

            contents_tokens = {}
            for nid, node in occupied:
                contents_tokens[nid] = self._extract_tokens(node.content)

            duplicates = []
            checked = set()
            threshold = PCNNConfig.DUPLICATE_TOKEN_OVERLAP

            for i, (nid_a, node_a) in enumerate(occupied):
                tokens_a = contents_tokens[nid_a]
                if not tokens_a:
                    continue
                for j in range(i + 1, min(i + 50, len(occupied))):
                    nid_b, node_b = occupied[j]
                    pair_key = (min(nid_a, nid_b), max(nid_a, nid_b))
                    if pair_key in checked:
                        continue
                    checked.add(pair_key)

                    tokens_b = contents_tokens.get(nid_b, set())
                    if not tokens_b:
                        continue

                    intersection = len(tokens_a & tokens_b)
                    union = len(tokens_a | tokens_b)
                    if union == 0:
                        continue
                    jaccard = intersection / union

                    if jaccard >= threshold:
                        duplicates.append({
                            "node_a": nid_a[:12],
                            "node_b": nid_b[:12],
                            "similarity": round(jaccard, 3),
                            "content_a": node_a.content[:80],
                            "content_b": node_b.content[:80],
                            "weight_a": float(node_a.weight),
                            "weight_b": float(node_b.weight),
                            "labels_a": list(node_a.labels),
                            "labels_b": list(node_b.labels),
                        })
            return duplicates

    def detect_isolated(self) -> List[Dict[str, Any]]:
        with self._lock:
            isolated = []
            for nid, node in self._nodes.items():
                if not node.is_occupied:
                    continue
                occupied_neighbors = 0
                total_neighbors = 0
                for fnid in node.face_neighbors:
                    fn = self._nodes.get(fnid)
                    total_neighbors += 1
                    if fn and fn.is_occupied:
                        occupied_neighbors += 1
                for enid in node.edge_neighbors:
                    en = self._nodes.get(enid)
                    total_neighbors += 1
                    if en and en.is_occupied:
                        occupied_neighbors += 1

                if occupied_neighbors == 0:
                    isolated.append({
                        "id": nid[:12],
                        "content": node.content[:60],
                        "weight": float(node.weight),
                        "activation": round(node.activation, 4),
                        "total_neighbors": total_neighbors,
                        "labels": list(node.labels),
                    })
            return isolated

    def browse_timeline(self, direction: str = "newest", limit: int = 20,
                        label_filter=None, min_weight: float = 0.0,
                        offset: int = 0) -> tuple:
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
                    "weight": float(node.weight),
                    "labels": list(node.labels),
                    "creation_time": node.creation_time,
                    "activation": node.activation,
                })
            items.sort(key=lambda x: x["creation_time"], reverse=(direction == "newest"))
            total = len(items)
            return items[offset:offset + limit], total

    def associate(self, tetra_id: str, max_depth: int = 2) -> List[Dict]:
        with self._lock:
            node = self._nodes.get(tetra_id)
            if node is None:
                return []

            visited = {tetra_id}
            frontier = [tetra_id]
            results = []
            shortcut_edges = {}
            if self._self_organize and hasattr(self._self_organize, '_shortcut_by_node'):
                for nid_key, sc_list in self._self_organize._shortcut_by_node.items():
                    if sc_list:
                        best = max(sc_list, key=lambda s: s[1])
                        shortcut_edges[nid_key] = (best[0][0] if best[0][0] != nid_key else best[0][1], best[1])

            for depth in range(max_depth):
                next_frontier = []
                for fid in frontier:
                    fn = self._nodes.get(fid)
                    if fn is None:
                        continue
                    neighbors = list(fn.face_neighbors) + list(fn.edge_neighbors) + list(fn.vertex_neighbors)
                    if fid in shortcut_edges:
                        target_id, strength = shortcut_edges[fid]
                        if target_id not in visited:
                            neighbors.append(target_id)
                    for nid in neighbors:
                        if nid in visited:
                            continue
                        visited.add(nid)
                        nn = self._nodes.get(nid)
                        if nn and nn.is_occupied:
                            conn = "face" if nid in fn.face_neighbors else "edge" if nid in fn.edge_neighbors else "vertex" if nid in fn.vertex_neighbors else "shortcut"
                            hebb_w = self._hebbian.get_path_bias(fid, nid)
                            crystal_raw = 0.0
                            if isinstance(fn.crystal_channels, dict):
                                ch = fn.crystal_channels.get(nid)
                                if isinstance(ch, (int, float, np.floating, np.integer)):
                                    crystal_raw = float(ch)
                            relevance = nn.weight * 0.3 + nn.activation * 0.15 + hebb_w * 0.15 + crystal_raw * 0.2
                            if conn == "shortcut":
                                relevance += strength * 0.5
                            if conn == "face":
                                relevance += 0.3
                            elif conn == "edge":
                                relevance += 0.1
                            results.append({
                                "id": nid,
                                "content": nn.content,
                                "type": conn,
                                "weight": float(nn.weight),
                                "labels": list(nn.labels),
                                "activation": nn.activation,
                                "relevance": round(relevance, 3),
                                "depth": depth + 1,
                            })
                        next_frontier.append(nid)
                frontier = next_frontier

            results.sort(key=lambda x: -x.get("relevance", 0))
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
                "cascade_count": self._cascade_count,
                "engine_running": self._pulse_engine is not None and self._pulse_engine.is_alive(),
                "recent_bridges": recent,
                "hot_nodes": [(nid[:8], round(acc, 3)) for nid, acc in hot_nodes],
                "pulse_type_counts": {t.value: c for t, c in self._pulse_type_counts.items()},
                "adaptive_interval": round(self._adaptive_interval, 3),
                "hebbian": self._hebbian.stats(),
                "hebbian_top_paths": self._hebbian.get_top_paths(10),
                "crystallized": self._crystallized.stats(),
                "crystallized_top": self._crystallized.top_crystals(10),
            }

    def get_pcnn_node_states(self, n: int = 20) -> List[Dict]:
        with self._lock:
            nodes = sorted(
                [(nid, nd) for nid, nd in self._nodes.items() if nd.is_occupied],
                key=lambda x: -x[1].internal_activity
            )[:n]
            return [
                {
                    "id": nid[:8],
                    "feeding": round(node.feeding, 4),
                    "linking": round(node.linking, 4),
                    "internal_activity": round(node.internal_activity, 4),
                    "threshold": round(node.threshold, 4),
            "fired": bool(node.fired),
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

    def run_lattice_check(self) -> Dict[str, Any]:
        if self._lattice_checker is None:
            self._lattice_checker = LatticeIntegrityChecker(self)
        report = self._lattice_checker.run_full_check()
        return report.to_dict()

    def lattice_check_status(self) -> Dict[str, Any]:
        if self._lattice_checker is None:
            return {"checker_running": False, "checks_performed": 0}
        return self._lattice_checker.stats()

    def lattice_check_history(self, n: int = 10) -> List[Dict]:
        if self._lattice_checker is None:
            return []
        return self._lattice_checker.get_history(n)

    def crystallized_status(self) -> Dict[str, Any]:
        return {
            "stats": self._crystallized.stats(),
            "top_crystals": self._crystallized.top_crystals(20),
        }

    def trigger_cascade(self, source_id: Optional[str] = None, strength: float = 0.5) -> Dict[str, Any]:
        with self._lock:
            if source_id is None:
                occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
                if not occupied:
                    return {"error": "no occupied nodes"}
                source_id = self._select_source_cascade(occupied)
            self._emit_pulse(source_id, strength, PulseType.CASCADE)
            self._pulse_count += 1
            return {
                "triggered": True,
                "source": source_id[:12],
                "strength": strength,
                "cascade_count": self._cascade_count,
            }

    def _dream_cascade_pulse(self, source_a: str, source_b: str) -> Optional[Dict]:
        """
        Run a targeted cascade pulse from source_a toward source_b for dream engine.
        Returns path taken and resonance scores at intermediate nodes.
        """
        na = self._nodes.get(source_a)
        nb = self._nodes.get(source_b)
        if na is None or nb is None:
            return None

        target_pos = nb.position
        resonance_nodes: Dict[str, float] = {}
        path_nodes: List[str] = []
        visited = {source_a}
        current_id = source_a
        max_steps = 10

        for step in range(max_steps):
            if current_id == source_b:
                break
            current = self._nodes.get(current_id)
            if current is None:
                break

            path_nodes.append(current_id)

            candidates = []
            for fnid in current.face_neighbors[:8]:
                if fnid in visited:
                    continue
                fn = self._nodes.get(fnid)
                if fn is None:
                    continue
                direction_score = 1.0
                if fn.is_occupied:
                    dist_to_target = float(np.linalg.norm(fn.position - target_pos))
                    dist_current = float(np.linalg.norm(current.position - target_pos))
                    if dist_to_target < dist_current:
                        direction_score = 1.5
                crystal_boost = self._crystallized.get_boost(current_id, fnid)
                hebbian_w = self._hebbian.get_path_bias(current_id, fnid)
                candidates.append((fnid, direction_score * (1.0 + crystal_boost) * (1.0 + hebbian_w * 0.5), "face"))

            for enid in current.edge_neighbors[:6]:
                if enid in visited:
                    continue
                en = self._nodes.get(enid)
                if en is None:
                    continue
                direction_score = 1.0
                if en.is_occupied:
                    dist_to_target = float(np.linalg.norm(en.position - target_pos))
                    dist_current = float(np.linalg.norm(current.position - target_pos))
                    if dist_to_target < dist_current:
                        direction_score = 1.3
                crystal_boost = self._crystallized.get_boost(current_id, enid)
                hebbian_w = self._hebbian.get_path_bias(current_id, enid)
                candidates.append((enid, direction_score * 0.7 * (1.0 + crystal_boost) * (1.0 + hebbian_w * 0.5), "edge"))

            if not candidates:
                break

            for nid, score, _ in candidates:
                node = self._nodes.get(nid)
                if node and node.is_occupied and nid != source_a and nid != source_b:
                    resonance = node.pulse_accumulator * 0.5 + node.activation * 0.3 + node.weight * 0.2
                    if resonance > 0.1:
                        resonance_nodes[nid] = min(1.0, resonance)

            total_w = sum(w for _, w, _ in candidates)
            if total_w <= 0:
                break
            weights = [w for _, w, _ in candidates]
            if not all(self._is_finite_pos(w) for w in weights):
                break

            chosen_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
            next_id = candidates[chosen_idx][0]
            visited.add(next_id)
            next_node = self._nodes.get(next_id)
            if next_node:
                next_node.pulse_accumulator = min(
                    PCNNConfig.MAX_PULSE_ACCUMULATOR,
                    next_node.pulse_accumulator + 0.3
                )
            current_id = next_id

        cascade_strength = 0.0
        if resonance_nodes:
            cascade_strength = max(resonance_nodes.values())

        return {
            "path_nodes": path_nodes,
            "resonance_nodes": resonance_nodes,
            "cascade_strength": cascade_strength,
            "steps_taken": len(path_nodes),
        }

    @staticmethod
    def _is_finite_pos(v: float) -> bool:
        import math as _math
        return _math.isfinite(v) and v > 0

    def trigger_structure_pulse(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if source_id is None:
                occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
                if not occupied:
                    return {"error": "no occupied nodes"}
                source_id = self._select_source_structure(occupied)
            self._emit_pulse(source_id, PCNNConfig.STRUCTURE_STRENGTH, PulseType.STRUCTURE)
            self._pulse_count += 1
            return {
                "triggered": True,
                "source": source_id[:12],
                "strength": PCNNConfig.STRUCTURE_STRENGTH,
            }

    def force_crystallize(self) -> Dict[str, Any]:
        with self._lock:
            self._crystal_maintenance()
            return {
                "crystallized": self._crystallized.stats(),
                "hebbian_edges": len(self._hebbian._edges),
            }

    def run_self_organize(self) -> Dict[str, Any]:
        if self._self_organize is None:
            self._self_organize = SelfOrganizeEngine(self)
        result = self._self_organize.run_cycle()
        return result.to_dict()

    def self_organize_status(self) -> Dict[str, Any]:
        if self._self_organize is None:
            return {"engine_active": False}
        status = self._self_organize.stats()
        status["clusters"] = self._self_organize.get_clusters()
        status["shortcuts"] = self._self_organize.get_shortcuts(10)
        return status

    def self_organize_history(self, n: int = 10) -> List[Dict]:
        if self._self_organize is None:
            return []
        return self._self_organize.get_history(n)

    def get_clusters(self) -> List[Dict]:
        if self._self_organize is None:
            return []
        return self._self_organize.get_clusters()

    def get_shortcuts(self, n: int = 20) -> List[Dict]:
        if self._self_organize is None:
            return []
        return self._self_organize.get_shortcuts(n)

    def honeycomb_analysis(self) -> Dict[str, Any]:
        with self._lock:
            self._cell_map.update_all_densities(self._nodes)
            analysis = self._cell_map.structural_analysis(self._nodes)
            best_cells = self._cell_map.get_best_cells(10)
            dense_cells = self._cell_map.get_cells_by_density(10)
            analysis["best_quality_cells"] = [c.to_dict() for c in best_cells]
            analysis["highest_density_cells"] = [c.to_dict() for c in dense_cells]
            return analysis

    def get_tetrahedral_cells(self, n: int = 20, sort_by: str = "quality") -> List[Dict]:
        with self._lock:
            self._cell_map.update_all_densities(self._nodes)
            if sort_by == "density":
                cells = self._cell_map.get_cells_by_density(n)
            else:
                cells = self._cell_map.get_best_cells(n)
            return [c.to_dict() for c in cells]

    def get_cell_for_node(self, node_id: str) -> List[Dict]:
        cells = self._cell_map.get_cells_for_node(node_id)
        with self._lock:
            for c in cells:
                c.update_density(self._nodes)
            return [c.to_dict() for c in cells]

    def run_dream_cycle(self) -> Dict[str, Any]:
        if self._dream_engine is None:
            self._dream_engine = DreamEngine(self)
        result = self._dream_engine.run_dream_cycle()
        return result.to_dict()

    def dream_status(self) -> Dict[str, Any]:
        if self._dream_engine is None:
            return {"engine_active": False}
        return self._dream_engine.stats()

    def dream_history(self, n: int = 10) -> List[Dict]:
        if self._dream_engine is None:
            return []
        return self._dream_engine.get_history(n)

    def agent_get_context(self, topic: str, max_memories: int = 15) -> Dict[str, Any]:
        if self._agent_driver is None:
            self._agent_driver = AgentMemoryDriver(self)
        return self._agent_driver.get_context(topic, max_memories)

    def agent_reasoning_chain(self, source_id: str, target_query: str, max_hops: int = 5) -> Dict[str, Any]:
        if self._agent_driver is None:
            self._agent_driver = AgentMemoryDriver(self)
        return self._agent_driver.reasoning_chain(source_id, target_query, max_hops)

    def agent_suggest(self, context: str = "") -> Dict[str, Any]:
        if self._agent_driver is None:
            self._agent_driver = AgentMemoryDriver(self)
        return self._agent_driver.suggest_actions(context)

    def agent_navigate(self, source_id: str, target_id: str, max_hops: int = 6) -> Dict[str, Any]:
        if self._agent_driver is None:
            self._agent_driver = AgentMemoryDriver(self)
        return self._agent_driver.navigate(source_id, target_id, max_hops)

    def feedback_record(self, action: str, context_id: str, outcome: str,
                        confidence: float = 0.5, reasoning: str = "",
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        if self._feedback_loop is None:
            self._feedback_loop = FeedbackLoop(self)
        return self._feedback_loop.record_outcome(action, context_id, outcome, confidence, reasoning, metadata)

    def feedback_learn(self, action: str, source_id: str, target_id: str,
                       success: bool, confidence: float = 0.5) -> Dict[str, Any]:
        if self._feedback_loop is None:
            self._feedback_loop = FeedbackLoop(self)
        return self._feedback_loop.learn_from_action(action, source_id, target_id, success, confidence)

    def feedback_stats(self) -> Dict[str, Any]:
        if self._feedback_loop is None:
            return {"total_feedback": 0}
        return self._feedback_loop.get_stats()

    def feedback_insights(self) -> List[Dict[str, Any]]:
        if self._feedback_loop is None:
            return []
        return self._feedback_loop.get_learning_insights()

    def session_create(self, agent_id: str, metadata: Dict = None) -> str:
        if self._session_manager is None:
            self._session_manager = SessionManager(self)
        return self._session_manager.create_session(agent_id, metadata)

    def session_add(self, session_id: str, role: str, content: str,
                    metadata: Dict = None) -> Dict[str, Any]:
        if self._session_manager is None:
            self._session_manager = SessionManager(self)
        return self._session_manager.add_to_session(session_id, role, content, metadata)

    def session_recall(self, session_id: str, n: int = 20) -> Dict[str, Any]:
        if self._session_manager is None:
            self._session_manager = SessionManager(self)
        return self._session_manager.recall_session(session_id, n)

    def session_consolidate(self, session_id: str) -> Dict[str, Any]:
        if self._session_manager is None:
            self._session_manager = SessionManager(self)
        return self._session_manager.consolidate_session(session_id)

    def session_list(self) -> List[Dict[str, Any]]:
        if self._session_manager is None:
            return []
        return self._session_manager.list_sessions()

    def session_cleanup(self, max_age: int = 3600) -> Dict[str, Any]:
        if self._session_manager is None:
            return {"expired": 0, "session_ids": []}
        return self._session_manager.session_cleanup(max_age)

    def session_get(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self._session_manager is None:
            return None
        return self._session_manager.get_session(session_id)

    def export_full_state(self) -> Dict[str, Any]:
        with self._lock:
            state: Dict[str, Any] = {}

            nodes: Dict[str, Dict] = {}
            for nid, node in self._nodes.items():
                if not node.is_occupied:
                    continue
                nodes[nid] = {
                    "content": node.content,
                    "labels": list(node.labels),
                    "weight": float(node.weight),
                    "activation": float(node.activation),
                    "base_activation": float(node.base_activation),
                    "centroid": node.position.tolist(),
                    "metadata": {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else bool(v) if isinstance(v, np.bool_) else v
                        for k, v in node.metadata.items()
                    } if node.metadata else {},
                    "creation_time": float(node.creation_time),
                    "reinforcement_count": int(node.reinforcement_count),
                    "domain_affinity": {k: float(v) for k, v in node.domain_affinity.items()},
                    "access_count": int(node.access_count),
                    "crystal_channels": {k: float(v) for k, v in node.crystal_channels.items()},
                    "pulse_accumulator": float(node.pulse_accumulator),
                    "decay_rate": float(node.decay_rate),
                }
            state["nodes"] = nodes

            hebbian_edges = []
            if self._hebbian:
                for (a, b), w in self._hebbian._edges.items():
                    tc = self._hebbian._traversal_count.get((a, b), 0)
                    hebbian_edges.append({
                        "source": a, "target": b,
                        "weight": float(w), "traversal_count": int(tc),
                    })
            state["hebbian_edges"] = hebbian_edges

            crystals = []
            if self._crystallized:
                for key, crystal in self._crystallized._crystals.items():
                    crystals.append({
                        "nodes": [crystal["nodes"][0], crystal["nodes"][1]],
                        "hebbian_weight": float(crystal["hebbian_weight"]),
                        "crystal_weight": float(crystal["crystal_weight"]),
                        "created_at": float(crystal["created_at"]),
                        "last_reinforced": float(crystal["last_reinforced"]),
                        "transmission_count": int(crystal.get("transmission_count", 0)),
                    })
            state["crystals"] = crystals

            if self._self_organize:
                shortcuts = []
                for (a, b), s in self._self_organize._shortcuts.items():
                    shortcuts.append({"from": a, "to": b, "strength": float(s)})
                state["shortcuts"] = shortcuts

                clusters = []
                for c in self._self_organize._clusters:
                    clusters.append({
                        "cluster_id": c.cluster_id,
                        "labels": sorted(c.labels),
                        "node_ids": list(c.node_ids),
                        "centroid": c.centroid.tolist() if c.centroid is not None else None,
                        "avg_weight": float(c.avg_weight),
                        "total_activation": float(c.total_activation),
                        "quality_score": float(c.quality_score),
                    })
                state["clusters"] = clusters
            else:
                state["shortcuts"] = []
                state["clusters"] = []

            if self._dream_engine:
                state["dream_total"] = self._dream_engine._total_dreams
                de = {}
                for did, dinfo in self._dream_engine._dream_effectiveness.items():
                    de[did] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in dinfo.items()}
                state["dream_effectiveness"] = de
            else:
                state["dream_total"] = 0
                state["dream_effectiveness"] = {}

            if self._feedback_loop:
                state["feedback_outcome_counts"] = dict(self._feedback_loop._outcome_counts)
                state["feedback_action_counts"] = dict(self._feedback_loop._action_counts)
                state["feedback_consecutive_positive"] = dict(self._feedback_loop._consecutive_positive)
                state["feedback_records"] = [
                    {
                        "action": r.action, "context_id": r.context_id,
                        "outcome": r.outcome, "confidence": float(r.confidence),
                        "reasoning": r.reasoning, "timestamp": float(r.timestamp),
                        "metadata": r.metadata,
                    }
                    for r in self._feedback_loop._records
                ]
            else:
                state["feedback_outcome_counts"] = {"positive": 0, "negative": 0, "neutral": 0}
                state["feedback_action_counts"] = {}
                state["feedback_consecutive_positive"] = {}
                state["feedback_records"] = []

            if self._session_manager:
                sessions = {}
                for sid, session in self._session_manager._sessions.items():
                    sessions[sid] = {
                        "session_id": session.session_id,
                        "agent_id": session.agent_id,
                        "created_at": float(session.created_at),
                        "last_active": float(session.last_active),
                        "metadata": session.metadata,
                        "records": [
                            {
                                "role": r.role, "content": r.content,
                                "timestamp": float(r.timestamp),
                                "memory_id": r.memory_id, "metadata": r.metadata,
                            }
                            for r in session.records
                        ],
                        "ephemeral_ids": list(session.ephemeral_ids),
                    }
                state["sessions"] = sessions
            else:
                state["sessions"] = {}

            if self._reflection_field:
                state["reflection_field"] = {
                    "node_energy": {k: float(v) for k, v in self._reflection_field._node_energy.items()},
                    "field_entropy": float(self._reflection_field._field_entropy),
                    "phase_state": self._reflection_field._phase_state,
                    "phase_transitions": int(self._reflection_field._phase_transitions),
                    "energy_history": [float(v) for v in self._reflection_field._energy_history],
                }
            else:
                state["reflection_field"] = {
                    "node_energy": {}, "field_entropy": 1.0,
                    "phase_state": "fluid", "phase_transitions": 0,
                    "energy_history": [],
                }

            state["counters"] = {
                "pulse_count": self._pulse_count,
                "bridge_count": self._bridge_count,
                "cascade_count": self._cascade_count,
                "adaptive_interval": float(self._adaptive_interval),
                "recent_bridge_rate": float(self._recent_bridge_rate),
                "current_phase": self._current_phase,
                "spatial_autocorrelation": float(self._spatial_autocorrelation),
                "autocorrelation_history": [float(v) for v in self._autocorrelation_history],
                "pulse_type_counts": {t.value: c for t, c in self._pulse_type_counts.items()},
                "crystal_maintenance_cycle": self._crystal_maintenance_cycle,
            }

            state["config"] = {
                "resolution": self._resolution,
                "spacing": float(self._spacing),
            }
            state["resonance_events"] = [
                {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in ev.items()}
                for ev in self._resonance_events
            ]
            state["query_emergence_stats"] = dict(self._query_emergence_stats)

            temporal_export = {}
            for nid, edges in self._temporal_edges.items():
                temporal_export[nid] = [(eid, float(p)) for eid, p in edges]
            state["temporal_edges"] = temporal_export
            state["recent_stores"] = [(rid, float(ts)) for rid, ts in self._recent_stores]

            attention_export = []
            for focus in self._attention_foci:
                attention_export.append({
                    "center": focus["center"],
                    "radius": focus["radius"],
                    "strength": focus["strength"],
                    "labels": list(focus["labels"]),
                    "created_at": float(focus["created_at"]),
                })
            state["attention_foci"] = attention_export
            state["attention_mask"] = {nid: float(v) for nid, v in self._attention_mask.items()}
            state["attention_stats"] = dict(self._attention_stats)

            emergence_export = []
            for snap in self._emergence_history[-50:]:
                snap_copy = dict(snap)
                emergence_export.append(snap_copy)
            state["emergence_history"] = emergence_export

            if self._self_regulation:
                state["self_regulation"] = {
                    "params": dict(self._self_regulation._params),
                    "hormones": dict(self._self_regulation._endocrine_hormones),
                    "stress_level": float(self._self_regulation._stress_level),
                    "autonomic_mode": self._self_regulation._autonomic_mode,
                    "circadian_phase": self._self_regulation._circadian_phase,
                    "immune_total_repairs": self._self_regulation._immune_total_repairs,
                    "regulation_count": self._self_regulation._regulation_count,
                    "query_success_history": list(self._self_regulation._query_success_history[-50:]),
                    "start_time": float(self._self_regulation._start_time),
                }

            return state

    def import_full_state(self, state_dict: Dict[str, Any]):
        with self._lock:
            self._stats_cache = None
            self._stats_cache_time = 0

            nodes_data = state_dict.get("nodes", {})
            for nid, ndata in nodes_data.items():
                node = self._nodes.get(nid)
                if node is None:
                    continue
                if node.is_occupied:
                    self._clear_node(nid, node)

                node.content = ndata.get("content")
                node.labels = ndata.get("labels", [])
                node.weight = float(ndata.get("weight", 1.0))
                node.activation = float(ndata.get("activation", node.weight))
                node.base_activation = float(ndata.get("base_activation", max(0.01, node.weight * 0.1)))
                meta_raw = ndata.get("metadata", {})
                node.metadata = {k: v for k, v in meta_raw.items()} if meta_raw else {}
                node.creation_time = float(ndata.get("creation_time", 0))
                node.reinforcement_count = int(ndata.get("reinforcement_count", 0))
                da_raw = ndata.get("domain_affinity", {})
                node.domain_affinity = {k: float(v) for k, v in da_raw.items()} if da_raw else {}
                node.access_count = int(ndata.get("access_count", 0))
                cc_raw = ndata.get("crystal_channels", {})
                node.crystal_channels = {k: float(v) for k, v in cc_raw.items()} if cc_raw else {}
                node.pulse_accumulator = float(ndata.get("pulse_accumulator", 0))
                node.decay_rate = float(ndata.get("decay_rate", 0.001))
                node.feeding = node.weight
                node.threshold = PCNNConfig.V_THETA * 0.5

                if node.content:
                    chash = hashlib.sha256(node.content.encode()).hexdigest()[:12]
                    self._content_hash_index[chash] = nid
                for lbl in node.labels:
                    self._label_index[lbl].add(nid)
                for tok in self._extract_tokens(node.content or ""):
                    self._content_token_index[tok].add(nid)
                if node.content:
                    self._occupied_ids.add(nid)
                    self._occupied_count += 1
                    if self._centroid is None:
                        self._centroid = node.position.copy()
                    else:
                        self._centroid = (self._centroid * (self._occupied_count - 1) + node.position) / self._occupied_count

            if self._hebbian:
                for edge in state_dict.get("hebbian_edges", []):
                    src = edge.get("source")
                    tgt = edge.get("target")
                    if src and tgt:
                        self._hebbian._edges[(src, tgt)] = float(edge.get("weight", 0.5))
                        self._hebbian._traversal_count[(src, tgt)] = int(edge.get("traversal_count", 0))

            if self._crystallized:
                self._crystallized._crystals.clear()
                for cdata in state_dict.get("crystals", []):
                    cn = cdata.get("nodes", [])
                    if len(cn) == 2:
                        key = (min(cn[0], cn[1]), max(cn[0], cn[1]))
                        self._crystallized._crystals[key] = {
                            "nodes": (cn[0], cn[1]),
                            "hebbian_weight": float(cdata.get("hebbian_weight", 2.0)),
                            "crystal_weight": float(cdata.get("crystal_weight", 1.0)),
                            "created_at": float(cdata.get("created_at", time.time())),
                            "last_reinforced": float(cdata.get("last_reinforced", time.time())),
                            "transmission_count": int(cdata.get("transmission_count", 0)),
                        }

            if self._self_organize:
                self._self_organize._shortcuts.clear()
                self._self_organize._shortcut_by_node.clear()
                for sc in state_dict.get("shortcuts", []):
                    a, b = sc.get("from"), sc.get("to")
                    s = float(sc.get("strength", 0.5))
                    if a and b:
                        key = (min(a, b), max(a, b))
                        self._self_organize._shortcuts[key] = s
                        self._self_organize._shortcut_by_node[a].append((key, s))
                        self._self_organize._shortcut_by_node[b].append((key, s))

                self._self_organize._clusters.clear()
                for cdata in state_dict.get("clusters", []):
                    cluster = SemanticCluster(
                        cdata.get("cluster_id", ""), set(cdata.get("labels", []))
                    )
                    cluster.node_ids = cdata.get("node_ids", [])
                    centroid = cdata.get("centroid")
                    cluster.centroid = np.array(centroid, dtype=np.float32) if centroid else None
                    cluster.avg_weight = float(cdata.get("avg_weight", 0))
                    cluster.total_activation = float(cdata.get("total_activation", 0))
                    cluster.quality_score = float(cdata.get("quality_score", 0))
                    self._self_organize._clusters.append(cluster)

            if self._dream_engine:
                self._dream_engine._total_dreams = state_dict.get("dream_total", 0)
                self._dream_engine._dream_effectiveness.clear()
                for did, dinfo in state_dict.get("dream_effectiveness", {}).items():
                    self._dream_engine._dream_effectiveness[did] = dict(dinfo)

            if self._feedback_loop:
                fb_oc = state_dict.get("feedback_outcome_counts", {"positive": 0, "negative": 0, "neutral": 0})
                self._feedback_loop._outcome_counts = dict(fb_oc)
                self._feedback_loop._action_counts = dict(state_dict.get("feedback_action_counts", {}))
                self._feedback_loop._consecutive_positive = dict(state_dict.get("feedback_consecutive_positive", {}))
                self._feedback_loop._records.clear()
                for rdata in state_dict.get("feedback_records", []):
                    r = FeedbackRecord(
                        rdata.get("action", ""),
                        rdata.get("context_id", ""),
                        rdata.get("outcome", "neutral"),
                        float(rdata.get("confidence", 0.5)),
                        rdata.get("reasoning", ""),
                        rdata.get("metadata"),
                    )
                    r.timestamp = float(rdata.get("timestamp", time.time()))
                    self._feedback_loop._records.append(r)

            if self._session_manager:
                self._session_manager._sessions.clear()
                for sid, sdata in state_dict.get("sessions", {}).items():
                    session = Session(sid, sdata.get("agent_id", ""), sdata.get("metadata"))
                    session.created_at = float(sdata.get("created_at", time.time()))
                    session.last_active = float(sdata.get("last_active", time.time()))
                    session.ephemeral_ids = sdata.get("ephemeral_ids", [])
                    for rdata in sdata.get("records", []):
                        r = SessionRecord(
                            rdata.get("role", ""),
                            rdata.get("content", ""),
                            rdata.get("memory_id"),
                            rdata.get("metadata"),
                        )
                        r.timestamp = float(rdata.get("timestamp", time.time()))
                        session.records.append(r)
                    self._session_manager._sessions[sid] = session

            if self._reflection_field:
                rf = state_dict.get("reflection_field", {})
                ne = rf.get("node_energy", {})
                self._reflection_field._node_energy = {k: float(v) for k, v in ne.items()}
                self._reflection_field._field_entropy = float(rf.get("field_entropy", 1.0))
                self._reflection_field._phase_state = rf.get("phase_state", "fluid")
                self._reflection_field._phase_transitions = int(rf.get("phase_transitions", 0))
                eh = rf.get("energy_history", [])
                self._reflection_field._energy_history = [float(v) for v in eh]

            counters = state_dict.get("counters", {})
            self._pulse_count = counters.get("pulse_count", 0)
            self._bridge_count = counters.get("bridge_count", 0)
            self._cascade_count = counters.get("cascade_count", 0)
            self._adaptive_interval = float(counters.get("adaptive_interval", PCNNConfig.BASE_PULSE_INTERVAL))
            self._recent_bridge_rate = float(counters.get("recent_bridge_rate", 0.0))
            self._current_phase = counters.get("current_phase", "fluid")
            self._spatial_autocorrelation = float(counters.get("spatial_autocorrelation", 0.0))
            ah = counters.get("autocorrelation_history", [])
            self._autocorrelation_history = [float(v) for v in ah]
            ptc = counters.get("pulse_type_counts", {})
            for t in PulseType:
                self._pulse_type_counts[t] = ptc.get(t.value, 0)
            self._crystal_maintenance_cycle = counters.get("crystal_maintenance_cycle", 0)

            re_raw = state_dict.get("resonance_events", [])
            self._resonance_events = list(re_raw)

            qe_raw = state_dict.get("query_emergence_stats", {})
            if qe_raw:
                for k, v in qe_raw.items():
                    if k in self._query_emergence_stats:
                        self._query_emergence_stats[k] = int(v)

            te_raw = state_dict.get("temporal_edges", {})
            self._temporal_edges.clear()
            for nid, edges in te_raw.items():
                self._temporal_edges[nid] = [(eid, float(p)) for eid, p in edges]

            rs_raw = state_dict.get("recent_stores", [])
            self._recent_stores = [(rid, float(ts)) for rid, ts in rs_raw]

            af_raw = state_dict.get("attention_foci", [])
            self._attention_foci = []
            for fd in af_raw:
                self._attention_foci.append({
                    "center": fd["center"],
                    "radius": float(fd.get("radius", 5.0)),
                    "strength": float(fd.get("strength", 1.0)),
                    "labels": set(fd.get("labels", [])),
                    "created_at": float(fd.get("created_at", time.time())),
                })
            am_raw = state_dict.get("attention_mask", {})
            self._attention_mask = {nid: float(v) for nid, v in am_raw.items()}
            as_raw = state_dict.get("attention_stats", {})
            if as_raw:
                for k, v in as_raw.items():
                    if k in self._attention_stats:
                        self._attention_stats[k] = int(v)

            eh_raw = state_dict.get("emergence_history", [])
            self._emergence_history = []
            for snap in eh_raw[-50:]:
                self._emergence_history.append(dict(snap))

            sr_raw = state_dict.get("self_regulation", {})
            if sr_raw and self._self_regulation:
                self._self_regulation._params.update(sr_raw.get("params", {}))
                self._self_regulation._endocrine_hormones.update(sr_raw.get("hormones", {}))
                self._self_regulation._stress_level = float(sr_raw.get("stress_level", 0))
                self._self_regulation._autonomic_mode = sr_raw.get("autonomic_mode", "balanced")
                self._self_regulation._circadian_phase = sr_raw.get("circadian_phase", "work")
                self._self_regulation._immune_total_repairs = int(sr_raw.get("immune_total_repairs", 0))
                self._self_regulation._regulation_count = int(sr_raw.get("regulation_count", 0))
                self._self_regulation._query_success_history = sr_raw.get("query_success_history", [])
                self._self_regulation._start_time = float(sr_raw.get("start_time", time.time()))

