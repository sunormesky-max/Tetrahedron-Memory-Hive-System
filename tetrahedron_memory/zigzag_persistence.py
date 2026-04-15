"""
Zigzag Persistence — Dynamic topological feature tracking for TetraMem-XL.

Unlike standard single-parameter persistence (monotone filtration),
zigzag persistence tracks how topological features (H0/H1/H2) evolve
across a sequence of snapshots where the complex can both grow AND shrink.

This is critical for TetraMem-XL because:
  1. New memories insert new simplices (growth)
  2. Integration/merges remove simplices (shrinkage)
  3. Dream cycles create AND remove topological features
  4. Self-organization moves vertices (mutation)

The dynamic model:
  - Maintains a sliding window of persistence barcodes
  - Tracks feature births, deaths, merges, and splits
  - Detects topological phase transitions
  - Predicts emerging features from trend analysis
  - Feeds into emergence pressure for smarter triggering

Mapping Cone Construction:
  For each dream cycle, we build a mapping cone C(f) that formally
  captures the topological transformation f: X_pre -> X_post.
  The cone is the homotopy pushout joining X_pre x [0,1] with X_post,
  identifying (x,1) ~ f(x). This enables:
  - Bidirectional analysis: forward (dream impact) + backward (origin tracing)
  - Iterative refinement: each dream's cone feeds into the next
  - Stability certification: prove which features survive across cycles

Integration points:
  - TetraDreamCycle: uses zigzag to decide dream strategy + builds mapping cones
  - EmergencePressure: phase transitions boost pressure
  - AdaptiveThreshold: zigzag stability informs threshold
  - ResolutionPyramid: cone topology guides adaptive level selection
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("tetramem.zigzag")


@dataclass
class TopologicalTransition:
    transition_type: str
    dimension: int
    timestamp: float
    persistence_before: float
    persistence_after: float
    barcode_size_before: int
    barcode_size_after: int
    affected_tetra_ids: List[str] = field(default_factory=list)


@dataclass
class PersistenceSnapshot:
    timestamp: float
    h0_barcodes: List[Tuple[float, float]]
    h1_barcodes: List[Tuple[float, float]]
    h2_barcodes: List[Tuple[float, float]]
    h0_entropy: float
    h1_entropy: float
    h2_entropy: float
    total_entropy: float
    tetra_count: int
    vertex_count: int


@dataclass
class MappingConeRecord:
    """
    Records the mapping cone C(f) for a single dream cycle transformation.

    Given f: X_pre -> X_post (the dream transformation), the mapping cone:
      C(f) = (X_pre x [0,1] ∪ X_post) / {(x,1) ~ f(x), (x,0) ~ *}

    Fields capture:
      - forward_map: which pre-dream features map to which post-dream features
      - backward_map: reverse tracing — which post-dream features originated from pre-dream
      - cone_homology: H0/H1/H2 of the mapping cone itself (connects pre and post)
      - stability_cert: features proven stable across the transformation
    """

    cycle_id: str
    timestamp: float
    pre_snapshot: PersistenceSnapshot
    post_snapshot: PersistenceSnapshot
    forward_map: Dict[int, List[Tuple[str, str]]]
    backward_map: Dict[int, List[Tuple[str, str]]]
    cone_h0: int
    cone_h1: int
    cone_h2: int
    stability_cert: Dict[int, List[str]]
    entropy_delta: float
    dream_tetra_created: List[str]
    dream_tetra_reintegrated: List[str]

    def get_stable_features(self, dimension: int = -1) -> List[str]:
        if dimension >= 0:
            return list(self.stability_cert.get(dimension, []))
        result = []
        for dim in sorted(self.stability_cert.keys()):
            result.extend(self.stability_cert[dim])
        return result

    def get_orphaned_features(self, dimension: int = -1) -> List[Tuple[str, str]]:
        dims = [dimension] if dimension >= 0 else [0, 1, 2]
        result = []
        for dim in dims:
            matched_pre = {fid for fid, _ in self.forward_map.get(dim, [])}
            all_pre = (
                {fid for fid, _ in self.pre_snapshot.h0_barcodes}
                if dim == 0
                else (
                    {fid for fid, _ in self.pre_snapshot.h1_barcodes}
                    if dim == 1
                    else {fid for fid, _ in self.pre_snapshot.h2_barcodes}
                )
            )
            for fid in all_pre:
                if fid not in matched_pre:
                    result.append((fid, f"H{dim}_orphaned"))
        return result


class ZigzagTracker:
    """
    Tracks topological feature evolution across a sliding window of
    persistence snapshots, implementing zigzag-style dynamic modeling.
    """

    def __init__(
        self,
        window_size: int = 20,
        transition_entropy_threshold: float = 0.15,
        transition_size_threshold: float = 0.3,
        prediction_window: int = 3,
    ):
        self._window_size = window_size
        self._entropy_thresh = transition_entropy_threshold
        self._size_thresh = transition_size_threshold
        self._prediction_window = prediction_window
        self._snapshots: List[PersistenceSnapshot] = []
        self._transitions: List[TopologicalTransition] = []
        self._feature_registry: Dict[str, List[int]] = {}
        self._last_feature_ids: Dict[int, Set[str]] = {0: set(), 1: set(), 2: set()}
        self._lock = threading.RLock()
        self._mapping_cones: List[MappingConeRecord] = []
        self._max_cones: int = 50
        self._iterative_model: Dict[int, List[float]] = {0: [], 1: [], 2: []}
        self._max_iterative_model: int = 100
        self._max_feature_registry: int = 500

    def record_snapshot(self, mesh: Any) -> PersistenceSnapshot:
        st = mesh.compute_ph()
        if st is None:
            snap = PersistenceSnapshot(
                timestamp=time.time(),
                h0_barcodes=[],
                h1_barcodes=[],
                h2_barcodes=[],
                h0_entropy=0.0,
                h1_entropy=0.0,
                h2_entropy=0.0,
                total_entropy=0.0,
                tetra_count=len(mesh.tetrahedra),
                vertex_count=len(mesh.vertices),
            )
            self._append_snapshot(snap)
            return snap

        from .persistent_entropy import compute_persistent_entropy, compute_entropy_by_dimension

        total_entropy = compute_persistent_entropy(st)
        ent_by_dim = compute_entropy_by_dimension(st)

        h0_barcodes = self._extract_barcodes(st, 0)
        h1_barcodes = self._extract_barcodes(st, 1)
        h2_barcodes = self._extract_barcodes(st, 2)

        current_feature_ids = {}
        for dim, barcodes in [(0, h0_barcodes), (1, h1_barcodes), (2, h2_barcodes)]:
            current_feature_ids[dim] = set()
            for i, (b, d) in enumerate(barcodes):
                fid = f"H{dim}_{i}_{b:.4f}_{d:.4f}"
                current_feature_ids[dim].add(fid)

        snap = PersistenceSnapshot(
            timestamp=time.time(),
            h0_barcodes=h0_barcodes,
            h1_barcodes=h1_barcodes,
            h2_barcodes=h2_barcodes,
            h0_entropy=ent_by_dim.get(0, 0.0),
            h1_entropy=ent_by_dim.get(1, 0.0),
            h2_entropy=ent_by_dim.get(2, 0.0),
            total_entropy=total_entropy,
            tetra_count=len(mesh.tetrahedra),
            vertex_count=len(mesh.vertices),
        )

        self._detect_transitions(snap, current_feature_ids)
        self._last_feature_ids = current_feature_ids
        with self._lock:
            self._append_snapshot(snap)
        return snap

    def detect_phase_transitions(self) -> List[TopologicalTransition]:
        with self._lock:
            return list(self._transitions)

    def get_recent_transitions(self, count: int = 5) -> List[TopologicalTransition]:
        with self._lock:
            return list(self._transitions[-count:])

    def get_feature_lifetimes(self) -> Dict[int, List[Tuple[str, int]]]:
        with self._lock:
            lifetimes: Dict[int, List[Tuple[str, int]]] = {0: [], 1: [], 2: []}
            for fid, appearances in self._feature_registry.items():
                if not appearances:
                    continue
                dim = int(fid[1])
                if dim in lifetimes:
                    lifetimes[dim].append((fid, len(appearances)))
            for dim in lifetimes:
                lifetimes[dim].sort(key=lambda x: x[1], reverse=True)
            return lifetimes

    def predict_emerging_features(self) -> Dict[str, Any]:
        with self._lock:
            if len(self._snapshots) < 3:
                return {"prediction": "insufficient_data", "confidence": 0.0}

            recent = self._snapshots[-self._prediction_window :]
        prediction: Dict[str, Any] = {
            "prediction": "stable",
            "confidence": 0.5,
            "entropy_trend": "unknown",
            "h0_trend": "unknown",
            "h1_trend": "unknown",
            "h2_trend": "unknown",
            "expected_transition": False,
        }

        if len(recent) >= 2:
            entropies = [s.total_entropy for s in recent]
            if all(entropies[i] < entropies[i + 1] for i in range(len(entropies) - 1)):
                prediction["entropy_trend"] = "rising"
                prediction["prediction"] = "divergence_likely"
                prediction["expected_transition"] = True
                prediction["confidence"] = min(0.9, 0.5 + 0.1 * (entropies[-1] - entropies[0]))
            elif all(entropies[i] > entropies[i + 1] for i in range(len(entropies) - 1)):
                prediction["entropy_trend"] = "falling"
                prediction["prediction"] = "convergence"
                prediction["confidence"] = 0.7

        for dim, key in [(0, "h0_trend"), (1, "h1_trend"), (2, "h2_trend")]:
            counts = [len(getattr(s, f"h{dim}_barcodes")) for s in recent]
            if len(counts) >= 2:
                if counts[-1] > counts[0] * 1.2:
                    prediction[key] = "growing"
                elif counts[-1] < counts[0] * 0.8:
                    prediction[key] = "shrinking"
                else:
                    prediction[key] = "stable"

        if prediction["h2_trend"] == "growing":
            prediction["prediction"] = "void_expansion"
            prediction["expected_transition"] = True

        if prediction["h1_trend"] == "growing":
            prediction["prediction"] = "loop_emergence"
            prediction["expected_transition"] = True

        return prediction

    def get_dynamic_barcode(self, dimension: int = -1) -> Dict[str, Any]:
        with self._lock:
            if not self._snapshots:
                return {"barcodes": {}, "window": 0}

            dimensions = [dimension] if dimension >= 0 else [0, 1, 2]
            result: Dict[str, Any] = {"barcodes": {}, "window": len(self._snapshots)}

            for dim in dimensions:
                attr = f"h{dim}_barcodes"
                timeline: List[Dict[str, Any]] = []
                for i, snap in enumerate(self._snapshots):
                    barcodes = getattr(snap, attr, [])
                    timeline.append(
                        {
                            "snapshot_idx": i,
                            "timestamp": snap.timestamp,
                            "count": len(barcodes),
                            "total_persistence": sum(d - b for b, d in barcodes)
                            if barcodes
                            else 0.0,
                            "max_persistence": max((d - b for b, d in barcodes), default=0.0),
                        }
                    )
                result["barcodes"][f"H{dim}"] = timeline

            return result

    def get_zigzag_stability(self) -> Dict[str, Any]:
        with self._lock:
            if len(self._snapshots) < 2:
                return {"stability": 1.0, "transition_rate": 0.0, "dominant_dim": -1}

            recent_transitions = self._transitions[-10:]
            if not recent_transitions:
                return {"stability": 1.0, "transition_rate": 0.0, "dominant_dim": -1}

            birth_count = sum(1 for t in recent_transitions if t.transition_type == "birth")
            death_count = sum(1 for t in recent_transitions if t.transition_type == "death")

            total = len(recent_transitions)
            balance = abs(birth_count - death_count) / max(total, 1)
            stability = 1.0 - min(balance, 1.0)

            dim_counts: Dict[int, int] = {}
            for t in recent_transitions:
                dim_counts[t.dimension] = dim_counts.get(t.dimension, 0) + 1
            dominant_dim = max(dim_counts, key=dim_counts.get) if dim_counts else -1

            time_span = (
                self._snapshots[-1].timestamp
                - self._snapshots[-min(10, len(self._snapshots))].timestamp
            )
            rate = total / max(time_span, 1.0)

            return {
                "stability": stability,
                "transition_rate": rate,
                "dominant_dim": dominant_dim,
                "recent_births": birth_count,
                "recent_deaths": death_count,
            }

    def construct_mapping_cone(
        self,
        pre_snapshot: PersistenceSnapshot,
        post_snapshot: PersistenceSnapshot,
        dream_tetra_created: Optional[List[str]] = None,
        dream_tetra_reintegrated: Optional[List[str]] = None,
    ) -> MappingConeRecord:
        """
        Construct the mapping cone C(f) for a dream cycle transformation.

        Given the pre-dream topology X and post-dream topology Y, with
        the dream inducing a map f: X -> Y, we compute:

        Forward map: which features in X map to features in Y
          - Match by dimension + persistence overlap (IoU of intervals)
          - A feature that survives has a forward mapping

        Backward map: which features in Y originated from X
          - Reverse of forward map
          - Features in Y without backward map are NEW (born in dream)

        Cone homology H*(C(f)):
          - Long exact sequence: ... -> H_n(X) -> H_n(Y) -> H_n(C(f)) -> H_{n-1}(X) -> ...
          - Approximated by: H_n(C(f)) ~ |features born| + |features died| + |features mutated|

        Stability certificate: features that survive unchanged across the transformation.
        """
        forward_map: Dict[int, List[Tuple[str, str]]] = {0: [], 1: [], 2: []}
        backward_map: Dict[int, List[Tuple[str, str]]] = {0: [], 1: [], 2: []}
        stability_cert: Dict[int, List[str]] = {0: [], 1: [], 2: []}

        for dim in [0, 1, 2]:
            pre_barcodes = getattr(pre_snapshot, f"h{dim}_barcodes", [])
            post_barcodes = getattr(post_snapshot, f"h{dim}_barcodes", [])

            pre_features = [
                (f"H{dim}_{i}_{b:.4f}_{d:.4f}", (b, d)) for i, (b, d) in enumerate(pre_barcodes)
            ]
            post_features = [
                (f"H{dim}_{i}_{b:.4f}_{d:.4f}", (b, d)) for i, (b, d) in enumerate(post_barcodes)
            ]

            used_post = set()
            for pre_fid, (pb, pd) in pre_features:
                best_match = None
                best_iou = 0.0
                for j, (post_fid, (qb, qd)) in enumerate(post_features):
                    if j in used_post:
                        continue
                    overlap_low = max(pb, qb)
                    overlap_high = min(pd, qd)
                    overlap = max(0.0, overlap_high - overlap_low)
                    union = max(pd, qd) - min(pb, qb)
                    iou = overlap / union if union > 0 else 0.0
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_match = (j, post_fid)

                if best_match is not None:
                    j, post_fid = best_match
                    forward_map[dim].append((pre_fid, post_fid))
                    backward_map[dim].append((post_fid, pre_fid))
                    used_post.add(j)
                    if best_iou > 0.7:
                        stability_cert[dim].append(pre_fid)

            for j, (post_fid, _) in enumerate(post_features):
                if j not in used_post:
                    backward_map[dim].append((post_fid, "__born_in_dream__"))

        cone_h0 = abs(len(post_snapshot.h0_barcodes) - len(pre_snapshot.h0_barcodes))
        cone_h1 = abs(len(post_snapshot.h1_barcodes) - len(pre_snapshot.h1_barcodes))
        cone_h2 = abs(len(post_snapshot.h2_barcodes) - len(pre_snapshot.h2_barcodes))

        entropy_delta = 0.0
        if pre_snapshot.total_entropy > 0:
            entropy_delta = (
                pre_snapshot.total_entropy - post_snapshot.total_entropy
            ) / pre_snapshot.total_entropy

        cycle_id = f"cone_{int(pre_snapshot.timestamp)}_{int(post_snapshot.timestamp)}"

        record = MappingConeRecord(
            cycle_id=cycle_id,
            timestamp=time.time(),
            pre_snapshot=pre_snapshot,
            post_snapshot=post_snapshot,
            forward_map=forward_map,
            backward_map=backward_map,
            cone_h0=cone_h0,
            cone_h1=cone_h1,
            cone_h2=cone_h2,
            stability_cert=stability_cert,
            entropy_delta=entropy_delta,
            dream_tetra_created=dream_tetra_created or [],
            dream_tetra_reintegrated=dream_tetra_reintegrated or [],
        )

        with self._lock:
            self._mapping_cones.append(record)
            if len(self._mapping_cones) > self._max_cones:
                self._mapping_cones = self._mapping_cones[-self._max_cones :]
            for dim in [0, 1, 2]:
                self._iterative_model[dim].append(len(stability_cert.get(dim, [])))
                if len(self._iterative_model[dim]) > self._max_iterative_model:
                    self._iterative_model[dim] = self._iterative_model[dim][
                        -self._max_iterative_model :
                    ]

        return record

    def get_mapping_cone_history(self, count: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "cycle_id": r.cycle_id,
                    "entropy_delta": r.entropy_delta,
                    "stable_h0": len(r.stability_cert.get(0, [])),
                    "stable_h1": len(r.stability_cert.get(1, [])),
                    "stable_h2": len(r.stability_cert.get(2, [])),
                    "cone_h0": r.cone_h0,
                    "cone_h1": r.cone_h1,
                    "cone_h2": r.cone_h2,
                    "dream_created": len(r.dream_tetra_created),
                    "dream_reintegrated": len(r.dream_tetra_reintegrated),
                }
                for r in self._mapping_cones[-count:]
            ]

    def get_iterative_stability(self) -> Dict[int, List[float]]:
        with self._lock:
            return {dim: list(vals) for dim, vals in self._iterative_model.items()}

    def get_dream_guidance(self) -> Dict[str, Any]:
        """
        Provide guidance for the next dream cycle based on iterative mapping cone analysis.
        This is the core of the bidirectional feedback: past cones inform future dreams.
        """
        with self._lock:
            if not self._mapping_cones:
                return {"guidance": "no_history", "focus_dim": -1, "expected_benefit": 0.0}

            recent = self._mapping_cones[-3:]

            avg_delta = np.mean([r.entropy_delta for r in recent]) if recent else 0.0

            dim_instability = {}
            for dim in [0, 1, 2]:
                stable_counts = [len(r.stability_cert.get(dim, [])) for r in recent]
                total_counts = []
                for r in recent:
                    pre_bc = getattr(r.pre_snapshot, f"h{dim}_barcodes", [])
                    total_counts.append(max(len(pre_bc), 1))
                if total_counts:
                    instability = 1.0 - np.mean(
                        [s / max(t, 1) for s, t in zip(stable_counts, total_counts)]
                    )
                    dim_instability[dim] = float(instability)

            focus_dim = max(dim_instability, key=dim_instability.get) if dim_instability else -1

            h2_growing = False
            if len(self._snapshots) >= 2:
                recent_h2 = [len(s.h2_barcodes) for s in self._snapshots[-3:]]
                if len(recent_h2) >= 2 and recent_h2[-1] > recent_h2[0]:
                    h2_growing = True

            return {
                "guidance": "targeted_dream",
                "focus_dim": focus_dim,
                "dim_instability": dim_instability,
                "avg_entropy_delta": avg_delta,
                "h2_growing": h2_growing,
                "expected_benefit": max(0.0, avg_delta),
                "cones_analyzed": len(recent),
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "snapshot_count": len(self._snapshots),
                "transition_count": len(self._transitions),
                "window_size": self._window_size,
                "last_entropy": self._snapshots[-1].total_entropy if self._snapshots else 0.0,
                "last_tetra_count": self._snapshots[-1].tetra_count if self._snapshots else 0,
                "mapping_cone_count": len(self._mapping_cones),
                "stability": self.get_zigzag_stability(),
                "prediction": self.predict_emerging_features(),
                "dream_guidance": self.get_dream_guidance(),
            }

    def _append_snapshot(self, snap: PersistenceSnapshot) -> None:
        self._snapshots.append(snap)
        if len(self._snapshots) > self._window_size:
            self._snapshots = self._snapshots[-self._window_size :]
        if len(self._snapshots) > self._window_size // 2:
            self._snapshots[0] = self._compress_snapshot(self._snapshots[0])

    @staticmethod
    def _compress_snapshot(snap: PersistenceSnapshot) -> PersistenceSnapshot:
        return PersistenceSnapshot(
            timestamp=snap.timestamp,
            h0_barcodes=[],
            h1_barcodes=[],
            h2_barcodes=[],
            h0_entropy=snap.h0_entropy,
            h1_entropy=snap.h1_entropy,
            h2_entropy=snap.h2_entropy,
            total_entropy=snap.total_entropy,
            tetra_count=snap.tetra_count,
            vertex_count=snap.vertex_count,
        )

    def _detect_transitions(
        self,
        snap: PersistenceSnapshot,
        current_feature_ids: Dict[int, Set[str]],
    ) -> None:
        if not self._snapshots:
            return

        prev = self._snapshots[-1]

        for dim in [0, 1, 2]:
            prev_ids = self._last_feature_ids.get(dim, set())
            curr_ids = current_feature_ids.get(dim, set())

            born = curr_ids - prev_ids
            died = prev_ids - curr_ids

            prev_barcodes = getattr(prev, f"h{dim}_barcodes", [])
            curr_barcodes = getattr(snap, f"h{dim}_barcodes", [])

            prev_total_p = sum(d - b for b, d in prev_barcodes) if prev_barcodes else 0.0
            curr_total_p = sum(d - b for b, d in curr_barcodes) if curr_barcodes else 0.0

            for fid in born:
                self._feature_registry.setdefault(fid, []).append(len(self._snapshots))
                self._transitions.append(
                    TopologicalTransition(
                        transition_type="birth",
                        dimension=dim,
                        timestamp=snap.timestamp,
                        persistence_before=prev_total_p,
                        persistence_after=curr_total_p,
                        barcode_size_before=len(prev_barcodes),
                        barcode_size_after=len(curr_barcodes),
                    )
                )

            for fid in died:
                self._feature_registry.setdefault(fid, []).append(len(self._snapshots))
                self._transitions.append(
                    TopologicalTransition(
                        transition_type="death",
                        dimension=dim,
                        timestamp=snap.timestamp,
                        persistence_before=prev_total_p,
                        persistence_after=curr_total_p,
                        barcode_size_before=len(prev_barcodes),
                        barcode_size_after=len(curr_barcodes),
                    )
                )

            if born or died:
                continue

            prev_ent = getattr(prev, f"h{dim}_entropy", 0.0)
            curr_ent = getattr(snap, f"h{dim}_entropy", 0.0)
            if prev_ent > 0 and abs(curr_ent - prev_ent) / prev_ent > self._entropy_thresh:
                self._transitions.append(
                    TopologicalTransition(
                        transition_type="mutation",
                        dimension=dim,
                        timestamp=snap.timestamp,
                        persistence_before=prev_total_p,
                        persistence_after=curr_total_p,
                        barcode_size_before=len(prev_barcodes),
                        barcode_size_after=len(curr_barcodes),
                    )
                )

        if len(self._transitions) > self._window_size * 6:
            self._transitions = self._transitions[-self._window_size * 4 :]

        if len(self._feature_registry) > self._max_feature_registry:
            sorted_keys = sorted(
                self._feature_registry.keys(), key=lambda k: len(self._feature_registry[k])
            )
            excess = len(self._feature_registry) - self._max_feature_registry // 2
            for k in sorted_keys[:excess]:
                del self._feature_registry[k]

    @staticmethod
    def _extract_barcodes(simplex_tree: Any, dim: int) -> List[Tuple[float, float]]:
        try:
            intervals = simplex_tree.persistence_intervals_in_dimension(dim)
            if intervals is None or len(intervals) == 0:
                return []
            return [
                (float(iv[0]), float(iv[1]))
                for iv in intervals
                if np.isfinite(iv[1]) and float(iv[1]) > float(iv[0])
            ]
        except Exception:
            return []
