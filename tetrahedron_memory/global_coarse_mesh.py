"""
Global Coarse Mesh for topology correction.

Samples centroids from all tetrahedra in the mesh, builds a global
PH diagram, and feeds corrections back to the self-organizer.

Per the v2.0 spec: sample 8k~20k points, global PH, periodic rebuild.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


class GlobalCoarseMesh:
    def __init__(
        self,
        mesh: Any,
        sample_limit: int = 20000,
        rebuild_interval: float = 600.0,
    ):
        self.mesh = mesh
        self.sample_limit = sample_limit
        self.rebuild_interval = rebuild_interval
        self._simplex_tree = None
        self._last_rebuild_time: float = 0.0
        self._last_h0: int = 0
        self._last_h1: int = 0
        self._last_h2: int = 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def rebuild(self) -> Dict[str, Any]:
        with self._lock:
            tetrahedra = self.mesh.tetrahedra
            if len(tetrahedra) < 4 or not GUDHI_AVAILABLE:
                return {"status": "insufficient_data"}

            centroids = []
            weights = []
            for t in tetrahedra.values():
                centroids.append(t.centroid)
                weights.append(t.weight)

            centroids = np.array(centroids)
            weights = np.array(weights)

            if len(centroids) > self.sample_limit:
                indices = np.random.choice(
                    len(centroids), self.sample_limit, replace=False
                )
                centroids = centroids[indices]
                weights = weights[indices]

            ac = gudhi.AlphaComplex(
                points=centroids.tolist(),
                weights=weights.tolist(),
                precision="fast",
            )
            self._simplex_tree = ac.create_simplex_tree()
            self._simplex_tree.compute_persistence(
                homology_coeff_field=2, min_persistence=0.01
            )

            h0 = self._simplex_tree.persistence_intervals_in_dimension(0)
            h1 = self._simplex_tree.persistence_intervals_in_dimension(1)
            h2 = self._simplex_tree.persistence_intervals_in_dimension(2)

            self._last_h0 = len(h0)
            self._last_h1 = len(h1)
            self._last_h2 = len(h2)
            self._last_rebuild_time = time.time()

            return {
                "status": "ok",
                "sampled_points": len(centroids),
                "H0": len(h0),
                "H1": len(h1),
                "H2": len(h2),
            }

    def get_topology_report(self) -> Dict[str, Any]:
        if self._simplex_tree is None:
            return {"status": "not_built"}

        h0 = self._simplex_tree.persistence_intervals_in_dimension(0)
        h1 = self._simplex_tree.persistence_intervals_in_dimension(1)
        h2 = self._simplex_tree.persistence_intervals_in_dimension(2)

        avg_h0_persistence = 0.0
        if len(h0) > 0:
            avg_h0_persistence = float(np.mean(h0[:, 1] - h0[:, 0]))

        long_h2 = sum(1 for b, d in h2 if (d - b) > 1.0)
        long_h1 = sum(1 for b, d in h1 if (d - b) > 1.0)
        short_h0 = sum(1 for b, d in h0 if (d - b) < 0.1)

        recommendations = self._generate_recommendations(long_h2, long_h1, short_h0)

        return {
            "status": "ok",
            "H0": len(h0),
            "H1": len(h1),
            "H2": len(h2),
            "avg_H0_persistence": avg_h0_persistence,
            "long_H2_features": long_h2,
            "long_H1_features": long_h1,
            "short_H0_features": short_h0,
            "recommendations": recommendations,
            "corrections_applied": self._apply_corrections(long_h2, long_h1, short_h0),
        }

    def _apply_corrections(self, long_h2: int, long_h1: int, short_h0: int) -> Dict[str, Any]:
        applied: Dict[str, Any] = {
            "repulsion_inserted": 0,
            "loops_filled": 0,
            "clusters_merged": 0,
        }
        tetrahedra = self.mesh.tetrahedra
        if len(tetrahedra) < 4:
            return applied

        if long_h2 > 0:
            high_weight = sorted(
                tetrahedra.items(), key=lambda x: x[1].weight, reverse=True
            )
            for i in range(min(long_h2, 3)):
                if i < len(high_weight):
                    src = high_weight[i]
                    offset = np.random.normal(0, 0.1, size=3)
                    point = src[1].centroid + offset
                    self.mesh.store(
                        content=f"__gcm_repulsion_{i}__",
                        seed_point=point,
                        labels=["__system__", "__repulsion__"],
                        weight=8.0,
                    )
                    applied["repulsion_inserted"] += 1

        if short_h0 > 5:
            low_weight = sorted(
                tetrahedra.items(), key=lambda x: x[1].weight
            )
            candidates = [
                (tid, t) for tid, t in low_weight
                if t.weight < 0.5 and "__system__" not in t.labels
            ][:short_h0 // 2]
            if candidates:
                result = self.mesh.catalyze_integration_batch(
                    [tid for tid, _ in candidates], strength=2.0
                )
                applied["clusters_merged"] = result.get("catalyzed", 0)

        return applied

    def _generate_recommendations(
        self, long_h2: int, long_h1: int, short_h0: int
    ) -> List[str]:
        recs = []
        if long_h2 > 0:
            recs.append(f"Insert {long_h2} repulsion tetrahedra for H2 caves")
        if long_h1 > 2:
            recs.append(f"Consider filling {long_h1 // 2} H1 loops")
        if short_h0 > 5:
            recs.append(f"Merge {short_h0 // 2} low-persistence H0 clusters")
        if not recs:
            recs.append("Topology is healthy — no corrections needed")
        return recs

    def start_periodic_rebuild(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._rebuild_loop, name="coarse-mesh", daemon=True
        )
        self._thread.start()

    def stop_periodic_rebuild(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _rebuild_loop(self) -> None:
        while not self._stop.wait(timeout=self.rebuild_interval):
            self.rebuild()

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "last_rebuild_time": self._last_rebuild_time,
            "H0": self._last_h0,
            "H1": self._last_h1,
            "H2": self._last_h2,
        }
