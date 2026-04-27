from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("tetramem.cell")


class TetrahedralCell:
    __slots__ = (
        "cell_id", "bcc_unit_id", "vertex_ids", "vertex_positions",
        "_quality", "_volume", "_jacobian", "_skewness", "_effective_quality",
        "_density", "_label_density",
    )

    def __init__(
        self,
        cell_id: str,
        bcc_unit_id: str,
        vertex_ids: List[str],
        vertex_positions: List[np.ndarray],
    ):
        self.cell_id = cell_id
        self.bcc_unit_id = bcc_unit_id
        self.vertex_ids = vertex_ids
        self.vertex_positions = vertex_positions
        self._quality: float = 0.0
        self._volume: float = 0.0
        self._jacobian: float = 0.0
        self._skewness: float = 0.0
        self._effective_quality: float = 0.0
        self._density: float = 0.0
        self._label_density: float = 0.0
        self._compute_geometry()

    def _compute_geometry(self):
        if len(self.vertex_positions) != 4:
            self._quality = 0.0
            self._volume = 0.0
            self._jacobian = 0.0
            self._skewness = 1.0
            self._effective_quality = 0.0
            return

        p = [np.asarray(v, dtype=np.float64) for v in self.vertex_positions]
        edges = [
            p[1] - p[0], p[2] - p[0], p[3] - p[0],
            p[2] - p[1], p[3] - p[1], p[3] - p[2],
        ]
        edge_lengths = sorted(float(np.linalg.norm(e)) for e in edges)

        mat = np.column_stack([edges[0], edges[1], edges[2]])
        det = float(np.linalg.det(mat))
        self._volume = abs(det) / 6.0

        if abs(det) < 1e-15:
            self._jacobian = 0.0
        else:
            frob = float(np.linalg.norm(mat, 'fro'))
            ideal_ratio = abs(det) / (frob ** 3 + 1e-15)
            self._jacobian = min(1.0, ideal_ratio * 3.7)

        if edge_lengths[-1] < 1e-15:
            self._skewness = 1.0
        else:
            self._skewness = (edge_lengths[-1] - edge_lengths[0]) / (edge_lengths[-1] + 1e-15)

        edge_ratio = edge_lengths[0] / (edge_lengths[-1] + 1e-15)
        self._quality = edge_ratio * self._jacobian * min(1.0, self._volume * 10.0)
        self._quality = min(1.0, max(0.0, self._quality))

        self._effective_quality = self._quality

    @property
    def quality(self) -> float:
        return self._quality

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def jacobian(self) -> float:
        return self._jacobian

    @property
    def skewness(self) -> float:
        return self._skewness

    @property
    def effective_quality(self) -> float:
        return self._effective_quality

    def update_density(self, nodes: Dict[str, Any]) -> None:
        occupied = 0
        labels_seen: Set[str] = set()
        for vid in self.vertex_ids:
            n = nodes.get(vid)
            if n is not None and n.is_occupied:
                occupied += 1
                for lbl in n.labels:
                    if not lbl.startswith("__"):
                        labels_seen.add(lbl)
        self._density = occupied / 4.0
        self._label_density = len(labels_seen) / max(1, occupied) if occupied > 0 else 0.0
        self._effective_quality = self._quality * (1.0 + 0.2 * self._density)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "bcc_unit_id": self.bcc_unit_id,
            "vertex_ids": self.vertex_ids,
            "quality": round(self._quality, 6),
            "volume": round(self._volume, 8),
            "jacobian": round(self._jacobian, 6),
            "skewness": round(self._skewness, 6),
            "effective_quality": round(self._effective_quality, 6),
            "density": round(self._density, 4),
            "label_density": round(self._label_density, 4),
        }


class HoneycombCellMap:
    def __init__(self):
        self._cells: List[TetrahedralCell] = []
        self._bcc_cell_index: Dict[str, List[TetrahedralCell]] = {}
        self._node_cell_index: Dict[str, List[TetrahedralCell]] = {}
        self._built = False

    def build(
        self,
        nodes: Dict[str, Any],
        position_index: Dict[Tuple, str],
        spacing: float,
    ) -> None:
        self._cells.clear()
        self._bcc_cell_index.clear()
        self._node_cell_index.clear()

        body_keys = [k for k in position_index
                     if isinstance(k, tuple) and len(k) == 4 and k[3] == "b"]

        cell_counter = 0
        for bkey in body_keys:
            ix, iy, iz = bkey[0], bkey[1], bkey[2]
            bid = position_index[bkey]

            corners: List[Tuple[Tuple, str]] = []
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        ck = (ix + dx, iy + dy, iz + dz)
                        cnid = position_index.get(ck)
                        if cnid:
                            corners.append((ck, cnid))

            if len(corners) < 4 or not nodes.get(bid):
                continue

            corner_ids = [cid for _, cid in corners]
            body_node = nodes.get(bid)
            if body_node is None:
                continue
            body_pos = body_node.position

            corner_positions = []
            for ck, cid in corners:
                n = nodes.get(cid)
                if n is not None:
                    corner_positions.append((cid, n.position))

            if len(corner_positions) < 4:
                continue

            cids = [cp[0] for cp in corner_positions]
            cposs = [cp[1] for cp in corner_positions]

            tetras = self._subdivide_bcc(bid, body_pos, cids, cposs)
            for tet_verts, tet_pos in tetras:
                cell_id = f"tc_{cell_counter:06x}"
                cell_counter += 1
                cell = TetrahedralCell(
                    cell_id=cell_id,
                    bcc_unit_id=bid,
                    vertex_ids=tet_verts,
                    vertex_positions=tet_pos,
                )
                self._cells.append(cell)
                self._bcc_cell_index.setdefault(bid, []).append(cell)
                for vid in tet_verts:
                    self._node_cell_index.setdefault(vid, []).append(cell)

        self._built = True
        logger.info(
            "HoneycombCellMap: built %d tetrahedral cells across %d BCC units, %d nodes indexed",
            len(self._cells), len(self._bcc_cell_index), len(self._node_cell_index),
        )

    def _subdivide_bcc(
        self,
        body_id: str,
        body_pos: np.ndarray,
        corner_ids: List[str],
        corner_positions: List[np.ndarray],
    ) -> List[Tuple[List[str], List[np.ndarray]]]:
        n = len(corner_ids)
        if n < 4:
            return []

        tetras: List[Tuple[List[str], List[np.ndarray]]] = []

        for i in range(n - 1):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    tri_ids = [body_id, corner_ids[i], corner_ids[j], corner_ids[k]]
                    tri_pos = [body_pos, corner_positions[i], corner_positions[j], corner_positions[k]]
                    tetras.append((tri_ids, tri_pos))

        max_tetras_per_unit = 24
        if len(tetras) > max_tetras_per_unit:
            scored = []
            for tet_verts, tet_pos in tetras:
                mat = np.column_stack([
                    tet_pos[1] - tet_pos[0],
                    tet_pos[2] - tet_pos[0],
                    tet_pos[3] - tet_pos[0],
                ])
                det = abs(float(np.linalg.det(mat)))
                scored.append((det, tet_verts, tet_pos))
            scored.sort(key=lambda x: x[0], reverse=True)
            tetras = [(t[1], t[2]) for t in scored[:max_tetras_per_unit]]

        return tetras

    def get_cells_for_node(self, nid: str) -> List[TetrahedralCell]:
        return self._node_cell_index.get(nid, [])

    def update_all_densities(self, nodes: Dict[str, Any]) -> None:
        for cell in self._cells:
            cell.update_density(nodes)

    def structural_analysis(self, nodes: Dict[str, Any]) -> Dict[str, Any]:
        if not self._cells:
            return {
                "total_cells": 0,
                "bcc_units": 0,
                "occupied_cells": 0,
                "avg_quality": 0.0,
                "avg_volume": 0.0,
                "avg_jacobian": 0.0,
                "avg_skewness": 0.0,
                "density": {},
            }

        qualities = [c.quality for c in self._cells]
        volumes = [c.volume for c in self._cells]
        jacobians = [c.jacobian for c in self._cells]
        skewnesses = [c.skewness for c in self._cells]
        densities = [c._density for c in self._cells]

        occupied = sum(1 for d in densities if d > 0)

        density_dist = {"empty": 0, "sparse": 0, "medium": 0, "dense": 0}
        for d in densities:
            if d == 0:
                density_dist["empty"] += 1
            elif d < 0.25:
                density_dist["sparse"] += 1
            elif d < 0.5:
                density_dist["medium"] += 1
            else:
                density_dist["dense"] += 1

        return {
            "total_cells": len(self._cells),
            "bcc_units": len(self._bcc_cell_index),
            "occupied_cells": occupied,
            "avg_quality": float(np.mean(qualities)),
            "avg_volume": float(np.mean(volumes)),
            "avg_jacobian": float(np.mean(jacobians)),
            "avg_skewness": float(np.mean(skewnesses)),
            "min_quality": float(np.min(qualities)),
            "max_quality": float(np.max(qualities)),
            "avg_effective_quality": float(np.mean([c.effective_quality for c in self._cells])),
            "density": density_dist,
        }

    def get_best_cells(self, n: int) -> List[TetrahedralCell]:
        sorted_cells = sorted(self._cells, key=lambda c: c.effective_quality, reverse=True)
        return sorted_cells[:n]

    def get_cells_by_density(self, n: int) -> List[TetrahedralCell]:
        sorted_cells = sorted(self._cells, key=lambda c: c._density, reverse=True)
        return sorted_cells[:n]

    def find_optimal_placement_cells(
        self,
        nodes: Dict[str, Any],
        label_set: Set[str],
        label_index: Dict[str, Set[str]],
        count: int = 20,
    ) -> List[TetrahedralCell]:
        if not label_set:
            return self.get_best_cells(count)

        candidate_nodes: Set[str] = set()
        for lbl in label_set:
            candidate_nodes.update(label_index.get(lbl, set()))

        scored: List[Tuple[float, TetrahedralCell]] = []
        for cell in self._cells:
            overlap = sum(1 for vid in cell.vertex_ids if vid in candidate_nodes)
            if overlap == 0:
                continue
            score = (overlap / len(cell.vertex_ids)) * cell.quality
            scored.append((score, cell))

        if not scored:
            return self.get_best_cells(count)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:count]]
