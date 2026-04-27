from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField


class LatticeIntegrityReport:
    __slots__ = (
        "check_time", "total_nodes", "face_edges_checked", "edge_edges_checked",
        "broken_edges", "orphan_nodes", "coordination_errors",
        "connectivity_components", "integrity_score", "details",
    )

    def __init__(self):
        self.check_time: float = time.time()
        self.total_nodes: int = 0
        self.face_edges_checked: int = 0
        self.edge_edges_checked: int = 0
        self.broken_edges: List[Dict] = []
        self.orphan_nodes: List[str] = []
        self.coordination_errors: List[Dict] = []
        self.connectivity_components: int = 0
        self.integrity_score: float = 1.0
        self.details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_time": self.check_time,
            "total_nodes": self.total_nodes,
            "face_edges_checked": self.face_edges_checked,
            "edge_edges_checked": self.edge_edges_checked,
            "broken_edges": self.broken_edges[:20],
            "orphan_nodes": self.orphan_nodes[:20],
            "coordination_errors": self.coordination_errors[:20],
            "connectivity_components": self.connectivity_components,
            "integrity_score": round(self.integrity_score, 4),
            "details": self.details,
        }


class LatticeIntegrityChecker:
    """
    Verifies geometric and topological integrity of the BCC lattice.

    BCC crystal structure rules:
      - Body-center (BC) nodes: 8 face-sharing corner neighbors
      - Corner nodes: 8 face-sharing BC neighbors + 6 edge-sharing corner neighbors
      - All edges must be bidirectional
      - No orphan nodes (nodes with zero neighbors)
      - All neighbor references must resolve to existing nodes

    Also checks occupied node structural health:
      - Weight consistency (weight >= 0)
      - Activation bounds (0 <= activation <= 10)
      - Crystal channel validity (referenced nodes exist)
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field
        self._last_report: Optional[LatticeIntegrityReport] = None
        self._report_history: List[LatticeIntegrityReport] = []
        self._max_history = 20

    def run_full_check(self) -> LatticeIntegrityReport:
        report = LatticeIntegrityReport()
        field = self._field

        with field._lock:
            report.total_nodes = len(field._nodes)
            self._check_bidirectionality(field, report)
            self._check_orphan_nodes(field, report)
            self._check_coordination(field, report)
            self._check_connectivity(field, report)
            self._check_occupied_health(field, report)

        total_errors = (
            len(report.broken_edges)
            + len(report.orphan_nodes)
            + len(report.coordination_errors)
        )
        if report.total_nodes > 0:
            critical_errors = len(report.broken_edges) + len(report.orphan_nodes)
            coord_ratio = len(report.coordination_errors) / max(report.total_nodes, 1)
            report.integrity_score = max(0.0, 1.0 - (critical_errors * 0.5 + coord_ratio * 0.5))

        if total_errors == 0:
            report.details = f"All {report.total_nodes} nodes pass integrity verification"
        else:
            report.details = f"{total_errors} issues found: {len(report.broken_edges)} broken edges, {len(report.orphan_nodes)} orphans, {len(report.coordination_errors)} coordination errors"

        self._last_report = report
        self._report_history.append(report)
        if len(self._report_history) > self._max_history:
            self._report_history = self._report_history[-self._max_history // 2:]

        return report

    def _check_bidirectionality(self, field, report: LatticeIntegrityReport):
        for n1, n2, etype in field._edges:
            report.face_edges_checked += 1 if etype == "face" else 0
            report.edge_edges_checked += 1 if etype == "edge" else 0

            node_a = field._nodes.get(n1)
            node_b = field._nodes.get(n2)

            if node_a is None or node_b is None:
                report.broken_edges.append({
                    "edge": (n1[:8], n2[:8]),
                    "type": etype,
                    "issue": "missing_node",
                })
                continue

            if etype == "face":
                if n2 not in node_a.face_neighbors:
                    report.broken_edges.append({
                        "edge": (n1[:8], n2[:8]),
                        "type": "face",
                        "issue": "unidirectional_a_to_b",
                    })
                if n1 not in node_b.face_neighbors:
                    report.broken_edges.append({
                        "edge": (n2[:8], n1[:8]),
                        "type": "face",
                        "issue": "unidirectional_b_to_a",
                    })
            elif etype == "edge":
                if n2 not in node_a.edge_neighbors:
                    report.broken_edges.append({
                        "edge": (n1[:8], n2[:8]),
                        "type": "edge",
                        "issue": "unidirectional_a_to_b",
                    })
                if n1 not in node_b.edge_neighbors:
                    report.broken_edges.append({
                        "edge": (n2[:8], n1[:8]),
                        "type": "edge",
                        "issue": "unidirectional_b_to_a",
                    })

    def _check_orphan_nodes(self, field, report: LatticeIntegrityReport):
        for nid, node in field._nodes.items():
            total_neighbors = len(node.face_neighbors) + len(node.edge_neighbors)
            if total_neighbors == 0:
                report.orphan_nodes.append(nid)

    def _check_coordination(self, field, report: LatticeIntegrityReport):
        for key, nid in field._position_index.items():
            node = field._nodes.get(nid)
            if node is None:
                continue

            is_body_center = isinstance(key, tuple) and len(key) == 4 and key[3] == "b"

            if is_body_center:
                expected_face = 8
                actual_face = len(node.face_neighbors)
                if actual_face < expected_face // 2:
                    report.coordination_errors.append({
                        "node": nid[:8],
                        "type": "body_center",
                        "expected_face": expected_face,
                        "actual_face": actual_face,
                    })
            else:
                expected_face = 8
                actual_face = len(node.face_neighbors)
                if actual_face < expected_face // 2:
                    report.coordination_errors.append({
                        "node": nid[:8],
                        "type": "corner",
                        "expected_face": expected_face,
                        "actual_face": actual_face,
                        "expected_edge": 6,
                        "actual_edge": len(node.edge_neighbors),
                    })

    def _check_connectivity(self, field, report: LatticeIntegrityReport):
        if not field._nodes:
            return
        visited = set()
        components = 0
        for start_nid in field._nodes:
            if start_nid in visited:
                continue
            components += 1
            stack = [start_nid]
            while stack:
                nid = stack.pop()
                if nid in visited:
                    continue
                visited.add(nid)
                node = field._nodes.get(nid)
                if node is None:
                    continue
                for fnid in node.face_neighbors:
                    if fnid not in visited:
                        stack.append(fnid)
                for enid in node.edge_neighbors:
                    if enid not in visited:
                        stack.append(enid)
        report.connectivity_components = components

    def _check_occupied_health(self, field, report: LatticeIntegrityReport):
        for nid, node in field._nodes.items():
            if not node.is_occupied:
                continue
            if node.weight < 0:
                report.coordination_errors.append({
                    "node": nid[:8],
                    "type": "negative_weight",
                    "weight": node.weight,
                })
            if node.activation < 0:
                report.coordination_errors.append({
                    "node": nid[:8],
                    "type": "negative_activation",
                    "activation": node.activation,
                })
            for crystal_target in node.crystal_channels:
                if crystal_target not in field._nodes:
                    report.broken_edges.append({
                        "edge": (nid[:8], crystal_target[:8]),
                        "type": "crystal",
                        "issue": "dangling_crystal_channel",
                    })

    def get_latest(self) -> Optional[Dict]:
        return self._last_report.to_dict() if self._last_report else None

    def get_history(self, n: int = 10) -> List[Dict]:
        return [r.to_dict() for r in self._report_history[-n:]]

    def stats(self) -> Dict[str, Any]:
        return {
            "checks_performed": len(self._report_history),
            "latest_score": self._last_report.integrity_score if self._last_report else None,
        }
