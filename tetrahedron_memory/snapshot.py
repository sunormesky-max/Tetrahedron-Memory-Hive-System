import numpy as np
from typing import Dict, List, Optional


class FieldSnapshot:
    __slots__ = (
        "positions", "weights", "activations", "labels", "contents",
        "occupied_mask", "node_ids", "neighbor_map", "timestamp",
    )

    def __init__(self, field):
        self.timestamp = field._start_time if hasattr(field, '_start_time') else 0
        nodes = field._nodes
        n = len(nodes)
        self.node_ids = []
        self.positions = np.zeros((n, 3), dtype=np.float32)
        self.weights = np.zeros(n, dtype=np.float32)
        self.activations = np.zeros(n, dtype=np.float32)
        self.occupied_mask = np.zeros(n, dtype=bool)
        self.labels = {}
        self.contents = {}
        self.neighbor_map = {}

        id_to_idx = {}
        for i, (nid, node) in enumerate(nodes.items()):
            id_to_idx[nid] = i
            self.node_ids.append(nid)
            self.positions[i] = node.position
            self.weights[i] = node.weight
            self.activations[i] = node.activation
            if node.is_occupied:
                self.occupied_mask[i] = True
                self.contents[nid] = node.content
                self.labels[nid] = list(node.labels)
            self.neighbor_map[nid] = list(node.face_neighbors) + list(node.edge_neighbors)

        self._id_to_idx = id_to_idx

    def get_occupied_count(self) -> int:
        return int(self.occupied_mask.sum())

    def get_node_weight(self, node_id: str) -> float:
        idx = self._id_to_idx.get(node_id)
        if idx is not None:
            return float(self.weights[idx])
        return 0.0

    def get_neighbors(self, node_id: str) -> List[str]:
        return self.neighbor_map.get(node_id, [])

    def find_in_radius(self, center_idx: int, radius: float) -> List[int]:
        center = self.positions[center_idx]
        dists = np.linalg.norm(self.positions - center, axis=1)
        return np.where((dists <= radius) & self.occupied_mask)[0].tolist()
