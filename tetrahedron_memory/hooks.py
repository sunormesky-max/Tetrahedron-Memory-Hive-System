"""
TetraMem-XL session hook — auto-loads core memories on startup.
"""

import os

from tetrahedron_memory.core import GeoMemoryBody

_memory = None
STORAGE_DIR = os.environ.get("TETRAMEM_STORAGE", os.path.expanduser("~/.tetramem_data"))


def get_memory():
    """Get or create the global memory body instance."""
    global _memory
    if _memory is None:
        from tetrahedron_memory.persistence import MemoryPersistence

        _memory = GeoMemoryBody(dimension=3, precision='fast')

        persistence = MemoryPersistence(storage_dir=STORAGE_DIR)
        saved_nodes = persistence.load_nodes()

        if saved_nodes:
            for node_id, node in saved_nodes.items():
                if _memory._use_mesh:
                    _memory._mesh.store(
                        content=node.content,
                        seed_point=node.geometry,
                        labels=node.labels,
                        metadata=node.metadata,
                        weight=node.weight,
                    )
                    _memory._mesh_node_map[node_id] = node.geometry
                else:
                    _memory._nodes_dict[node_id] = node
                    for label in node.labels:
                        _memory._label_index_legacy[label].add(node_id)
            _memory._needs_rebuild = True
            print(f"[TetraMem Hook] Loaded {len(_memory._nodes)} memories from {STORAGE_DIR}")
        else:
            print("[TetraMem Hook] No existing memories found, starting empty")

    return _memory

