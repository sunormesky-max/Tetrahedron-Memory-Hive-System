"""
Command-line interface for Tetrahedron Memory System.

Stateful CLI with persistence — memories survive across invocations.
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

from tetrahedron_memory.core import GeoMemoryBody

STORAGE_DIR = os.environ.get("TETRAMEM_STORAGE", os.path.expanduser("~/.tetramem_data"))


def _load_memory() -> GeoMemoryBody:
    memory = GeoMemoryBody(dimension=3, precision="fast")
    try:
        from tetrahedron_memory.persistence import MemoryPersistence

        persistence = MemoryPersistence(storage_dir=STORAGE_DIR)
        saved_nodes = persistence.load_nodes()
        if saved_nodes:
            for node_id, node in saved_nodes.items():
                if memory._use_mesh:
                    memory._mesh.store(
                        content=node.content,
                        seed_point=node.geometry,
                        labels=node.labels,
                        metadata=node.metadata,
                        weight=node.weight,
                    )
                    memory._mesh_node_map[node_id] = node.geometry
                else:
                    memory._nodes_dict[node_id] = node
                    for label in node.labels:
                        memory._label_index_legacy[label].add(node_id)
            memory._needs_rebuild = True
    except Exception as e:
        logging.getLogger("tetramem.cli").warning("Failed to load persisted memory: %s", e)
    return memory


def _save_memory(memory: GeoMemoryBody) -> None:
    try:
        from tetrahedron_memory.persistence import MemoryPersistence

        persistence = MemoryPersistence(storage_dir=STORAGE_DIR)
        nodes = dict(memory._nodes)
        if nodes:
            persistence.save_nodes(nodes)
    except Exception as e:
        logging.getLogger("tetramem.cli").warning("Failed to save memory: %s", e)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tetramem", description="Tetrahedron Memory System - Geometric-driven AI memory"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    store_parser = subparsers.add_parser("store", help="Store a memory")
    store_parser.add_argument("content", help="Memory content to store")
    store_parser.add_argument("-l", "--labels", nargs="*", help="Labels for the memory")
    store_parser.add_argument("-w", "--weight", type=float, default=1.0, help="Initial weight")

    query_parser = subparsers.add_parser("query", help="Query memories")
    query_parser.add_argument("query_text", help="Query text")
    query_parser.add_argument("-k", type=int, default=5, help="Number of results")

    label_parser = subparsers.add_parser("label", help="Query by label")
    label_parser.add_argument("label", help="Label to search")
    label_parser.add_argument("-k", type=int, default=10, help="Number of results")

    subparsers.add_parser("stats", help="Show statistics")

    subparsers.add_parser("clear", help="Clear all memories")

    subparsers.add_parser("persist", help="Flush memories to disk")

    dream_parser = subparsers.add_parser("dream", help="Trigger a dream cycle")
    dream_parser.add_argument("-n", "--count", type=int, default=1, help="Number of dream cycles")

    subparsers.add_parser("self-org", help="Run self-organization")

    subparsers.add_parser("catalyze", help="Global catalyze integration")

    subparsers.add_parser("status", help="Show emergence/consistency status")

    mp_parser = subparsers.add_parser("mquery", help="Multi-parameter filtered query")
    mp_parser.add_argument("query_text", help="Query text")
    mp_parser.add_argument("-k", type=int, default=10, help="Number of results")
    mp_parser.add_argument("--labels", nargs="*", help="Required labels")
    mp_parser.add_argument("--preferred", nargs="*", help="Preferred labels")

    subparsers.add_parser("build-pyramid", help="Build resolution pyramid")

    py_parser = subparsers.add_parser("pyquery", help="Query via resolution pyramid")
    py_parser.add_argument("query_text", help="Query text")
    py_parser.add_argument("-k", type=int, default=5, help="Number of results")
    py_parser.add_argument("--level", type=int, default=-1, help="Pyramid level (-1=auto)")

    subparsers.add_parser("zigzag", help="Record zigzag persistence snapshot")

    subparsers.add_parser("predict", help="Predict topological changes")

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    memory = _load_memory()

    try:
        if args.command == "store":
            memory_id = memory.store(content=args.content, labels=args.labels, weight=args.weight)
            _save_memory(memory)
            print(memory_id)

        elif args.command == "query":
            results = memory.query(args.query_text, k=args.k)
            if not results:
                print("No results found")
            else:
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result.node.content}")
                    print(f"   Distance: {result.distance:.4f}")
                    print(f"   Score: {result.persistence_score:.4f}")

        elif args.command == "label":
            results = memory.query_by_label(args.label, k=args.k)
            if not results:
                print(f"No memories with label: {args.label}")
            else:
                for i, node in enumerate(results, 1):
                    print(f"{i}. {node.content}")
                    print(f"   Weight: {node.weight:.4f}")

        elif args.command == "stats":
            stats = memory.get_statistics()
            print(json.dumps(stats, indent=2))

        elif args.command == "clear":
            memory.clear()
            _save_memory(memory)
            print("All memories cleared")

        elif args.command == "persist":
            _save_memory(memory)
            print(f"Persisted {len(memory._nodes)} memories to {STORAGE_DIR}")

        elif args.command == "dream":
            if not memory._use_mesh:
                print("Dream cycle requires mesh mode")
                return 1
            from tetrahedron_memory.tetra_dream import TetraDreamCycle

            dc = TetraDreamCycle(memory._mesh)
            for i in range(args.count):
                result = dc.trigger_now()
                if result:
                    print(
                        f"Dream {i + 1}: fused {result.get('path_length', 0)} nodes -> {result.get('new_id', 'N/A')}"
                    )
                else:
                    print(f"Dream {i + 1}: no fusion produced")
            _save_memory(memory)

        elif args.command == "self-org":
            result = memory.self_organize()
            print(json.dumps(result, indent=2, default=str))
            _save_memory(memory)

        elif args.command == "catalyze":
            result = memory.global_catalyze_integration(strength=1.0)
            print(json.dumps(result, indent=2))
            _save_memory(memory)

        elif args.command == "status":
            info = {
                "memories": len(memory._nodes),
                "mesh_mode": memory._use_mesh,
                "tetrahedra": len(memory._mesh.tetrahedra) if memory._use_mesh else 0,
                "emergence_running": memory.is_emergence_running(),
                "consistency": memory.get_consistency_status(),
                "adaptive_threshold": memory._adaptive_threshold.get_status(),
                "storage_dir": STORAGE_DIR,
            }
            print(json.dumps(info, indent=2, default=str))

        elif args.command == "mquery":
            results = memory.query_multiparam(
                query_text=args.query_text,
                k=args.k,
                labels_required=args.labels,
                labels_preferred=args.preferred,
            )
            if not results:
                print("No results found")
            else:
                for i, r in enumerate(results, 1):
                    print(f"{i}. {r.node.content}")
                    print(f"   Score: {r.persistence_score:.4f} | Type: {r.association_type}")

        elif args.command == "build-pyramid":
            result = memory.build_pyramid()
            print(json.dumps(result, indent=2))

        elif args.command == "pyquery":
            results = memory.query_pyramid(args.query_text, k=args.k, level=args.level)
            if not results:
                print("No results found")
            else:
                for i, r in enumerate(results, 1):
                    print(f"{i}. {r.node.content}")
                    print(f"   Distance: {r.distance:.4f} | Type: {r.association_type}")

        elif args.command == "zigzag":
            result = memory.record_zigzag_snapshot()
            if result:
                print(json.dumps(result, indent=2))
            else:
                print("Zigzag requires mesh mode")

        elif args.command == "predict":
            pred = memory.predict_topology()
            print(json.dumps(pred, indent=2))

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
