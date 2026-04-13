"""Basic usage demo: store, query, associate, self-organize, and integration catalyst."""

from tetrahedron_memory import GeoMemoryBody


def main():
    memory = GeoMemoryBody(dimension=3, precision="fast")

    # Store several memories
    ids = []
    for content, labels, weight in [
        ("Alpha Complex builds simplicial complexes from point clouds", ["topology", "alpha"], 1.0),
        (
            "Persistent Homology tracks birth and death of topological features",
            ["topology", "ph"],
            1.2,
        ),
        ("Memory nodes live on the unit sphere in 3D space", ["geometry"], 0.8),
        ("Association rules link memories across simplices", ["association"], 0.9),
        (
            "Weighted Alpha Complex uses power distance for persistence",
            ["topology", "weighted"],
            1.1,
        ),
    ]:
        node_id = memory.store(content=content, labels=labels, weight=weight)
        ids.append(node_id)
        print(f"Stored: {node_id[:8]}... | {content[:50]}")

    print(f"\n--- {len(ids)} memories stored ---\n")

    # Query
    results = memory.query("topological features", k=3)
    print("Query 'topological features' (top 3):")
    for r in results:
        print(f"  [{r.distance:.4f}] score={r.persistence_score:.4f} | {r.node.content[:60]}")

    # Associate
    print("\nAssociations from first memory:")
    assoc = memory.associate(memory_id=ids[0], max_depth=2)
    for node, score, atype in assoc[:5]:
        print(f"  [{atype}] score={score:.3f} | {node.content[:50]}")

    # Self-organize
    stats = memory.self_organize()
    print(
        f"\nSelf-organize: actions={stats['actions']}, repulsions={stats['repulsions']}, caves={stats['cave_growths']}"
    )

    # Integration catalyst (replaces decay — memories are eternal)
    integration = memory.global_catalyze_integration(strength=1.0)
    print(f"Integration: catalyzed={integration['catalyzed']}, total={integration['total']}")

    # Stats
    print(f"\nFinal: {len(memory._nodes)} nodes in memory")


if __name__ == "__main__":
    main()
