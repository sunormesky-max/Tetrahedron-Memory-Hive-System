"""Distributed architecture demo: BucketActor pool, SpatialBucketRouter, cross-bucket sync."""



def main():
    from tetrahedron_memory.partitioning import (
        BucketActor,
        SpatialBucketRouter,
        TetraMemRayController,
        global_coarse_grid_sync,
        register_bucket,
    )

    # --- Local mode: pool of bucket actors ---
    ctrl = TetraMemRayController(num_buckets=4)
    ctrl.initialize()
    print(f"Initialized {ctrl.num_buckets} bucket actors")

    ctrl.store("bucket_0", content="Memory in bucket 0", labels=["test"])
    ctrl.store("bucket_1", content="Memory in bucket 1", labels=["test"])
    ctrl.store("bucket_0", content="Another memory in bucket 0", labels=["extra"])
    print("Stored 3 memories across 2 buckets")

    ctrl.self_organize("bucket_0")
    print("Self-organized bucket_0")

    diagram = ctrl.sync_all()
    print(f"Global sync: {len(diagram)} persistence pairs")

    # --- Spatial Bucket Router ---
    from tetrahedron_memory.core import GeoMemoryBody

    router = SpatialBucketRouter(num_buckets=4)
    router.initialize(GeoMemoryBody)
    print(f"\nSpatialBucketRouter: {router.num_buckets} buckets")

    node_id = router.route_store(content="Routed memory", labels=["spatial"], weight=1.0)
    print(f"Stored via router: {node_id[:8]}...")

    results = router.route_query("routed", k=3)
    print(f"Query via router: {len(results)} results")

    # --- Global registry ---
    actor = BucketActor("demo_bucket")
    register_bucket("demo_bucket", actor)
    actor.store(content="Global registry memory", labels=["global"])
    print("\nRegistered 'demo_bucket' in global registry")

    diagram2 = global_coarse_grid_sync()
    print(f"Coarse grid sync: {len(diagram2)} persistence pairs")


if __name__ == "__main__":
    main()
