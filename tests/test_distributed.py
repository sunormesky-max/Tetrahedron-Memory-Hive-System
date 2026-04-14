"""
Tests for P3: Distributed TetraMem-XL.
"""

import time

import numpy as np
import pytest

from tetrahedron_memory.partitioning import BoundingBox, GhostCell
from tetrahedron_memory.tetra_router import TetraBucket, TetraMeshRouter


class TestTetraBucket:
    def test_store_and_query(self):
        bounds = BoundingBox(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        bucket = TetraBucket("b0", bounds)
        tid = bucket.store("hello", np.zeros(3))
        assert tid is not None
        assert bucket.tetrahedra_count() == 1

    def test_query_returns_results(self):
        bounds = BoundingBox(np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0]))
        bucket = TetraBucket("b1", bounds)
        for i in range(5):
            bucket.store("item_" + str(i), np.array([float(i) * 0.1, 0.0, 0.0]))
        results = bucket.query(np.array([0.2, 0.0, 0.0]), k=3)
        assert len(results) >= 1

    def test_ghost_cells(self):
        bounds = BoundingBox(np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0]))
        bucket = TetraBucket("b2", bounds)
        gc = GhostCell(
            node_id="ghost_1",
            source_bucket_id="other",
            geometry=np.array([0.5, 0.0, 0.0]),
            labels=["test"],
        )
        bucket.ghost_cells["ghost_1"] = gc
        assert len(bucket.ghost_cells) == 1


class TestTetraMeshRouterBasic:
    def test_initialize(self):
        router = TetraMeshRouter(max_tetra_per_bucket=100)
        router.initialize()
        assert len(router.get_all_bucket_ids()) == 1

    def test_route_store(self):
        router = TetraMeshRouter(max_tetra_per_bucket=100)
        router.initialize()
        bid, tid = router.route_store(
            np.zeros(3), "hello world", labels=["test"],
        )
        assert bid is not None
        assert tid is not None
        stats = router.get_statistics()
        assert stats["total_tetrahedra"] == 1

    def test_route_query(self):
        router = TetraMeshRouter(max_tetra_per_bucket=100)
        router.initialize()
        for i in range(10):
            router.route_store(
                np.array([float(i) * 0.1, 0.0, 0.0]),
                "memory_" + str(i),
                labels=["test"],
            )
        results = router.route_query(np.array([0.3, 0.0, 0.0]), k=5)
        assert len(results) >= 1

    def test_cross_bucket_associate(self):
        router = TetraMeshRouter(max_tetra_per_bucket=100)
        router.initialize()
        bid, tid = router.route_store(
            np.zeros(3), "test associate", labels=["assoc"],
        )
        results = router.cross_bucket_associate(bid, tid)
        assert isinstance(results, list)

    def test_auto_split(self):
        router = TetraMeshRouter(max_tetra_per_bucket=5)
        router.initialize()
        for i in range(20):
            router.route_store(
                np.array([float(i) * 0.01, 0.0, 0.0]),
                "dense_" + str(i),
            )
        stats = router.get_statistics()
        assert stats["total_tetrahedra"] >= 15


class TestCrossBucketNavigation:
    def _setup_two_buckets(self):
        router = TetraMeshRouter(max_tetra_per_bucket=100, ghost_ttl=3600.0)
        bounds_a = BoundingBox(np.array([-2.0, -2.0, -2.0]), np.array([0.0, 2.0, 2.0]))
        bounds_b = BoundingBox(np.array([0.0, -2.0, -2.0]), np.array([2.0, 2.0, 2.0]))
        router._create_bucket(bounds_a)
        router._create_bucket(bounds_b)
        return router

    def test_navigate_local_only(self):
        router = TetraMeshRouter(max_tetra_per_bucket=100)
        router.initialize()
        bid, tid = router.route_store(
            np.zeros(3), "start point", labels=["nav"],
        )
        for i in range(5):
            router.route_store(
                np.array([float(i) * 0.05, 0.0, 0.0]),
                "neighbor_" + str(i),
                labels=["nav"],
            )
        results = router.navigate_cross_bucket(bid, tid, max_hops=10)
        assert len(results) >= 1

    def test_navigate_with_ghost_cells(self):
        router = self._setup_two_buckets()
        bids = router.get_all_bucket_ids()

        for i in range(5):
            router.route_store(
                np.array([-0.5 + float(i) * 0.1, 0.0, 0.0]),
                "left_" + str(i), labels=["test"],
            )
        for i in range(5):
            router.route_store(
                np.array([0.5 + float(i) * 0.1, 0.0, 0.0]),
                "right_" + str(i), labels=["test"],
            )

        bid_a = bids[0]
        bucket_a = router.get_bucket(bid_a)
        assert bucket_a is not None
        assert bucket_a.tetrahedra_count() > 0

        first_tid = list(bucket_a.mesh.tetrahedra.keys())[0]

        gc = GhostCell(
            node_id="cross_bridge_1",
            source_bucket_id=bids[1],
            geometry=np.array([0.0, 0.0, 0.0]),
            weight=1.0,
            labels=["test"],
        )
        bucket_a.ghost_cells["cross_bridge_1"] = gc

        results = router.navigate_cross_bucket(bid_a, first_tid, max_hops=10)
        assert isinstance(results, list)

        ghost_results = [r for r in results if r[2] == "ghost_bridge"]
        if ghost_results:
            assert ghost_results[0][3] == bids[1]


class TestDistributedDream:
    def test_dream_with_sufficient_data(self):
        router = TetraMeshRouter(max_tetra_per_bucket=100)
        router.initialize()
        for i in range(15):
            router.route_store(
                np.array([float(i) * 0.1, 0.0, 0.0]),
                "dream source " + str(i),
                labels=["topic", "ai"],
                weight=1.0 + float(i) * 0.1,
            )
        stats = router.distributed_dream(walk_steps=8)
        assert stats["phase"] == "complete"
        assert stats["buckets_processed"] >= 1
        assert stats["local_dreams_created"] >= 0

    def test_dream_no_buckets(self):
        router = TetraMeshRouter(max_tetra_per_bucket=100)
        stats = router.distributed_dream()
        assert stats["phase"] == "no_buckets"


class TestDistributedSelfOrg:
    def test_self_org_with_data(self):
        router = TetraMeshRouter(max_tetra_per_bucket=100)
        router.initialize()
        for i in range(10):
            router.route_store(
                np.array([float(i) * 0.1, 0.0, 0.0]),
                "org_" + str(i),
                labels=["test"],
            )
        stats = router.distributed_self_org(max_iterations=3)
        assert stats["phase"] == "complete"
        assert stats["buckets_processed"] >= 1

    def test_auto_balance(self):
        router = TetraMeshRouter(max_tetra_per_bucket=100)
        router.initialize()
        result = router.auto_balance()
        assert "splits" in result
        assert "buckets" in result


class TestTetraDistributedController:
    def test_init_and_store(self):
        from tetrahedron_memory.tetra_distributed import TetraDistributedController
        ctrl = TetraDistributedController(num_buckets=2, use_ray=False)
        init_result = ctrl.initialize()
        assert init_result["initialized"]
        assert not init_result["ray_mode"]

        bid, tid = ctrl.store(
            "test memory", np.array([1.0, 0.0, 0.0]),
            labels=["test"], weight=1.0,
        )
        assert bid is not None
        assert tid is not None

    def test_multi_store_and_query(self):
        from tetrahedron_memory.tetra_distributed import TetraDistributedController
        ctrl = TetraDistributedController(num_buckets=2, use_ray=False)
        ctrl.initialize()

        for i in range(20):
            ctrl.store(
                "distributed memory " + str(i),
                np.array([float(i) * 0.1, 0.0, 0.0]),
                labels=["ai", "topic"],
                weight=1.0 + float(i) * 0.05,
            )

        results = ctrl.query(np.array([1.0, 0.0, 0.0]), k=5)
        assert len(results) >= 1

        stats = ctrl.get_statistics()
        assert stats["store_count"] == 20
        assert stats["total_tetrahedra"] >= 15

    def test_dream_cycle(self):
        from tetrahedron_memory.tetra_distributed import TetraDistributedController
        ctrl = TetraDistributedController(num_buckets=2, use_ray=False)
        ctrl.initialize()

        for i in range(15):
            ctrl.store(
                "dream input " + str(i),
                np.array([float(i) * 0.1, 0.0, 0.0]),
                labels=["topic"],
                weight=1.0,
            )

        dream_stats = ctrl.run_dream_cycle(walk_steps=8)
        assert dream_stats["phase"] == "complete"

        stats = ctrl.get_statistics()
        assert stats["dream_cycles"] == 1

    def test_self_organization(self):
        from tetrahedron_memory.tetra_distributed import TetraDistributedController
        ctrl = TetraDistributedController(num_buckets=2, use_ray=False)
        ctrl.initialize()

        for i in range(10):
            ctrl.store(
                "org input " + str(i),
                np.array([float(i) * 0.1, 0.0, 0.0]),
                labels=["test"],
            )

        org_stats = ctrl.run_self_organization(max_iterations=3)
        assert org_stats["phase"] == "complete"

    def test_abstract_reorganize_distributed(self):
        from tetrahedron_memory.tetra_distributed import TetraDistributedController
        ctrl = TetraDistributedController(num_buckets=2, use_ray=False)
        ctrl.initialize()

        bid, tid = ctrl.store(
            "base for reorg", np.zeros(3), labels=["reorg"],
        )
        ctrl.store_secondary(bid, tid, "secondary 1", labels=["reorg"], weight=1.0)
        ctrl.store_secondary(bid, tid, "secondary 2", labels=["reorg"], weight=1.0)

        reorg_stats = ctrl.abstract_reorganize(min_density=2)
        assert reorg_stats["buckets_processed"] >= 1

    def test_navigate_cross_bucket(self):
        from tetrahedron_memory.tetra_distributed import TetraDistributedController
        ctrl = TetraDistributedController(num_buckets=2, use_ray=False)
        ctrl.initialize()

        bid, tid = ctrl.store(
            "nav start", np.zeros(3), labels=["nav"],
        )
        for i in range(5):
            ctrl.store(
                "nav_" + str(i),
                np.array([float(i) * 0.1, 0.0, 0.0]),
                labels=["nav"],
            )

        results = ctrl.navigate_cross_bucket(bid, tid, max_hops=10)
        assert isinstance(results, list)

    def test_full_pipeline(self):
        from tetrahedron_memory.tetra_distributed import TetraDistributedController
        ctrl = TetraDistributedController(
            num_buckets=2, max_tetra_per_bucket=500, use_ray=False,
        )
        ctrl.initialize()

        for i in range(30):
            ctrl.store(
                "pipeline memory " + str(i) + " about topic",
                np.array([float(i) * 0.05, 0.0, 0.0]),
                labels=["topic", "pipeline"],
                weight=1.0 + float(i % 5) * 0.2,
            )

        query_results = ctrl.query(np.array([0.5, 0.0, 0.0]), k=5)
        assert len(query_results) >= 1

        dream = ctrl.run_dream_cycle(walk_steps=8)
        assert dream["phase"] == "complete"

        org = ctrl.run_self_organization(max_iterations=3)
        assert org["phase"] == "complete"

        balance = ctrl.auto_balance()
        assert "buckets" in balance

        stats = ctrl.get_statistics()
        assert stats["store_count"] == 30
        assert stats["dream_cycles"] == 1
        assert stats["self_org_cycles"] == 1

        ctrl.shutdown()
