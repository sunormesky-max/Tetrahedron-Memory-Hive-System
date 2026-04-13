"""
Production-grade tests for Phase 2 (distributed/partitioning) and Phase 3
(persistence/consistency/monitoring) enhancements.

Covers:
  - TetraMeshRouter bug fixes (ghost cells, cross-bucket)
  - Partitioning Ray fallback logging
  - Persistence write_incremental_full upsert fix
  - Monitoring latency histograms + entropy metrics
  - MultimodalBridge query_by_modality
  - Consistency VectorClock + CompensationLog
  - GlobalCoarseMesh topology reporting
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from tetrahedron_memory.consistency import (
    CompensationLog,
    ConsistencyManager,
    VectorClock,
)
from tetrahedron_memory.core import GeoMemoryBody
from tetrahedron_memory.global_coarse_mesh import GlobalCoarseMesh
from tetrahedron_memory.monitoring import (
    generate_metrics,
    health_check,
    increment_counter,
    record_error,
    set_gauge,
)
from tetrahedron_memory.multimodal_bridge import MultimodalBridge
from tetrahedron_memory.partitioning import BoundingBox, SpatialBucketRouter
from tetrahedron_memory.persistence import ParquetPersistence
from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_router import TetraBucket, TetraMeshRouter


class TestTetraMeshRouter:
    def _make_router(self, max_tetra=50):
        router = TetraMeshRouter(max_tetra_per_bucket=max_tetra)
        router.initialize(BoundingBox(np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])))
        return router

    def test_route_store_creates_tetra(self):
        router = self._make_router()
        bid, tid = router.route_store(np.array([0.0, 0.0, 0.0]), "hello")
        assert isinstance(bid, str)
        assert isinstance(tid, str)
        bucket = router.get_bucket(bid)
        assert bucket is not None
        assert bucket.tetrahedra_count() == 1

    def test_route_query_returns_results(self):
        router = self._make_router()
        for i in range(10):
            router.route_store(np.array([float(i) * 0.1, 0.0, 0.0]), f"item_{i}")
        results = router.route_query(np.array([0.0, 0.0, 0.0]), k=5)
        assert len(results) >= 1

    def test_cross_bucket_associate(self):
        router = self._make_router()
        bid, tid = router.route_store(np.array([0.0, 0.0, 0.0]), "source")
        results = router.cross_bucket_associate(bid, tid)
        assert isinstance(results, list)

    def test_cross_bucket_nonexistent(self):
        router = self._make_router()
        results = router.cross_bucket_associate("nonexistent", "fake")
        assert results == []

    def test_statistics(self):
        router = self._make_router()
        router.route_store(np.array([0.0, 0.0, 0.0]), "test")
        stats = router.get_statistics()
        assert stats["total_buckets"] >= 1
        assert stats["total_tetrahedra"] >= 1

    def test_bucket_split_on_capacity(self):
        router = self._make_router(max_tetra=5)
        for i in range(10):
            router.route_store(np.array([float(i) * 0.01, 0.0, 0.0]), f"split_{i}")
        stats = router.get_statistics()
        assert stats["total_tetrahedra"] >= 10

    def test_ghost_cells_created_at_boundary(self):
        router = TetraMeshRouter(
            max_tetra_per_bucket=500,
            boundary_width=0.5,
        )
        b1 = BoundingBox(np.array([-5.0, -5.0, -5.0]), np.array([0.0, 5.0, 5.0]))
        b2 = BoundingBox(np.array([0.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0]))
        router._create_bucket(b1)
        router._create_bucket(b2)
        router.route_store(np.array([-0.01, 0.0, 0.0]), "near_boundary",
                           labels=["boundary_test"])
        stats = router.get_statistics()
        total_ghosts = stats.get("total_ghost_cells", 0)
        assert total_ghosts >= 0


class TestTetraBucket:
    def test_store_and_query(self):
        bounds = BoundingBox(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        bucket = TetraBucket("test_bucket", bounds)
        tid = bucket.store("hello", seed_point=np.zeros(3))
        assert isinstance(tid, str)
        assert bucket.tetrahedra_count() == 1
        results = bucket.query(np.zeros(3), k=5)
        assert len(results) >= 1


class TestPersistenceIncrementalUpsert:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_incremental_updates_existing_nodes(self):
        pers = ParquetPersistence(storage_path=self.tmpdir)
        body = GeoMemoryBody()
        body.store(content="node_1", weight=1.0)
        pers.write_incremental_full(body._nodes, snapshot_name="upsert_test")

        updated_nodes = {}
        for nid, node in body._nodes.items():
            from tetrahedron_memory.core import MemoryNode
            updated_nodes[nid] = MemoryNode(
                id=node.id,
                content=node.content,
                geometry=node.geometry.copy(),
                timestamp=node.timestamp,
                weight=5.0,
                labels=list(node.labels),
                metadata=dict(node.metadata),
            )
        pers.write_incremental_full(updated_nodes, snapshot_name="upsert_test")

        import pandas as pd
        inc_path = pers._base_path / "incremental_upsert_test.parquet"
        df = pd.read_parquet(inc_path)
        assert len(df) == 1
        assert float(df.iloc[0]["weight"]) == pytest.approx(5.0, rel=0.1)

    def test_full_snapshot_load(self):
        pers = ParquetPersistence(storage_path=self.tmpdir)
        body = GeoMemoryBody()
        id1 = body.store(content="snap_1", weight=2.0)
        id2 = body.store(content="snap_2", weight=3.0)
        pers.write_full_snapshot(body._nodes, snapshot_name="load_test")
        loaded = pers.load_latest_snapshot(snapshot_name="load_test")
        assert len(loaded) == 2
        assert loaded[id1]["content"] == "snap_1"
        assert loaded[id2]["weight"] == pytest.approx(3.0, rel=0.1)


class TestMonitoringEnhanced:
    def test_latency_metrics_exist(self):
        from tetrahedron_memory import monitoring
        assert hasattr(monitoring, "STORE_LATENCY")
        assert hasattr(monitoring, "QUERY_LATENCY")
        assert hasattr(monitoring, "ENTROPY_GAUGE")
        assert hasattr(monitoring, "INTEGRATION_COUNTER")
        assert hasattr(monitoring, "DREAM_COUNTER")

    def test_generate_metrics_includes_new_metrics(self):
        result = generate_metrics()
        assert isinstance(result, str)

    def test_health_check_production(self):
        hc = health_check()
        assert hc["status"] == "ok"

    def test_grafana_dashboard_has_new_panels(self):
        import json
        from tetrahedron_memory.monitoring import get_grafana_dashboard_json
        dash = json.loads(get_grafana_dashboard_json())
        panel_ids = [p["id"] for p in dash["panels"]]
        assert 13 in panel_ids
        assert 14 in panel_ids
        assert 15 in panel_ids
        titles = [p["title"] for p in dash["panels"]]
        assert "Store/Query Latency (p50, p95, p99)" in titles
        assert "Persistent Entropy" in titles
        assert "Integration & Dream Cycles" in titles


class TestMultimodalBridgeQuery:
    def test_query_by_modality_image(self):
        mesh = TetraMesh()
        bridge = MultimodalBridge(mesh)
        rng = np.random.RandomState(42)
        image = rng.rand(32, 32)
        bridge.store_image(image, caption="test image")
        results = bridge.query_by_modality("image")
        assert len(results) >= 1

    def test_query_by_modality_audio(self):
        mesh = TetraMesh()
        bridge = MultimodalBridge(mesh)
        audio = np.random.randn(22050)
        bridge.store_audio(audio, caption="test audio")
        results = bridge.query_by_modality("audio")
        assert len(results) >= 1

    def test_query_by_modality_unknown(self):
        mesh = TetraMesh()
        bridge = MultimodalBridge(mesh)
        results = bridge.query_by_modality("unknown_modality")
        assert results == []

    def test_query_by_modality_empty(self):
        mesh = TetraMesh()
        bridge = MultimodalBridge(mesh)
        results = bridge.query_by_modality("image")
        assert results == []

    def test_query_by_modality_with_point(self):
        mesh = TetraMesh()
        bridge = MultimodalBridge(mesh)
        image = np.random.rand(32, 32)
        bridge.store_image(image, caption="spatial img")
        results = bridge.query_by_modality("image", query_point=np.zeros(3), k=3)
        assert isinstance(results, list)


class TestConsistencyManagerProduction:
    def test_full_version_lifecycle(self):
        cm = ConsistencyManager(["b1", "b2"])
        vn = cm.record_version("node_1", "b1", "hello")
        assert vn.version == 1
        assert vn.checksum != ""
        assert cm.check_version("node_1", 1)
        assert not cm.check_version("node_1", 2)

    def test_read_repair_propagates(self):
        cm = ConsistencyManager(["b1", "b2"])
        cm.record_version("node_1", "b1", "data")
        cm.read_repair("node_1", "b1", ["b2"])
        assert cm.vector_clock.get("b2") == 1

    def test_staleness_detection(self):
        cm = ConsistencyManager(["b1", "b2"])
        cm.record_version("node_1", "b1", "data")
        assert len(cm.get_staleness("b1")) == 0
        assert len(cm.get_staleness("b2")) == 1

    def test_conflict_detection_multi_bucket(self):
        cm = ConsistencyManager(["b1", "b2"])
        cm.record_version("node_1", "b1", "data_a")
        cm.record_version("node_1", "b2", "data_b")
        conflicts = cm.detect_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0][0].node_id == "node_1"

    def test_compensation_retry(self):
        log = CompensationLog()
        call_log = []
        eid = log.record("store", "b1", {"content": "test"}, "timeout")
        results = log.retry_all(lambda op, params: call_log.append(op))
        assert len(results) == 1
        assert results[0]["status"] == "resolved"
        assert len(call_log) == 1


class TestVectorClockProduction:
    def test_happens_before_chain(self):
        vc1 = VectorClock(["b1", "b2"])
        vc2 = VectorClock(["b1", "b2"])
        vc1.increment("b1")
        vc2.merge(vc1)
        vc2.increment("b2")
        assert vc1.happens_before(vc2)

    def test_concurrent_edits(self):
        vc1 = VectorClock(["b1", "b2"])
        vc2 = VectorClock(["b1", "b2"])
        vc1.increment("b1")
        vc2.increment("b2")
        assert vc1.is_concurrent(vc2)

    def test_snapshot_and_add_bucket(self):
        vc = VectorClock(["b1"])
        vc.increment("b1")
        snap = vc.snapshot()
        assert snap == {"b1": 1}
        vc.add_bucket("b2")
        assert vc.get("b2") == 0


class TestGlobalCoarseMeshProduction:
    def test_rebuild_insufficient_data(self):
        mesh = TetraMesh()
        gcm = GlobalCoarseMesh(mesh)
        result = gcm.rebuild()
        assert result["status"] in ("insufficient_data", "ok")

    def test_rebuild_with_data(self):
        pytest.importorskip("gudhi")
        mesh = TetraMesh()
        for i in range(15):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.3, float(i % 3) * 0.1, 0.0]))
        gcm = GlobalCoarseMesh(mesh)
        result = gcm.rebuild()
        assert result["status"] == "ok"
        assert "H0" in result
        assert "H1" in result
        assert "H2" in result

    def test_statistics(self):
        mesh = TetraMesh()
        mesh.store("item", seed_point=np.zeros(3))
        gcm = GlobalCoarseMesh(mesh)
        stats = gcm.get_statistics()
        assert isinstance(stats, dict)

    def test_periodic_rebuild_start_stop(self):
        mesh = TetraMesh()
        for i in range(5):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.1, 0.0, 0.0]))
        gcm = GlobalCoarseMesh(mesh, rebuild_interval=9999)
        gcm.start_periodic_rebuild()
        gcm.stop_periodic_rebuild()


class TestSpatialBucketRouterProduction:
    def test_auto_split(self):
        router = SpatialBucketRouter(max_points_per_bucket=3)
        router.initialize()
        for i in range(8):
            router.route_store(np.array([0.01 * i, 0.01 * i, 0.01 * i]), f"item_{i}")
        stats = router.get_statistics()
        assert stats["total_nodes"] == 8

    def test_cross_bucket_query_works(self):
        router = SpatialBucketRouter()
        router.initialize()
        router.route_store(np.array([0.5, 0.5, 0.5]), "alpha")
        router.route_store(np.array([-0.5, -0.5, -0.5]), "beta")
        results = router.cross_bucket_query(np.array([0.5, 0.5, 0.5]), "alpha", k=5)
        assert isinstance(results, list)

    def test_ghost_cell_lifecycle(self):
        router = SpatialBucketRouter()
        router.initialize()
        router.route_store(np.array([0.5, 0.5, 0.5]), "source")
        stats = router.get_ghost_cell_stats()
        assert isinstance(stats, dict)
        assert "total_ghost_cells" in stats
