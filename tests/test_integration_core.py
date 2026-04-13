"""
Tests for the three high-priority integrations:
  1. ConsistencyManager wired into core (GeoMemoryBody / TetraMeshRouter / BucketActor)
  2. Persistence wired into GeoMemoryBody
  3. Self-emergence daemon (autonomous dream/self-org/integration)

Also covers:
  4. GlobalCoarseMesh feedback loop (auto-applies corrections)
  5. REST API new endpoints (smoke, no FastAPI required)
  6. LLM Tool new tools
"""

import os
import shutil
import tempfile
import threading
import time

import numpy as np
import pytest

from tetrahedron_memory.consistency import ConsistencyManager, VectorClock
from tetrahedron_memory.core import GeoMemoryBody
from tetrahedron_memory.persistence import ParquetPersistence
from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_router import TetraMeshRouter
from tetrahedron_memory.llm_tool import execute_tool_call, TOOL_DEFINITIONS


class TestConsistencyManagerInCore:
    def test_store_records_version(self):
        cm = ConsistencyManager(["default"])
        body = GeoMemoryBody(consistency=cm)
        mid = body.store("hello world", labels=["test"])
        versions = cm._version_store.get(mid)
        assert versions is not None
        assert len(versions) == 1
        assert versions[0].bucket_id == "default"
        assert versions[0].version == 1

    def test_multiple_stores_increment_versions(self):
        cm = ConsistencyManager(["bucket_a"])
        body = GeoMemoryBody(bucket_id="bucket_a", consistency=cm)
        id1 = body.store("memory one")
        id2 = body.store("memory two")
        assert cm._version_store[id1][0].version == 1
        assert cm._version_store[id2][0].version == 1
        assert cm.vector_clock.get("bucket_a") == 2

    def test_vector_clock_advances(self):
        cm = ConsistencyManager(["b1", "b2"])
        body = GeoMemoryBody(bucket_id="b1", consistency=cm)
        body.store("item 1")
        body.store("item 2")
        assert cm.vector_clock.get("b1") == 2
        assert cm.vector_clock.get("b2") == 0

    def test_get_consistency_status_enabled(self):
        cm = ConsistencyManager(["default"])
        body = GeoMemoryBody(consistency=cm)
        status = body.get_consistency_status()
        assert status["enabled"] is True
        assert status["bucket_id"] == "default"
        assert status["conflicts"] == 0

    def test_get_consistency_status_disabled(self):
        body = GeoMemoryBody()
        status = body.get_consistency_status()
        assert status["enabled"] is False

    def test_cross_bucket_conflict_detection(self):
        cm = ConsistencyManager(["b1", "b2"])
        cm.record_version("node_x", "b1", "content_v1")
        cm.record_version("node_x", "b2", "content_v2")
        conflicts = cm.detect_conflicts()
        assert len(conflicts) >= 1

    def test_router_consistency_integration(self):
        cm = ConsistencyManager(["bucket_0"])
        router = TetraMeshRouter(consistency=cm)
        router.initialize()
        bid, tid = router.route_store(
            np.array([0.5, 0.5, 0.5]), "test content"
        )
        versions = cm._version_store.get(tid)
        assert versions is not None
        assert versions[0].bucket_id == bid

    def test_router_conflict_read_repair_on_query(self):
        cm = ConsistencyManager(["bucket_0"])
        router = TetraMeshRouter(consistency=cm)
        router.initialize()
        router.route_store(np.array([0.0, 0.0, 0.0]), "mem1")
        router.route_store(np.array([0.1, 0.1, 0.1]), "mem2")
        results = router.route_query(np.array([0.0, 0.0, 0.0]), k=2)
        assert len(results) >= 1


class TestPersistenceInCore:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.persister = ParquetPersistence(storage_path=self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_flush_persistence_saves_nodes(self):
        body = GeoMemoryBody(persistence=self.persister)
        body.store("alpha")
        body.store("beta")
        body.flush_persistence()
        loaded = self.persister.load_latest_snapshot(snapshot_name="default")
        assert len(loaded) >= 2

    def test_auto_persist_triggers_after_threshold(self):
        body = GeoMemoryBody(persistence=self.persister)
        body._persist_flush_interval = 5
        for i in range(6):
            body.store(f"item_{i}")
        loaded = self.persister.load_latest_snapshot(snapshot_name="default")
        assert len(loaded) >= 5

    def test_load_from_persistence(self):
        body = GeoMemoryBody(persistence=self.persister)
        for i in range(5):
            body.store(f"saved_{i}", labels=[f"label_{i}"])
        body.flush_persistence()

        body2 = GeoMemoryBody(persistence=self.persister)
        count = body2.load_from_persistence()
        assert count == 5
        stats = body2.get_statistics()
        assert stats["total_memories"] >= 5

    def test_load_from_persistence_none(self):
        body = GeoMemoryBody()
        count = body.load_from_persistence()
        assert count == 0

    def test_flush_persistence_no_persistence(self):
        body = GeoMemoryBody()
        body.flush_persistence()

    def test_router_persistence_integration(self):
        persister = ParquetPersistence(storage_path=self.tmpdir)
        router = TetraMeshRouter(persistence=persister)
        router.initialize()
        router.route_store(np.array([0.0, 0.0, 0.0]), "persisted memory")
        bucket_ids = router.get_all_bucket_ids()
        assert len(bucket_ids) >= 1
        loaded = persister.load_bucket(bucket_ids[0])
        assert len(loaded) >= 1


class TestSelfEmergenceDaemon:
    def test_start_stop_daemon(self):
        body = GeoMemoryBody(auto_emerge_interval=0.5)
        for i in range(10):
            body.store(f"emerge_{i}")
        assert body.is_emergence_running()
        body.stop_emergence_daemon()
        assert not body.is_emergence_running()

    def test_daemon_does_not_crash(self):
        body = GeoMemoryBody(auto_emerge_interval=0.2)
        for i in range(20):
            body.store(f"content_{i}")
        time.sleep(0.6)
        assert body.is_emergence_running()
        body.stop_emergence_daemon()

    def test_zero_interval_no_daemon(self):
        body = GeoMemoryBody(auto_emerge_interval=0)
        assert not body.is_emergence_running()

    def test_emergence_with_dream(self):
        body = GeoMemoryBody(auto_emerge_interval=0.3)
        for i in range(15):
            body.store(f"deep_content_{i}", labels=["emerge"])
        time.sleep(0.5)
        body.stop_emergence_daemon()
        stats = body.get_statistics()
        assert stats["total_memories"] >= 15


class TestGlobalCoarseMeshFeedback:
    def test_corrections_applied_in_report(self):
        from tetrahedron_memory.global_coarse_mesh import GlobalCoarseMesh
        mesh = TetraMesh()
        for i in range(30):
            angle = 2.0 * np.pi * i / 30
            point = np.array([np.cos(angle), np.sin(angle), 0.01 * i])
            mesh.store(f"gcm_{i}", seed_point=point, weight=1.0)
        gcm = GlobalCoarseMesh(mesh)
        gcm.rebuild()
        report = gcm.get_topology_report()
        assert report["status"] == "ok"
        assert "corrections_applied" in report
        assert isinstance(report["corrections_applied"], dict)

    def test_repulsion_inserted_for_h2(self):
        from tetrahedron_memory.global_coarse_mesh import GlobalCoarseMesh
        mesh = TetraMesh()
        for i in range(50):
            point = np.array([0.01 * i, 0.01 * i, 0.01 * i])
            mesh.store(f"h2_{i}", seed_point=point, weight=1.0)
        gcm = GlobalCoarseMesh(mesh)
        gcm.rebuild()
        report = gcm.get_topology_report()
        corr = report.get("corrections_applied", {})
        assert "repulsion_inserted" in corr
        assert "clusters_merged" in corr


class TestLLMToolNewTools:
    def test_tool_definitions_count(self):
        assert len(TOOL_DEFINITIONS) >= 10

    def test_dream_tool_definition_exists(self):
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        assert "tetramem_dream" in names
        assert "tetramem_closed_loop" in names
        assert "tetramem_weight_update" in names
        assert "tetramem_batch_store" in names
        assert "tetramem_persist" in names

    def test_execute_dream_tool(self):
        body = GeoMemoryBody()
        for i in range(5):
            body.store(f"dream_src_{i}")
        result = execute_tool_call("tetramem_dream", {}, memory=body)
        assert isinstance(result, dict)

    def test_execute_closed_loop_tool(self):
        body = GeoMemoryBody()
        for i in range(5):
            body.store(f"cl_{i}")
        result = execute_tool_call(
            "tetramem_closed_loop",
            {"context": "test", "k": 3},
            memory=body,
        )
        assert isinstance(result, dict)
        assert "phase" in result

    def test_execute_weight_update_tool(self):
        body = GeoMemoryBody()
        mid = body.store("weight me", weight=1.0)
        result = execute_tool_call(
            "tetramem_weight_update",
            {"memory_id": mid, "delta": 2.0},
            memory=body,
        )
        assert result["status"] == "ok"

    def test_execute_batch_store_tool(self):
        body = GeoMemoryBody()
        result = execute_tool_call(
            "tetramem_batch_store",
            {
                "items": [
                    {"content": "batch_a", "weight": 1.0},
                    {"content": "batch_b", "weight": 2.0},
                    {"content": "batch_c", "labels": ["x"]},
                ]
            },
            memory=body,
        )
        assert result["stored"] == 3
        assert len(result["ids"]) == 3

    def test_execute_persist_tool(self):
        body = GeoMemoryBody()
        body.store("persist me")
        result = execute_tool_call("tetramem_persist", {}, memory=body)
        assert result["status"] == "ok"

    def test_execute_persist_tool_with_persistence(self):
        tmpdir = tempfile.mkdtemp()
        try:
            persister = ParquetPersistence(storage_path=tmpdir)
            body = GeoMemoryBody(persistence=persister)
            body.store("persisted")
            result = execute_tool_call("tetramem_persist", {}, memory=body)
            assert result["status"] == "ok"
            loaded = persister.load_latest_snapshot(snapshot_name="default")
            assert len(loaded) >= 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestBucketActorWithConsistencyPersistence:
    def test_bucket_actor_with_consistency(self):
        from tetrahedron_memory.partitioning import BucketActor
        cm = ConsistencyManager(["actor_1"])
        actor = BucketActor("actor_1", consistency=cm)
        mid = actor.store("actor content", labels=["test"])
        versions = cm._version_store.get(mid)
        assert versions is not None

    def test_bucket_actor_with_persistence(self):
        from tetrahedron_memory.partitioning import BucketActor
        tmpdir = tempfile.mkdtemp()
        try:
            persister = ParquetPersistence(storage_path=tmpdir)
            actor = BucketActor("actor_p", persistence=persister)
            actor.store("persistent content")
            actor._ensure_body().flush_persistence()
            loaded = persister.load_latest_snapshot(snapshot_name="actor_p")
            assert len(loaded) >= 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
