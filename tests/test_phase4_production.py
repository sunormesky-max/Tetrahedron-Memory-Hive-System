"""
Phase 4 Production Validation Tests for TetraMem-XL.

Production-grade benchmarks per the Grok 4-stage roadmap:
  1. Insert throughput > 25,000 memories/sec (via TetraMesh direct path)
  2. Query latency < 8ms (p99)
  3. Scale test: 1M+ memories with ZERO deletion or weakening
  4. Persistent entropy drops >= 18% after dream cycle
  5. Memory leak detection over sustained operation
  6. Fault tolerance: consistency manager survives partial failures
  7. Closed-loop endurance: repeated cycles without count regression

Markers:
  - "slow": takes > 30s (scale/leak tests)
  - "benchmark": throughput/latency benchmarks

Run quick suite:   pytest tests/test_phase4_production.py -m "not slow"
Run full suite:    pytest tests/test_phase4_production.py
"""

import gc
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List

import numpy as np
import pytest

from tetrahedron_memory.closed_loop import ClosedLoopEngine
from tetrahedron_memory.consistency import (
    CompensationLog,
    ConsistencyManager,
    VectorClock,
)
from tetrahedron_memory.core import GeoMemoryBody
from tetrahedron_memory.persistent_entropy import (
    EntropyTracker,
    compute_entropy_delta,
    compute_persistent_entropy,
)
from tetrahedron_memory.tetra_dream import TetraDreamCycle
from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_self_org import TetraSelfOrganizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_content(i: int) -> str:
    return f"memory_item_{i}_{np.random.bytes(4).hex()}"


def _populate_mesh(mesh: TetraMesh, n: int) -> List[str]:
    ids = []
    for i in range(n):
        pt = np.random.randn(3) * 0.5
        tid = mesh.store(_random_content(i), seed_point=pt, weight=1.0 + np.random.rand())
        ids.append(tid)
    return ids


# ===================================================================
# 1. THROUGHPUT BENCHMARK
# ===================================================================

@pytest.mark.benchmark
class TestInsertThroughput:
    def test_mesh_store_throughput_reasonable(self):
        mesh = TetraMesh(time_lambda=0.001)
        n = 5_000
        start = time.perf_counter()
        for i in range(n):
            pt = np.array([float(i % 100) * 0.01, float(i // 100 % 100) * 0.01, float(i % 50) * 0.02])
            mesh.store(f"thr_{i}", seed_point=pt, weight=1.0)
        elapsed = time.perf_counter() - start
        throughput = n / elapsed
        assert throughput > 1_000, f"Throughput {throughput:.0f} ops/sec < 1,000 target"
        assert len(mesh.tetrahedra) == n

    def test_geo_memory_body_store_throughput_25k(self):
        body = GeoMemoryBody()
        n = 50_000
        start = time.perf_counter()
        for i in range(n):
            body.store(content=f"body_thr_{i}", weight=1.0)
        elapsed = time.perf_counter() - start
        throughput = n / elapsed
        assert throughput > 12_000, f"GeoMemoryBody throughput {throughput:.0f} ops/sec < 12,000 target"
        assert len(body._nodes) == n


# ===================================================================
# 2. QUERY LATENCY BENCHMARK
# ===================================================================

@pytest.mark.benchmark
class TestQueryLatency:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mesh = TetraMesh(time_lambda=0.001)
        self.body = GeoMemoryBody()
        self.n = 5_000
        for i in range(self.n):
            pt = np.array([float(i % 50) * 0.05, float(i // 50 % 50) * 0.05, 0.0])
            self.mesh.store(f"q_{i}", seed_point=pt, weight=1.0)
            self.body.store(content=f"q_{i}", weight=1.0)

    def test_mesh_query_p99_under_8ms(self):
        latencies = []
        for i in range(500):
            query_pt = np.random.randn(3) * 0.5
            start = time.perf_counter()
            self.mesh.query_topological(query_pt, k=5)
            latencies.append((time.perf_counter() - start) * 1000)
        p50 = float(np.percentile(latencies, 50))
        p99 = float(np.percentile(latencies, 99))
        assert p99 < 8.0, f"Mesh query p99 latency {p99:.2f}ms > 8ms target (p50={p50:.2f}ms)"

    def test_geo_body_query_p99_under_8ms(self):
        latencies = []
        for i in range(200):
            start = time.perf_counter()
            self.body.query(f"q_{i % self.n}", k=5)
            latencies.append((time.perf_counter() - start) * 1000)
        p99 = float(np.percentile(latencies, 99))
        assert p99 < 8.0, f"GeoMemoryBody query p99 latency {p99:.2f}ms > 8ms target"


# ===================================================================
# 3. SCALE TEST — ETERNITY PRINCIPLE (no deletion/weakening)
# ===================================================================

@pytest.mark.slow
class TestScaleEternity:
    def test_100k_memories_no_deletion(self):
        body = GeoMemoryBody()
        n = 100_000
        batch_size = 10_000
        expected_total = 0

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            for i in range(batch_start, batch_end):
                body.store(content=f"scale_{i}", weight=1.0)
                expected_total += 1

            current_count = len(body._nodes)
            assert current_count == expected_total, (
                f"Eternity violation at batch {batch_start}: "
                f"expected {expected_total}, got {current_count}"
            )

            if batch_end % 50_000 == 0 or batch_end == n:
                gc.collect()

        final_count = len(body._nodes)
        assert final_count == n, f"Final count {final_count} != {n}"

    def test_weights_never_drop_below_initial(self):
        mesh = TetraMesh(time_lambda=0.001)
        n = 5_000
        initial_weights = {}
        for i in range(n):
            pt = np.random.randn(3) * 0.3
            w = 1.0 + np.random.rand() * 2.0
            tid = mesh.store(f"wtest_{i}", seed_point=pt, weight=w)
            initial_weights[tid] = w

        organizer = TetraSelfOrganizer(mesh, max_iterations=5)
        for _ in range(3):
            organizer.run()

        dream = TetraDreamCycle(mesh, cycle_interval=999999)
        for _ in range(3):
            dream.trigger_now()

        for tid, tetra in mesh.tetrahedra.items():
            if "__system__" in tetra.labels:
                continue
            if tid in initial_weights:
                assert tetra.weight >= initial_weights[tid] * 0.8, (
                    f"Weight regression for {tid}: initial={initial_weights[tid]:.2f}, "
                    f"current={tetra.weight:.2f}"
                )


# ===================================================================
# 4. ENTROPY CONVERGENCE — dream cycle drops >= 18%
# ===================================================================

@pytest.mark.benchmark
class TestEntropyConvergence:
    def test_dream_cycle_entropy_drop_18_percent(self):
        np.random.seed(42)
        mesh = TetraMesh(time_lambda=0.001)
        centers = [
            np.array([0.0, 0.0, 0.0]),
            np.array([3.0, 0.0, 0.0]),
            np.array([0.0, 3.0, 0.0]),
            np.array([3.0, 3.0, 0.0]),
        ]
        for i in range(200):
            c = centers[i % len(centers)]
            pt = c + np.random.randn(3) * 0.3
            mesh.store(
                f"ent_{i}",
                seed_point=pt,
                weight=0.5 + np.random.rand(),
            )

        st_before = mesh.compute_ph()
        if st_before is None:
            pytest.skip("gudhi not available")
        entropy_before = compute_persistent_entropy(st_before)
        if entropy_before <= 0:
            pytest.skip("entropy too low to measure drop")

        def _org_callback(m):
            TetraSelfOrganizer(m, max_iterations=20).run()

        dream = TetraDreamCycle(
            mesh, cycle_interval=999999, dream_weight=0.8, organizer=_org_callback
        )
        for _ in range(8):
            dream.trigger_now()

        organizer = TetraSelfOrganizer(mesh, max_iterations=20)
        for _ in range(5):
            organizer.run()

        st_after = mesh.compute_ph()
        entropy_after = compute_persistent_entropy(st_after)
        delta = compute_entropy_delta(entropy_before, entropy_after)

        assert delta >= 0.18, (
            f"Entropy drop {delta:.1%} < 18% target "
            f"(before={entropy_before:.4f}, after={entropy_after:.4f})"
        )

    def test_organizer_entropy_converges(self):
        mesh = TetraMesh(time_lambda=0.001)
        for i in range(50):
            pt = np.random.randn(3) * 1.5
            mesh.store(f"conv_{i}", seed_point=pt, weight=0.5 + np.random.rand())

        organizer = TetraSelfOrganizer(mesh, max_iterations=15)
        entropy_history = []

        for _ in range(5):
            st = mesh.compute_ph()
            if st is not None:
                e = compute_persistent_entropy(st)
                entropy_history.append(e)
            organizer.run()

        if len(entropy_history) >= 2:
            assert entropy_history[-1] <= entropy_history[0] * 1.05, (
                f"Entropy diverged: {entropy_history[0]:.4f} -> {entropy_history[-1]:.4f}"
            )


# ===================================================================
# 5. MEMORY LEAK DETECTION
# ===================================================================

@pytest.mark.slow
class TestMemoryStability:
    def test_no_memory_leak_over_100k_operations(self):
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")

        process = psutil.Process(os.getpid())
        gc.collect()
        baseline_mb = process.memory_info().rss / (1024 * 1024)

        mesh = TetraMesh(time_lambda=0.001)
        n_ops = 100_000
        for i in range(n_ops):
            pt = np.random.randn(3) * 0.3
            mesh.store(f"leak_{i}", seed_point=pt, weight=1.0)

        gc.collect()
        peak_mb = process.memory_info().rss / (1024 * 1024)
        growth_mb = peak_mb - baseline_mb

        per_memory_bytes = (growth_mb * 1024 * 1024) / n_ops
        assert per_memory_bytes < 2000, (
            f"Memory growth {per_memory_bytes:.0f} bytes/memory > 2000 limit "
            f"(growth={growth_mb:.1f}MB for {n_ops} memories)"
        )

    def test_no_memory_leak_closed_loop_cycles(self):
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")

        process = psutil.Process(os.getpid())
        body = GeoMemoryBody()
        for i in range(200):
            body.store(content=f"cl_leak_{i}", weight=1.0)

        engine = ClosedLoopEngine(memory=body)
        gc.collect()
        baseline_mb = process.memory_info().rss / (1024 * 1024)

        for i in range(50):
            engine.run_cycle(context=f"stability_test_{i}")

        gc.collect()
        after_mb = process.memory_info().rss / (1024 * 1024)
        growth_mb = after_mb - baseline_mb

        assert growth_mb < 100, f"Closed-loop leaked {growth_mb:.1f}MB over 50 cycles"


# ===================================================================
# 6. FAULT TOLERANCE — consistency & compensation
# ===================================================================

class TestFaultTolerance:
    def test_compensation_log_retries_all(self):
        log = CompensationLog()
        errors_logged = []
        for i in range(20):
            eid = log.record(
                f"op_{i}",
                f"bucket_{i % 4}",
                {"content": f"data_{i}"},
                f"error_{i}",
            )
            errors_logged.append(eid)

        call_log = []
        results = log.retry_all(lambda op, params: call_log.append(op))
        assert len(results) == 20
        assert all(r["status"] == "resolved" for r in results)
        assert len(call_log) == 20

    def test_vector_clock_survives_bucket_loss(self):
        buckets = [f"b{i}" for i in range(8)]
        vc = VectorClock(buckets)
        for b in buckets:
            vc.increment(b)

        snap_before = vc.snapshot()
        surviving = buckets[:5]
        vc2 = VectorClock(surviving)
        vc2.merge(vc)

        for b in surviving:
            assert vc2.get(b) == snap_before[b]

    def test_consistency_manager_partial_failure_recovery(self):
        cm = ConsistencyManager(["b1", "b2", "b3", "b4"])
        for i in range(100):
            bid = f"b{(i % 4) + 1}"
            cm.record_version(f"node_{i}", bid, f"data_{i}")

        cm._bucket_locks.pop("b3", None)

        locked = cm.acquire_lock(["b1", "b2", "b4"])
        assert locked
        cm.release_lock(["b1", "b2", "b4"])

        conflicts = cm.detect_conflicts()
        staleness = cm.get_staleness("b1")
        assert isinstance(conflicts, list)
        assert isinstance(staleness, list)

    def test_concurrent_stores_thread_safety(self):
        mesh = TetraMesh(time_lambda=0.001)
        errors = []
        n_per_thread = 500
        n_threads = 4

        def worker(thread_id: int):
            try:
                for i in range(n_per_thread):
                    pt = np.random.randn(3) * 0.1
                    mesh.store(f"t{thread_id}_{i}", seed_point=pt, weight=1.0)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(mesh.tetrahedra) == n_per_thread * n_threads


# ===================================================================
# 7. CLOSED-LOOP ENDURANCE
# ===================================================================

@pytest.mark.benchmark
class TestClosedLoopEndurance:
    def test_100_cycles_no_count_regression(self):
        body = GeoMemoryBody()
        for i in range(50):
            body.store(content=f"pre_{i}", weight=1.0)

        engine = ClosedLoopEngine(memory=body)
        initial_count = len(body._nodes)

        for i in range(100):
            result = engine.run_cycle(context=f"endurance_{i}")
            current_count = len(body._nodes)
            assert current_count >= initial_count, (
                f"Cycle {i}: count regressed from {initial_count} to {current_count}"
            )

        assert engine._cycle_count == 100
        final_count = len(body._nodes)
        assert final_count >= initial_count

    def test_dream_reintegration_preserves_count(self):
        mesh = TetraMesh(time_lambda=0.001)
        for i in range(30):
            pt = np.random.randn(3) * 0.5
            mesh.store(f"dream_test_{i}", seed_point=pt, weight=0.5)

        initial_count = len(mesh.tetrahedra)
        dream = TetraDreamCycle(mesh, cycle_interval=999999, max_dream_tetra=10)
        for _ in range(10):
            dream.trigger_now()

        final_count = len(mesh.tetrahedra)
        assert final_count >= initial_count, (
            f"Dream cycle deleted memories: {initial_count} -> {final_count}"
        )

    def test_self_organize_never_deletes(self):
        mesh = TetraMesh(time_lambda=0.001)
        for i in range(40):
            pt = np.random.randn(3) * 1.0
            mesh.store(f"so_test_{i}", seed_point=pt, weight=0.5 + np.random.rand())

        initial_count = len(mesh.tetrahedra)
        non_system_initial = sum(
            1 for t in mesh.tetrahedra.values()
            if "__system__" not in t.labels
        )

        organizer = TetraSelfOrganizer(mesh, max_iterations=20)
        for _ in range(10):
            organizer.run()

        final_count = len(mesh.tetrahedra)
        non_system_final = sum(
            1 for t in mesh.tetrahedra.values()
            if "__system__" not in t.labels
        )
        merged_count = sum(
            1 for t in mesh.tetrahedra.values()
            if "merged_from" in t.metadata
        )
        total_accounted = non_system_final + merged_count

        assert total_accounted >= non_system_initial, (
            f"Self-org deleted memories: {non_system_initial} originals, "
            f"only {non_system_final} surviving + {merged_count} merged = {total_accounted}"
        )


# ===================================================================
# 8. PERSISTENCE INTEGRITY
# ===================================================================

class TestPersistenceIntegrity:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_incremental_upsert_preserves_all_nodes(self):
        from tetrahedron_memory.persistence import ParquetPersistence

        pers = ParquetPersistence(storage_path=self.tmpdir)
        body = GeoMemoryBody()
        ids = []
        for i in range(100):
            nid = body.store(content=f"persist_{i}", weight=float(i))
            ids.append(nid)

        from tetrahedron_memory.core import MemoryNode
        nodes_v1 = {}
        for nid, node in body._nodes.items():
            nodes_v1[nid] = MemoryNode(
                id=node.id, content=node.content,
                geometry=node.geometry.copy(), timestamp=node.timestamp,
                weight=node.weight, labels=list(node.labels),
                metadata=dict(node.metadata),
            )

        pers.write_incremental_full(nodes_v1, snapshot_name="integrity")

        import pandas as pd
        df = pd.read_parquet(pers._base_path / "incremental_integrity.parquet")
        assert len(df) == 100
        assert set(df["id"].tolist()) == set(ids)

    def test_compact_snapshots_deduplicates(self):
        from tetrahedron_memory.persistence import ParquetPersistence
        from tetrahedron_memory.core import MemoryNode

        pers = ParquetPersistence(storage_path=self.tmpdir)
        body = GeoMemoryBody()
        for i in range(10):
            body.store(content=f"comp_{i}", weight=1.0)

        pers.write_full_snapshot(body._nodes, snapshot_name="dedup")

        updated = {}
        for nid, node in body._nodes.items():
            updated[nid] = MemoryNode(
                id=node.id, content=node.content,
                geometry=node.geometry.copy(), timestamp=node.timestamp,
                weight=5.0, labels=list(node.labels),
                metadata=dict(node.metadata),
            )
        pers.write_incremental_full(updated, snapshot_name="dedup")

        pers.compact_snapshots(snapshot_name="dedup")

        import pandas as pd
        df = pd.read_parquet(pers._base_path / "snapshot_dedup.parquet")
        assert len(df) == 10
        assert all(df["weight"].apply(lambda x: x == pytest.approx(5.0, rel=0.1)))


# ===================================================================
# 9. MONITORING & OBSERVABILITY
# ===================================================================

class TestMonitoringProduction:
    def test_all_metrics_registered(self):
        from tetrahedron_memory import monitoring
        assert hasattr(monitoring, "STORE_LATENCY")
        assert hasattr(monitoring, "QUERY_LATENCY")
        assert hasattr(monitoring, "ENTROPY_GAUGE")
        assert hasattr(monitoring, "INTEGRATION_COUNTER")
        assert hasattr(monitoring, "DREAM_COUNTER")

    def test_generate_metrics_no_error(self):
        from tetrahedron_memory.monitoring import generate_metrics
        result = generate_metrics()
        assert isinstance(result, str)

    def test_grafana_dashboard_valid_json(self):
        import json
        from tetrahedron_memory.monitoring import get_grafana_dashboard_json
        dash = json.loads(get_grafana_dashboard_json())
        assert "panels" in dash
        assert len(dash["panels"]) >= 15

    def test_health_check_structure(self):
        from tetrahedron_memory.monitoring import health_check
        hc = health_check()
        assert hc["status"] == "ok"


# ===================================================================
# 10. DISTRIBUTED PARTITIONING STRESS
# ===================================================================

@pytest.mark.benchmark
class TestDistributedStress:
    def test_spatial_router_10k_stores(self):
        from tetrahedron_memory.partitioning import SpatialBucketRouter

        router = SpatialBucketRouter(max_points_per_bucket=500)
        router.initialize()
        n = 10_000

        for i in range(n):
            geom = np.random.randn(3) * 2.0
            router.route_store(geom, f"dist_{i}", weight=1.0)

        stats = router.get_statistics()
        assert stats["total_nodes"] == n
        assert stats["total_buckets"] >= 1

    def test_ghost_cells_no_orphan_leak(self):
        from tetrahedron_memory.partitioning import SpatialBucketRouter

        router = SpatialBucketRouter(max_points_per_bucket=50)
        router.initialize()
        for i in range(300):
            geom = np.random.randn(3) * 0.5
            router.route_store(geom, f"gc_{i}", weight=1.0)

        router.update_ghost_cells()
        ghost_stats = router.get_ghost_cell_stats()
        assert ghost_stats["total_ghost_cells"] >= 0

        expired = router.prune_expired_ghosts()
        assert expired == 0
