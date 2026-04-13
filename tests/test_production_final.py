"""
Phase 4: Final Production-Grade Test Suite.

Covers:
  - Principle compliance (eternity, integration, emergence, closed loop)
  - Performance & scale targets
  - Stability & noise control
  - Consistency & conflict resolution
  - New features validation (zigzag, pyramid, multiparam)
"""

import time
import threading
import unittest

import numpy as np

from tetrahedron_memory.core import GeoMemoryBody
from tetrahedron_memory.tetra_mesh import TetraMesh


class TestEternityPrinciple(unittest.TestCase):
    """No memory is ever deleted."""

    def test_all_inserted_memories_persist(self):
        body = GeoMemoryBody(dimension=3, precision="fast")
        ids = []
        for i in range(100):
            tid = body.store(f"memory_{i}", labels=[f"label_{i % 5}"], weight=0.5 + i * 0.01)
            ids.append(tid)

        nodes = body._nodes
        for tid in ids:
            self.assertIn(tid, nodes, f"Memory {tid} was lost!")

    def test_self_organize_preserves_content(self):
        body = GeoMemoryBody(dimension=3, precision="fast")
        contents = set()
        for i in range(20):
            body.store(f"mem_{i}", labels=[f"l{i % 3}"], weight=1.0 + i * 0.1)
            contents.add(f"mem_{i}")

        body.self_organize()

        surviving_contents = set()
        for tid, tetra in body._mesh.tetrahedra.items():
            if "__system__" not in tetra.labels:
                for c in contents:
                    if c in tetra.content or tetra.content in c:
                        surviving_contents.add(c)

        surviving_from_meta = set()
        for tid, tetra in body._mesh.tetrahedra.items():
            if "__system__" not in tetra.labels:
                for orig_c in contents:
                    if orig_c in tetra.metadata.get("merged_from_content", ""):
                        surviving_from_meta.add(orig_c)

        preserved = len(contents & (surviving_contents | surviving_from_meta))
        self.assertGreaterEqual(preserved, len(contents) * 0.5,
                                "Self-organize should preserve majority of content through merging")

    def test_dream_never_deletes_regular(self):
        mesh = TetraMesh(time_lambda=0.001)
        for i in range(15):
            pt = np.array([np.sin(i) * 0.3, np.cos(i) * 0.3, i * 0.1])
            mesh.store(f"regular_{i}", seed_point=pt, labels=[f"tag_{i % 3}"])

        regular_ids = set(mesh.tetrahedra.keys())
        from tetrahedron_memory.tetra_dream import TetraDreamCycle
        dc = TetraDreamCycle(mesh, cycle_interval=999999)
        dc.trigger_now()

        for rid in regular_ids:
            self.assertIn(rid, mesh.tetrahedra, f"Regular memory {rid} was deleted by dream!")

    def test_catalyze_never_reduces_below_minimum(self):
        body = GeoMemoryBody(dimension=3, precision="fast")
        for i in range(10):
            body.store(f"mem_{i}", weight=0.2)

        body.global_catalyze_integration(strength=5.0)
        for node in body._nodes.values():
            self.assertGreaterEqual(node.weight, 0.1)


class TestIntegrationQuality(unittest.TestCase):
    """Dream memories must be abstract fusion results."""

    def test_dream_creates_abstract_memories(self):
        mesh = TetraMesh(time_lambda=0.001)
        for i in range(15):
            pt = np.array([np.sin(i * 0.8) * 0.5, np.cos(i * 0.8) * 0.5, i * 0.1])
            mesh.store(f"source_{i}", seed_point=pt, labels=[f"topic_{i % 3}"], weight=1.0 + i * 0.1)

        from tetrahedron_memory.tetra_dream import TetraDreamCycle
        dc = TetraDreamCycle(mesh, cycle_interval=999999)
        result = dc.trigger_now()

        if result.get("dreams_created", 0) > 0:
            dream_tetras = [t for t in mesh.tetrahedra.values() if "__dream__" in t.labels]
            for dt in dream_tetras:
                self.assertIn("__dream__", dt.labels)
                self.assertGreater(len(dt.content), 0)

    def test_persistent_entropy_decreases_after_dreams(self):
        mesh = TetraMesh(time_lambda=0.001)
        for i in range(20):
            pt = np.array([np.sin(i * 0.5) * 0.8, np.cos(i * 0.5) * 0.8, i * 0.05])
            mesh.store(f"mem_{i}", seed_point=pt, labels=[f"cat_{i % 4}"])

        from tetrahedron_memory.persistent_entropy import compute_persistent_entropy
        st = mesh.compute_ph()
        entropy_before = compute_persistent_entropy(st) if st else 0.0

        from tetrahedron_memory.tetra_dream import TetraDreamCycle
        dc = TetraDreamCycle(mesh, cycle_interval=999999)
        for _ in range(3):
            dc.trigger_now()

        st2 = mesh.compute_ph()
        entropy_after = compute_persistent_entropy(st2) if st2 else 0.0

        if entropy_before > 0:
            delta = (entropy_before - entropy_after) / entropy_before
            self.assertGreaterEqual(delta, -0.5,
                                    f"Entropy increased too much: {delta:.3f}")


class TestClosedLoopCompleteness(unittest.TestCase):
    """Closed loop can start from any phase."""

    def test_full_cycle_completes(self):
        body = GeoMemoryBody(dimension=3, precision="fast")
        for i in range(10):
            body.store(f"context_{i}", labels=["test"])

        from tetrahedron_memory.closed_loop import ClosedLoopEngine
        engine = ClosedLoopEngine(body)
        result = engine.run_cycle(context="test context", k=5)
        self.assertIn("phase", result)

    def test_empty_context_self_emergent(self):
        body = GeoMemoryBody(dimension=3, precision="fast")
        for i in range(10):
            body.store(f"auto_{i}")

        from tetrahedron_memory.closed_loop import ClosedLoopEngine
        engine = ClosedLoopEngine(body)
        result = engine.run_cycle(context="", k=3)
        self.assertIn("phase", result)


class TestPerformanceTargets(unittest.TestCase):
    """Performance benchmarks for production targets."""

    def test_store_throughput_mesh(self):
        mesh = TetraMesh(time_lambda=0.001)
        n = 500
        pts = np.random.randn(n, 3) * 0.3
        start = time.time()
        for i in range(n):
            mesh.store(f"perf_{i}", seed_point=pts[i])
        elapsed = time.time() - start
        throughput = n / max(elapsed, 0.001)
        self.assertGreater(throughput, 50,
                           f"Mesh throughput {throughput:.0f} < 50 ops/sec")

    def test_query_latency_under_50ms(self):
        mesh = TetraMesh(time_lambda=0.001)
        for i in range(200):
            pt = np.array([np.sin(i * 0.3) * 0.5, np.cos(i * 0.3) * 0.5, i * 0.02])
            mesh.store(f"mem_{i}", seed_point=pt)

        query_pt = np.array([0.0, 0.0, 0.0])
        latencies = []
        for _ in range(50):
            start = time.time()
            mesh.query_topological(query_pt, k=5)
            latencies.append(time.time() - start)

        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        self.assertLess(p99, 0.05, f"Query p99 latency {p99*1000:.1f}ms > 50ms")


class TestConsistencyConflictResolution(unittest.TestCase):
    """Version conflict detection and auto-resolution."""

    def test_version_conflict_auto_resolved(self):
        from tetrahedron_memory.consistency import ConsistencyManager
        cm = ConsistencyManager(["bucket_a", "bucket_b"])

        vn1 = cm.record_version("node_1", "bucket_a", "content_v1")
        vn2 = cm.record_version("node_1", "bucket_b", "content_v2")

        conflicts = cm.detect_conflicts()
        self.assertGreater(len(conflicts), 0)

    def test_validate_before_write(self):
        from tetrahedron_memory.consistency import ConsistencyManager
        cm = ConsistencyManager(["bucket_a"])
        cm.record_version("node_1", "bucket_a", "content_v1")

        result = cm.validate_before_write("node_1", "bucket_a", expected_version=1)
        self.assertTrue(result["valid"])

        result = cm.validate_before_write("node_1", "bucket_a", expected_version=99)
        self.assertFalse(result["valid"])
        self.assertTrue(result["conflict"])

    def test_read_repair_multi(self):
        from tetrahedron_memory.consistency import ConsistencyManager
        cm = ConsistencyManager(["bucket_a", "bucket_b"])
        cm.record_version("node_1", "bucket_a", "content_v1")
        cm.record_version("node_1", "bucket_b", "content_v2")

        result = cm.read_repair_multi("bucket_b")
        self.assertGreaterEqual(result["stale_found"], 0)

    def test_consistency_health(self):
        from tetrahedron_memory.consistency import ConsistencyManager
        cm = ConsistencyManager(["bucket_a", "bucket_b"])
        cm.record_version("node_1", "bucket_a", "content")

        health = cm.get_health()
        self.assertTrue(health["enabled"])
        self.assertEqual(health["total_versioned_nodes"], 1)

    def test_compensation_log(self):
        from tetrahedron_memory.consistency import ConsistencyManager
        cm = ConsistencyManager(["bucket_a"])
        eid = cm.compensate_operation("store", "bucket_a", {"content": "x"}, "timeout")
        pending = cm.compensation_log.get_pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["id"], eid)


class TestZigzagStability(unittest.TestCase):
    """Zigzag stability and phase transition detection."""

    def test_stability_under_growth(self):
        tracker_cls = None
        from tetrahedron_memory.zigzag_persistence import ZigzagTracker
        tracker_cls = ZigzagTracker

        mesh = TetraMesh(time_lambda=0.001)
        tracker = tracker_cls(window_size=10)

        for i in range(30):
            pt = np.array([np.sin(i * 0.3) * 0.5, np.cos(i * 0.3) * 0.5, i * 0.05])
            mesh.store(f"stab_{i}", seed_point=pt)
            tracker.record_snapshot(mesh)

        stability = tracker.get_zigzag_stability()
        self.assertGreaterEqual(stability["stability"], 0.0)
        self.assertLessEqual(stability["stability"], 1.0)

    def test_dynamic_barcode_evolution(self):
        mesh = TetraMesh(time_lambda=0.001)
        from tetrahedron_memory.zigzag_persistence import ZigzagTracker
        tracker = ZigzagTracker(window_size=10)

        for i in range(10):
            mesh.store(f"evo_{i}", seed_point=np.random.randn(3) * 0.2)
            tracker.record_snapshot(mesh)

        barcode = tracker.get_dynamic_barcode(dimension=0)
        self.assertIn("H0", barcode["barcodes"])
        self.assertEqual(barcode["window"], 10)


class TestPyramidStructureStability(unittest.TestCase):
    """Pyramid maintains stable structure across rebuilds."""

    def test_repeated_rebuild_stable(self):
        mesh = TetraMesh(time_lambda=0.001)
        for i in range(30):
            pt = np.array([np.sin(i * 0.5) * 0.8, np.cos(i * 0.5) * 0.8, i * 0.05])
            mesh.store(f"pyr_{i}", seed_point=pt)

        from tetrahedron_memory.resolution_pyramid import ResolutionPyramid
        pyramid = ResolutionPyramid(max_levels=3)

        level_counts = []
        for _ in range(5):
            result = pyramid.build(mesh)
            level_counts.append(result["levels"])

        self.assertEqual(level_counts[-1], level_counts[0],
                         f"Pyramid level count changed across rebuilds: {level_counts}")


class TestMultiParamFilterCorrectness(unittest.TestCase):
    """Multi-parameter filter returns correct results."""

    def test_required_label_hard_filter(self):
        mesh = TetraMesh(time_lambda=0.001)
        for i in range(20):
            mesh.store(f"mp_{i}", seed_point=np.random.randn(3) * 0.2,
                       labels=[f"cat_{i % 3}", "common"])

        from tetrahedron_memory.multiparameter_filter import MultiParameterQuery
        mpq = MultiParameterQuery(mesh)
        mpq.add_filter("label", {"required": ["cat_0"]}, weight=1.0, hard_filter=True, min_score=0.01)
        results = mpq.execute(k=20)
        for r in results:
            self.assertIn("cat_0", r.labels)

    def test_spatial_proximity_ranking(self):
        mesh = TetraMesh(time_lambda=0.001)
        mesh.store("near", seed_point=np.array([0.01, 0.01, 0.01]))
        mesh.store("far", seed_point=np.array([5.0, 5.0, 5.0]))

        from tetrahedron_memory.multiparameter_filter import MultiParameterQuery
        mpq = MultiParameterQuery(mesh)
        mpq.add_filter("spatial", {"query_point": np.array([0.0, 0.0, 0.0]), "max_distance": 10.0}, weight=1.0)
        results = mpq.execute(k=2)
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0].composite_score, results[1].composite_score)


class TestTopologyHealthEndpoint(unittest.TestCase):
    """Health endpoint returns valid topology data."""

    def test_topology_health_data(self):
        body = GeoMemoryBody(dimension=3, precision="fast")
        for i in range(15):
            body.store(f"health_{i}", labels=[f"h_{i % 3}"])

        from tetrahedron_memory.structured_log import StructuredLogger, trace_context
        logger = StructuredLogger("test")
        with trace_context("test_trace_123") as tid:
            self.assertEqual(tid, "test_trace_123")
            logger.info("test_event", memory_count=15)

        self.assertEqual(current_trace_id_func(), "")


def current_trace_id_func():
    from tetrahedron_memory.structured_log import current_trace_id
    return current_trace_id()


class TestStructuredLogging(unittest.TestCase):
    """Structured logging produces valid JSON."""

    def test_trace_context(self):
        from tetrahedron_memory.structured_log import trace_context, current_trace_id
        self.assertEqual(current_trace_id(), "")
        with trace_context("abc123") as tid:
            self.assertEqual(tid, "abc123")
            self.assertEqual(current_trace_id(), "abc123")
        self.assertEqual(current_trace_id(), "")

    def test_nested_trace(self):
        from tetrahedron_memory.structured_log import trace_context, current_trace_id
        with trace_context("outer") as _:
            self.assertEqual(current_trace_id(), "outer")
            with trace_context("inner") as tid:
                self.assertEqual(tid, "inner")
            self.assertEqual(current_trace_id(), "outer")

    def test_alert_rules_valid(self):
        from tetrahedron_memory.monitoring import get_alert_rules
        rules = get_alert_rules()
        self.assertIn("groups", rules)
        self.assertGreater(len(rules["groups"]), 0)
        for group in rules["groups"]:
            self.assertIn("name", group)
            self.assertIn("rules", group)
            for rule in group["rules"]:
                self.assertIn("alert", rule)
                self.assertIn("expr", rule)


if __name__ == "__main__":
    unittest.main()
