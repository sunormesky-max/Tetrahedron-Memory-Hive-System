"""
Tests for Phase 3-4 extensions:
  - Zigzag Persistence dynamic modeling
  - Resolution Pyramid multi-scale representation
  - Multi-parameter filtering
"""

import time
import unittest

import numpy as np

from tetrahedron_memory.core import GeoMemoryBody, MemoryNode, QueryResult
from tetrahedron_memory.tetra_mesh import TetraMesh


class TestZigzagPersistence(unittest.TestCase):
    def setUp(self):
        self.mesh = TetraMesh(time_lambda=0.001)
        from tetrahedron_memory.zigzag_persistence import ZigzagTracker
        self.tracker = ZigzagTracker(window_size=10)

    def _populate(self, n=10):
        ids = []
        for i in range(n):
            pt = np.array([np.sin(i * 0.7) * 0.5, np.cos(i * 0.7) * 0.5, i * 0.1])
            tid = self.mesh.store(
                content=f"mem_{i}",
                seed_point=pt,
                labels=[f"label_{i % 3}"],
                weight=1.0 + i * 0.1,
            )
            ids.append(tid)
        return ids

    def test_single_snapshot(self):
        self._populate(10)
        snap = self.tracker.record_snapshot(self.mesh)
        self.assertEqual(snap.tetra_count, 10)
        self.assertGreaterEqual(snap.total_entropy, 0.0)
        self.assertEqual(len(self.tracker.detect_phase_transitions()), 0)

    def test_multiple_snapshots_detect_transitions(self):
        self._populate(10)
        self.tracker.record_snapshot(self.mesh)

        for i in range(10, 20):
            pt = np.random.randn(3) * 0.3
            self.mesh.store(content=f"mem_{i}", seed_point=pt, labels=["new"])

        self.tracker.record_snapshot(self.mesh)

        transitions = self.tracker.detect_phase_transitions()
        self.assertGreater(len(transitions), 0)
        self.assertIn(transitions[0].transition_type, ("birth", "death", "mutation"))

    def test_feature_lifetimes(self):
        self._populate(10)
        self.tracker.record_snapshot(self.mesh)
        self.tracker.record_snapshot(self.mesh)
        lifetimes = self.tracker.get_feature_lifetimes()
        self.assertIn(0, lifetimes)
        self.assertIsInstance(lifetimes[0], list)

    def test_predict_emerging_features_insufficient_data(self):
        pred = self.tracker.predict_emerging_features()
        self.assertEqual(pred["prediction"], "insufficient_data")

    def test_predict_after_multiple_snapshots(self):
        self._populate(10)
        for _ in range(5):
            self.tracker.record_snapshot(self.mesh)
            self.mesh.store(content="extra", seed_point=np.random.randn(3) * 0.2, labels=["extra"])

        pred = self.tracker.predict_emerging_features()
        self.assertIn("prediction", pred)
        self.assertIn("entropy_trend", pred)
        self.assertIn("confidence", pred)

    def test_dynamic_barcode(self):
        self._populate(10)
        self.tracker.record_snapshot(self.mesh)
        self.tracker.record_snapshot(self.mesh)

        barcode = self.tracker.get_dynamic_barcode(dimension=-1)
        self.assertIn("barcodes", barcode)
        self.assertIn("window", barcode)
        self.assertEqual(barcode["window"], 2)

        barcode_h0 = self.tracker.get_dynamic_barcode(dimension=0)
        self.assertIn("H0", barcode_h0["barcodes"])

    def test_zigzag_stability(self):
        self._populate(10)
        for _ in range(5):
            self.tracker.record_snapshot(self.mesh)

        stability = self.tracker.get_zigzag_stability()
        self.assertIn("stability", stability)
        self.assertIn("transition_rate", stability)
        self.assertGreaterEqual(stability["stability"], 0.0)
        self.assertLessEqual(stability["stability"], 1.0)

    def test_sliding_window(self):
        from tetrahedron_memory.zigzag_persistence import ZigzagTracker as ZT
        tracker = ZT(window_size=3)
        self._populate(5)
        for _ in range(5):
            tracker.record_snapshot(self.mesh)
        self.assertLessEqual(len(tracker._snapshots), 3)

    def test_status(self):
        self._populate(5)
        self.tracker.record_snapshot(self.mesh)
        status = self.tracker.get_status()
        self.assertEqual(status["snapshot_count"], 1)
        self.assertIn("stability", status)
        self.assertIn("prediction", status)

    def test_empty_mesh(self):
        snap = self.tracker.record_snapshot(self.mesh)
        self.assertEqual(snap.tetra_count, 0)
        self.assertEqual(snap.total_entropy, 0.0)

    def test_transition_types(self):
        self._populate(10)
        self.tracker.record_snapshot(self.mesh)

        for i in range(10):
            self.mesh.store(content=f"new_{i}", seed_point=np.random.randn(3) * 0.3, labels=["burst"])

        self.tracker.record_snapshot(self.mesh)

        transitions = self.tracker.detect_phase_transitions()
        types = {t.transition_type for t in transitions}
        self.assertTrue(len(types) > 0)
        for t in types:
            self.assertIn(t, ("birth", "death", "mutation"))


class TestResolutionPyramid(unittest.TestCase):
    def setUp(self):
        self.mesh = TetraMesh(time_lambda=0.001)
        from tetrahedron_memory.resolution_pyramid import ResolutionPyramid
        self.pyramid = ResolutionPyramid(max_levels=3, min_cluster_size=3, coarsening_ratio=0.4)

    def _populate(self, n=30):
        ids = []
        for i in range(n):
            pt = np.array([
                np.sin(i * 0.5) * 0.8,
                np.cos(i * 0.5) * 0.8,
                (i % 5) * 0.2,
            ])
            tid = self.mesh.store(
                content=f"mem_{i}",
                seed_point=pt,
                labels=[f"cat_{i % 4}"],
                weight=1.0 + (i % 5) * 0.3,
            )
            ids.append(tid)
        return ids

    def test_build_pyramid(self):
        self._populate(30)
        result = self.pyramid.build(self.mesh)
        self.assertGreater(result["levels"], 0)
        self.assertIn("nodes_per_level", result)

    def test_level_0_has_all_tetra(self):
        self._populate(20)
        self.pyramid.build(self.mesh)
        level_0 = self.pyramid._levels.get(0, {})
        self.assertEqual(len(level_0), 20)

    def test_coarser_levels_have_fewer_nodes(self):
        self._populate(50)
        self.pyramid.build(self.mesh)
        levels = sorted(self.pyramid._levels.keys())
        if len(levels) >= 2:
            self.assertGreaterEqual(
                len(self.pyramid._levels[levels[0]]),
                len(self.pyramid._levels[levels[-1]]),
            )

    def test_query_level_0(self):
        self._populate(20)
        self.pyramid.build(self.mesh)
        results = self.pyramid.query(np.array([0.0, 0.0, 0.0]), k=5, level=0)
        self.assertLessEqual(len(results), 5)
        self.assertGreater(len(results), 0)

    def test_query_coarse_level(self):
        self._populate(50)
        self.pyramid.build(self.mesh)
        max_level = max(self.pyramid._levels.keys())
        if max_level > 0:
            results = self.pyramid.query(np.array([0.0, 0.0, 0.0]), k=3, level=max_level)
            self.assertGreater(len(results), 0)

    def test_auto_route(self):
        self._populate(50)
        self.pyramid.build(self.mesh)
        results = self.pyramid.auto_route(np.array([0.0, 0.0, 0.0]), k=5)
        self.assertLessEqual(len(results), 5)

    def test_ensure_built(self):
        self._populate(20)
        self.pyramid.ensure_built(self.mesh)
        self.assertFalse(self.pyramid._dirty)

    def test_mark_dirty_rebuilds(self):
        self._populate(20)
        self.pyramid.build(self.mesh)
        self.assertFalse(self.pyramid._dirty)

        self.pyramid.mark_dirty()
        self.assertTrue(self.pyramid._dirty)

        self.pyramid.ensure_built(self.mesh)
        self.assertFalse(self.pyramid._dirty)

    def test_get_level_stats(self):
        self._populate(30)
        self.pyramid.build(self.mesh)
        stats = self.pyramid.get_level_stats(0)
        self.assertEqual(stats["nodes"], 30)
        self.assertIn("avg_weight", stats)

    def test_get_all_stats(self):
        self._populate(30)
        self.pyramid.build(self.mesh)
        stats = self.pyramid.get_all_stats()
        self.assertIn("num_levels", stats)
        self.assertIn("levels", stats)

    def test_pyramid_node_contains_point(self):
        from tetrahedron_memory.resolution_pyramid import PyramidNode
        node = PyramidNode(
            node_id="test",
            level=0,
            centroid=np.array([0.0, 0.0, 0.0]),
            bbox_min=np.array([-1.0, -1.0, -1.0]),
            bbox_max=np.array([1.0, 1.0, 1.0]),
        )
        self.assertTrue(node.contains_point(np.array([0.5, 0.5, 0.5])))
        self.assertFalse(node.contains_point(np.array([5.0, 5.0, 5.0]), margin=0.1))

    def test_pyramid_node_similarity_score(self):
        from tetrahedron_memory.resolution_pyramid import PyramidNode
        node = PyramidNode(
            node_id="test",
            level=0,
            centroid=np.array([0.0, 0.0, 0.0]),
            max_weight=2.0,
            tetra_count=5,
        )
        score_near = node.similarity_score(np.array([0.1, 0.1, 0.1]))
        score_far = node.similarity_score(np.array([5.0, 5.0, 5.0]))
        self.assertGreater(score_near, score_far)

    def test_too_few_tetra(self):
        self.mesh.store("a", seed_point=np.array([0.0, 0.0, 0.0]))
        result = self.pyramid.build(self.mesh)
        self.assertEqual(result["levels"], 0)

    def test_empty_mesh(self):
        result = self.pyramid.build(self.mesh)
        self.assertEqual(result["levels"], 0)


class TestMultiParameterFilter(unittest.TestCase):
    def setUp(self):
        self.mesh = TetraMesh(time_lambda=0.001)
        from tetrahedron_memory.multiparameter_filter import MultiParameterQuery
        self.mpq_class = MultiParameterQuery

    def _populate(self, n=20):
        ids = []
        for i in range(n):
            pt = np.array([
                np.sin(i * 0.8) * 0.6,
                np.cos(i * 0.8) * 0.6,
                i * 0.05,
            ])
            tid = self.mesh.store(
                content=f"mem_{i}",
                seed_point=pt,
                labels=[f"cat_{i % 3}", f"tag_{i % 5}"],
                weight=0.5 + (i % 5) * 0.3,
            )
            ids.append(tid)
        return ids

    def test_spatial_only(self):
        self._populate(20)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("spatial", {
            "query_point": np.array([0.0, 0.0, 0.0]),
            "max_distance": 3.0,
        }, weight=1.0)
        results = mpq.execute(k=5)
        self.assertLessEqual(len(results), 5)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertGreater(r.composite_score, 0.0)

    def test_temporal_filter(self):
        self._populate(20)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("temporal", {"recency_seconds": 3600.0}, weight=1.0)
        results = mpq.execute(k=10)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertGreater(r.individual_scores["temporal"], 0.0)

    def test_weight_filter(self):
        self._populate(20)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("weight", {"min_weight": 0.5, "max_weight": 3.0}, weight=1.0)
        results = mpq.execute(k=10)
        self.assertGreater(len(results), 0)

    def test_label_filter_required(self):
        self._populate(20)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("label", {"required": ["cat_0"]}, weight=1.0, hard_filter=True, min_score=0.01)
        results = mpq.execute(k=20)
        for r in results:
            self.assertIn("cat_0", r.labels)

    def test_label_filter_preferred(self):
        self._populate(20)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("label", {"preferred": ["cat_0", "cat_1"]}, weight=1.0)
        results = mpq.execute(k=10)
        self.assertGreater(len(results), 0)

    def test_density_filter(self):
        self._populate(20)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("density", {"neighbor_radius": 1.0}, weight=1.0)
        results = mpq.execute(k=5)
        self.assertGreater(len(results), 0)

    def test_topology_filter(self):
        self._populate(20)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("topology", {"integration_boost": True, "connectivity_weight": 0.3}, weight=1.0)
        results = mpq.execute(k=5)
        self.assertGreater(len(results), 0)

    def test_composite_filter(self):
        self._populate(20)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("spatial", {"query_point": np.array([0.0, 0.0, 0.0]), "max_distance": 3.0}, weight=0.4)
        mpq.add_filter("temporal", {"recency_seconds": 3600.0}, weight=0.2)
        mpq.add_filter("density", {"neighbor_radius": 1.0}, weight=0.2)
        mpq.add_filter("weight", {"min_weight": 0.1}, weight=0.2)
        results = mpq.execute(k=10)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("spatial", r.individual_scores)
            self.assertIn("temporal", r.individual_scores)
            self.assertGreater(r.composite_score, 0.0)

    def test_results_sorted_by_composite(self):
        self._populate(20)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("spatial", {"query_point": np.array([0.0, 0.0, 0.0]), "max_distance": 5.0}, weight=0.5)
        mpq.add_filter("weight", {"min_weight": 0.0}, weight=0.5)
        results = mpq.execute(k=10)
        scores = [r.composite_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_execute_with_ids(self):
        ids = self._populate(20)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("weight", {"min_weight": 0.0}, weight=1.0)
        results = mpq.execute_with_ids(ids[:10], k=5)
        self.assertLessEqual(len(results), 5)
        returned_ids = {r.tetra_id for r in results}
        self.assertTrue(returned_ids.issubset(set(ids[:10])))

    def test_clear_filters(self):
        self._populate(10)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("spatial", {"query_point": np.array([0.0, 0.0, 0.0]), "max_distance": 3.0})
        mpq.clear_filters()
        self.assertEqual(len(mpq._filters), 0)

    def test_empty_mesh(self):
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("spatial", {"query_point": np.array([0.0, 0.0, 0.0])})
        results = mpq.execute(k=5)
        self.assertEqual(len(results), 0)

    def test_no_filters(self):
        self._populate(10)
        mpq = self.mpq_class(self.mesh)
        results = mpq.execute(k=5)
        self.assertEqual(len(results), 0)

    def test_result_has_metadata(self):
        self._populate(10)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("spatial", {"query_point": np.array([0.0, 0.0, 0.0]), "max_distance": 5.0})
        results = mpq.execute(k=3)
        for r in results:
            self.assertTrue(hasattr(r, "content"))
            self.assertTrue(hasattr(r, "centroid"))
            self.assertTrue(hasattr(r, "labels"))
            self.assertTrue(hasattr(r, "individual_scores"))

    def test_access_filter(self):
        ids = self._populate(10)
        tetra = self.mesh.get_tetrahedron(ids[0])
        if tetra:
            tetra.access_count = 10
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("access", {"min_access_count": 5, "max_access_count": 100})
        results = mpq.execute(k=10)
        for r in results:
            self.assertIn("access", r.individual_scores)

    def test_hard_filter_excludes(self):
        self._populate(20)
        mpq = self.mpq_class(self.mesh)
        mpq.add_filter("weight", {"min_weight": 5.0, "max_weight": 10.0}, weight=1.0, hard_filter=True, min_score=0.01)
        results = mpq.execute(k=20)
        for r in results:
            self.assertGreaterEqual(r.weight, 5.0)


class TestGeoMemoryBodyIntegration(unittest.TestCase):
    """Integration tests for new features via GeoMemoryBody."""

    def setUp(self):
        self.body = GeoMemoryBody(dimension=3, precision="fast")

    def _populate(self, n=15):
        ids = []
        for i in range(n):
            tid = self.body.store(
                content=f"memory content {i}",
                labels=[f"topic_{i % 3}", f"level_{i % 2}"],
                weight=0.5 + (i % 4) * 0.3,
            )
            ids.append(tid)
        return ids

    def test_record_zigzag_snapshot(self):
        self._populate(10)
        result = self.body.record_zigzag_snapshot()
        self.assertIsNotNone(result)
        self.assertIn("total_entropy", result)
        self.assertIn("h0_count", result)

    def test_get_zigzag_status(self):
        self._populate(10)
        self.body.record_zigzag_snapshot()
        status = self.body.get_zigzag_status()
        self.assertIn("snapshot_count", status)
        self.assertEqual(status["snapshot_count"], 1)

    def test_predict_topology(self):
        self._populate(10)
        for _ in range(3):
            self.body.record_zigzag_snapshot()
        pred = self.body.predict_topology()
        self.assertIn("prediction", pred)

    def test_build_pyramid(self):
        self._populate(20)
        result = self.body.build_pyramid()
        self.assertGreater(result["levels"], 0)

    def test_query_pyramid(self):
        self._populate(20)
        self.body.build_pyramid()
        results = self.body.query_pyramid("test query", k=5)
        self.assertLessEqual(len(results), 5)
        for r in results:
            self.assertIsInstance(r, QueryResult)
            self.assertEqual(r.association_type, "pyramid")

    def test_query_multiparam(self):
        self._populate(20)
        results = self.body.query_multiparam(
            "test query",
            k=5,
            spatial_weight=0.4,
            temporal_weight=0.2,
            density_weight=0.2,
            weight_weight=0.2,
        )
        self.assertLessEqual(len(results), 5)
        for r in results:
            self.assertIsInstance(r, QueryResult)
            self.assertEqual(r.association_type, "multiparam")

    def test_query_multiparam_with_labels(self):
        self._populate(20)
        results = self.body.query_multiparam(
            "test query",
            k=10,
            labels_required=["topic_0"],
        )
        for r in results:
            self.assertIn("topic_0", r.node.labels)

    def test_statistics_include_new_features(self):
        self._populate(10)
        self.body.record_zigzag_snapshot()
        self.body.build_pyramid()
        stats = self.body.get_statistics()
        self.assertIn("zigzag", stats)
        self.assertIn("pyramid", stats)

    def test_get_dynamic_barcode(self):
        self._populate(10)
        for _ in range(3):
            self.body.record_zigzag_snapshot()
        barcode = self.body.get_dynamic_barcode(dimension=0)
        self.assertIn("barcodes", barcode)


if __name__ == "__main__":
    unittest.main()
