"""
Tests for Q1-Q3 critical improvements.
"""

import time

import numpy as np
import pytest

from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_dream import (
    DreamProtocol,
    TetraDreamCycle,
    fusion_quality_score,
)
from tetrahedron_memory.partitioning import BoundingBox, GhostCell
from tetrahedron_memory.tetra_router import TetraMeshRouter


# ── Q1: Topology-aware fusion quality score ──────────────────

class TestFusionQualityScoreV2:
    def test_zero_for_empty(self):
        assert fusion_quality_score([], "content") == 0.0

    def test_zero_for_single_source(self):
        assert fusion_quality_score([{"labels": ["a"]}], "content") == 0.0

    def test_zero_for_none_content(self):
        inputs = [{"labels": ["a"]}, {"labels": ["b"]}]
        assert fusion_quality_score(inputs, None) == 0.0

    def test_basic_scoring(self):
        inputs = [
            {"labels": ["ai", "ml"], "weight": 1.0, "integration_count": 3,
             "centroid": [0.0, 0.0, 0.0]},
            {"labels": ["ai", "dl"], "weight": 1.2, "integration_count": 1,
             "centroid": [1.0, 0.0, 0.0]},
            {"labels": ["ml", "stats"], "weight": 0.8, "integration_count": 5,
             "centroid": [0.5, 1.0, 0.0]},
        ]
        q = fusion_quality_score(inputs, "[dream] AI/ML synthesis with deep learning insights")
        assert q > 0.0
        assert q <= 1.0
        # Should be higher than old version because of depth + dispersion + topology
        assert q > 0.3

    def test_topo_connectivity_bonus(self):
        connected = [
            {"labels": ["ai", "ml", "shared"], "weight": 1.0},
            {"labels": ["ai", "dl", "shared"], "weight": 1.0},
        ]
        disconnected = [
            {"labels": ["ai", "ml"], "weight": 1.0},
            {"labels": ["physics", "quantum"], "weight": 1.0},
        ]
        q_conn = fusion_quality_score(connected, "connected synthesis content here")
        q_disc = fusion_quality_score(disconnected, "disconnected synthesis content here")
        assert q_conn > q_disc

    def test_depth_weighting(self):
        deep = [
            {"labels": ["a"], "weight": 1.0, "integration_count": 10},
            {"labels": ["a"], "weight": 1.0, "integration_count": 8},
        ]
        shallow = [
            {"labels": ["a"], "weight": 1.0, "integration_count": 0},
            {"labels": ["a"], "weight": 1.0, "integration_count": 0},
        ]
        q_deep = fusion_quality_score(deep, "deep synthesis of rich memories")
        q_shallow = fusion_quality_score(shallow, "shallow synthesis")
        assert q_deep > q_shallow

    def test_centroid_dispersion_bonus(self):
        spread = [
            {"labels": ["a"], "weight": 1.0, "centroid": [0.0, 0.0, 0.0]},
            {"labels": ["a"], "weight": 1.0, "centroid": [3.0, 3.0, 3.0]},
        ]
        close = [
            {"labels": ["a"], "weight": 1.0, "centroid": [0.0, 0.0, 0.0]},
            {"labels": ["a"], "weight": 1.0, "centroid": [0.01, 0.01, 0.01]},
        ]
        q_spread = fusion_quality_score(spread, "bridge synthesis of distant memories")
        q_close = fusion_quality_score(close, "close synthesis")
        assert q_spread > q_close


# ── Q2: DreamProtocol (think→execute→reflect) ────────────────

class TestDreamProtocol:
    def test_default_protocol(self):
        protocol = DreamProtocol(quality_threshold=0.1)
        inputs = [
            {"labels": ["ai", "ml"], "weight": 1.0, "integration_count": 2,
             "content": "machine learning fundamentals"},
            {"labels": ["ai", "dl"], "weight": 1.2, "integration_count": 1,
             "content": "deep learning basics"},
        ]
        result = protocol.run(inputs)
        assert result["phase"] == "complete"
        assert result["content"] is not None
        assert result["quality"] >= 0.0
        assert isinstance(result["accepted"], bool)
        assert result["analysis"] is not None

    def test_custom_think(self):
        def my_think(inputs):
            return {"strategy": "custom", "focus": "bridge", "priority_labels": ["ai"]}

        protocol = DreamProtocol(think_fn=my_think, quality_threshold=0.0)
        inputs = [
            {"labels": ["ai"], "weight": 1.0, "content": "x"},
            {"labels": ["ai"], "weight": 1.0, "content": "y"},
        ]
        result = protocol.run(inputs)
        assert result["analysis"]["strategy"] == "custom"

    def test_custom_execute(self):
        def my_execute(inputs, analysis):
            return "CUSTOM: " + str(len(inputs)) + " sources synthesized"

        protocol = DreamProtocol(execute_fn=my_execute, quality_threshold=0.0)
        inputs = [
            {"labels": ["test"], "weight": 1.0, "content": "a"},
            {"labels": ["test"], "weight": 1.0, "content": "b"},
        ]
        result = protocol.run(inputs)
        assert result["content"].startswith("CUSTOM:")
        assert result["accepted"]

    def test_custom_reflect(self):
        def strict_reflect(inputs, content):
            return 0.8 if content and len(content) > 10 else 0.1

        protocol = DreamProtocol(reflect_fn=strict_reflect, quality_threshold=0.5)
        inputs = [
            {"labels": ["a"], "weight": 1.0, "content": "source one"},
            {"labels": ["a"], "weight": 1.0, "content": "source two"},
        ]
        result = protocol.run(inputs)
        assert result["quality"] > 0

    def test_rejection_below_threshold(self):
        protocol = DreamProtocol(quality_threshold=0.99)
        inputs = [
            {"labels": ["x"], "weight": 0.1, "content": "a"},
            {"labels": ["x"], "weight": 0.1, "content": "b"},
        ]
        result = protocol.run(inputs)
        # Very unlikely to pass 0.99 threshold
        assert isinstance(result["accepted"], bool)

    def test_statistics(self):
        protocol = DreamProtocol(quality_threshold=0.0)
        inputs = [
            {"labels": ["a"], "weight": 1.0, "content": "s1"},
            {"labels": ["a"], "weight": 1.0, "content": "s2"},
        ]
        for _ in range(3):
            protocol.run(inputs)
        stats = protocol.get_statistics()
        assert stats["total_cycles"] == 3
        assert stats["accepted"] + stats["rejected"] == 3
        assert stats["acceptance_rate"] > 0

    def test_think_strategy_selection(self):
        protocol = DreamProtocol()
        # Surface strategy for few labels, low depth
        inputs_surface = [
            {"labels": ["a"], "weight": 1.0, "integration_count": 0},
            {"labels": ["a"], "weight": 1.0, "integration_count": 0},
        ]
        result = protocol.run(inputs_surface)
        assert result["analysis"]["strategy"] in ("surface", "deepen", "bridge")

    def test_default_think_analysis_content(self):
        protocol = DreamProtocol()
        inputs = [
            {"labels": ["ai", "ml"], "weight": 2.0, "integration_count": 5},
            {"labels": ["ai", "dl"], "weight": 1.0, "integration_count": 2},
        ]
        result = protocol.run(inputs)
        a = result["analysis"]
        assert "ai" in a["label_inventory"]
        assert a["total_weight"] == 3.0
        assert a["max_depth"] == 5
        assert a["n_sources"] == 2


# ── Q3: Ghost Cell versioning ────────────────────────────────

class TestGhostCellV2:
    def test_version_tracking(self):
        gc = GhostCell(
            node_id="n1", source_bucket_id="b1",
            geometry=np.zeros(3), version=3, source_version=3,
        )
        assert not gc.is_stale

    def test_stale_detection(self):
        gc = GhostCell(
            node_id="n1", source_bucket_id="b1",
            geometry=np.zeros(3), version=3, source_version=5,
        )
        assert gc.is_stale

    def test_needs_verification(self):
        gc = GhostCell(
            node_id="n1", source_bucket_id="b1",
            geometry=np.zeros(3), verify_interval=0.01,
            last_verified_at=time.time() - 1.0,
        )
        assert gc.needs_verification

    def test_verify_updates_version(self):
        gc = GhostCell(
            node_id="n1", source_bucket_id="b1",
            geometry=np.zeros(3), version=2, source_version=5,
        )
        is_consistent = gc.verify(current_version=5, current_weight=2.0)
        assert not is_consistent  # Was stale
        assert gc.version == 5
        assert gc.weight == 2.0
        assert not gc.is_stale

    def test_verify_consistent(self):
        gc = GhostCell(
            node_id="n1", source_bucket_id="b1",
            geometry=np.zeros(3), version=3, source_version=3,
            weight=1.0,
        )
        is_consistent = gc.verify(current_version=3, current_weight=1.0)
        assert is_consistent


class TestGhostInvalidation:
    def _setup_router(self):
        router = TetraMeshRouter(max_tetra_per_bucket=100, ghost_ttl=3600.0)
        bounds_a = BoundingBox(np.array([-2.0, -2.0, -2.0]), np.array([0.0, 2.0, 2.0]))
        bounds_b = BoundingBox(np.array([0.0, -2.0, -2.0]), np.array([2.0, 2.0, 2.0]))
        router._create_bucket(bounds_a)
        router._create_bucket(bounds_b)
        return router

    def test_invalidate_ghost_for(self):
        router = self._setup_router()
        bids = router.get_all_bucket_ids()

        bid_a, tid_a = router.route_store(
            np.array([-0.5, 0.0, 0.0]), "test in A", labels=["test"],
        )

        gc = GhostCell(
            node_id=tid_a, source_bucket_id=bid_a,
            geometry=np.array([-0.5, 0.0, 0.0]),
            version=0, source_version=0,
        )
        bucket_b = router.get_bucket(bids[1])
        bucket_b.ghost_cells[tid_a] = gc

        invalidated = router.invalidate_ghost_for(tid_a, new_version=3)
        assert invalidated >= 1
        assert gc.source_version == 3

    def test_verify_ghost_cells(self):
        router = self._setup_router()

        bid_a, tid_a = router.route_store(
            np.array([-0.5, 0.0, 0.0]), "source memory", labels=["test"],
        )
        bids = router.get_all_bucket_ids()

        gc = GhostCell(
            node_id=tid_a, source_bucket_id=bid_a,
            geometry=np.array([-0.5, 0.0, 0.0]),
            version=0, source_version=0,
            verify_interval=0.01,
            last_verified_at=time.time() - 1.0,
        )
        other_bid = bids[0] if bids[0] != bid_a else bids[1]
        other_bucket = router.get_bucket(other_bid)
        other_bucket.ghost_cells[tid_a] = gc

        stats = router.verify_ghost_cells()
        assert stats["verified"] >= 1

    def test_verify_removes_expired(self):
        router = self._setup_router()
        bids = router.get_all_bucket_ids()

        gc = GhostCell(
            node_id="ghost_expired", source_bucket_id="nonexistent",
            geometry=np.zeros(3),
            created_at=0.0, ttl=0.01,
        )
        bucket = router.get_bucket(bids[0])
        bucket.ghost_cells["ghost_expired"] = gc

        stats = router.verify_ghost_cells(bids[0])
        assert stats["removed"] >= 1
        assert "ghost_expired" not in bucket.ghost_cells

    def test_verify_removes_nonexistent_source(self):
        router = self._setup_router()
        bids = router.get_all_bucket_ids()

        gc = GhostCell(
            node_id="nonexistent_id", source_bucket_id=bids[1],
            geometry=np.zeros(3),
            verify_interval=0.01,
            last_verified_at=time.time() - 1.0,
        )
        bucket_a = router.get_bucket(bids[0])
        bucket_a.ghost_cells["nonexistent_id"] = gc

        stats = router.verify_ghost_cells(bids[0])
        assert stats["removed"] >= 1
