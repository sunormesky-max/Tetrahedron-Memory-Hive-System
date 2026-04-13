"""
Tests for three critical gaps identified in system assessment:
  GAP-1: Zigzag + Mapping Cone iterative bidirectional modeling
  GAP-2: Dynamic adaptive resolution pyramid with closed-loop feedback
  GAP-3: Eternity principle strict audit mechanism
"""

import threading
import time

import numpy as np
import pytest

from tetrahedron_memory.core import GeoMemoryBody
from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_dream import TetraDreamCycle
from tetrahedron_memory.zigzag_persistence import (
    ZigzagTracker,
    PersistenceSnapshot,
    MappingConeRecord,
)
from tetrahedron_memory.resolution_pyramid import ResolutionPyramid
from tetrahedron_memory.eternity_audit import EternityAudit


# ─── GAP-1: Mapping Cone + Iterative Bidirectional Modeling ───


class TestMappingConeConstruction:
    def test_mapping_cone_basic(self):
        mesh = TetraMesh()
        for i in range(20):
            mesh.store(f"mem {i}", seed_point=np.random.randn(3))

        tracker = ZigzagTracker()
        pre = tracker.record_snapshot(mesh)

        mesh.store("dream product", seed_point=np.random.randn(3), labels=["__dream__"])

        post = tracker.record_snapshot(mesh)
        cone = tracker.construct_mapping_cone(pre, post, dream_tetra_created=["dream_id"])

        assert isinstance(cone, MappingConeRecord)
        assert cone.pre_snapshot is pre
        assert cone.post_snapshot is post
        assert len(cone.dream_tetra_created) == 1

    def test_forward_and_backward_maps(self):
        mesh = TetraMesh()
        for i in range(30):
            mesh.store(f"mem {i}", seed_point=np.random.randn(3), labels=[f"l{i % 3}"])

        tracker = ZigzagTracker()
        pre = tracker.record_snapshot(mesh)

        for i in range(3):
            mesh.store(f"dream {i}", seed_point=np.random.randn(3), labels=["__dream__"])

        post = tracker.record_snapshot(mesh)
        cone = tracker.construct_mapping_cone(pre, post)

        has_forward = any(len(v) > 0 for v in cone.forward_map.values())
        has_backward = any(len(v) > 0 for v in cone.backward_map.values())
        assert has_forward or has_backward

    def test_stability_certification(self):
        mesh = TetraMesh()
        for i in range(20):
            mesh.store(f"mem {i}", seed_point=np.random.randn(3))

        tracker = ZigzagTracker()
        pre = tracker.record_snapshot(mesh)
        post = tracker.record_snapshot(mesh)

        cone = tracker.construct_mapping_cone(pre, post)

        stable = cone.get_stable_features()
        assert isinstance(stable, list)

    def test_iterative_cone_accumulation(self):
        mesh = TetraMesh()
        tracker = ZigzagTracker()

        for cycle in range(3):
            for i in range(5):
                mesh.store(f"mem_{cycle}_{i}", seed_point=np.random.randn(3))

            pre = tracker.record_snapshot(mesh)
            mesh.store(f"dream_{cycle}", seed_point=np.random.randn(3), labels=["__dream__"])
            post = tracker.record_snapshot(mesh)
            tracker.construct_mapping_cone(pre, post, dream_tetra_created=[f"d_{cycle}"])

        history = tracker.get_mapping_cone_history(count=10)
        assert len(history) == 3
        for h in history:
            assert "entropy_delta" in h
            assert "stable_h0" in h

    def test_dream_guidance_from_cones(self):
        mesh = TetraMesh()
        tracker = ZigzagTracker()

        for cycle in range(5):
            for i in range(10):
                mesh.store(f"mem_{cycle}_{i}", seed_point=np.random.randn(3), labels=[f"l{i % 4}"])
            pre = tracker.record_snapshot(mesh)
            mesh.store(f"dream_{cycle}", seed_point=np.random.randn(3), labels=["__dream__"])
            post = tracker.record_snapshot(mesh)
            tracker.construct_mapping_cone(pre, post)

        guidance = tracker.get_dream_guidance()
        assert guidance["guidance"] == "targeted_dream"
        assert "focus_dim" in guidance
        assert "dim_instability" in guidance
        assert "expected_benefit" in guidance


class TestDreamWithMappingCone:
    def test_dream_cycle_produces_cone(self):
        mesh = TetraMesh()
        for i in range(30):
            mesh.store(f"mem {i}", seed_point=np.random.randn(3), labels=[f"l{i % 3}"])

        tracker = ZigzagTracker()
        dc = TetraDreamCycle(mesh, zigzag_tracker=tracker)
        stats = dc.trigger_now()

        if stats["phase"] == "complete":
            assert stats.get("mapping_cone") is not None
            assert "cycle_id" in stats["mapping_cone"]
            assert "stable_features" in stats["mapping_cone"]

    def test_dream_guidance_in_stats(self):
        mesh = TetraMesh()
        for i in range(30):
            mesh.store(f"mem {i}", seed_point=np.random.randn(3))

        tracker = ZigzagTracker()
        dc = TetraDreamCycle(mesh, zigzag_tracker=tracker)
        stats = dc.trigger_now()

        assert stats.get("dream_guidance") is not None

    def test_iterative_dream_cones(self):
        mesh = TetraMesh()
        tracker = ZigzagTracker()
        dc = TetraDreamCycle(mesh, zigzag_tracker=tracker)

        for _ in range(3):
            for i in range(10):
                mesh.store(f"mem_{i}_{time.time()}", seed_point=np.random.randn(3))
            dc.trigger_now()

        history = tracker.get_mapping_cone_history(count=10)
        assert len(history) >= 1


# ─── GAP-2: Dynamic Adaptive Resolution Pyramid ───


class TestAdaptivePyramid:
    def test_initial_adaptive_params(self):
        pyramid = ResolutionPyramid()
        status = pyramid.get_adaptive_status()
        assert "adaptive_max_levels" in status
        assert "adaptive_coarsening" in status

    def test_feedback_adapts_parameters(self):
        pyramid = ResolutionPyramid()
        initial_coarsening = pyramid._adaptive_coarsening

        for _ in range(5):
            pyramid.record_dream_feedback(
                entropy_delta=0.25, dreams_created=2, dreams_reintegrated=1
            )

        new_coarsening = pyramid._adaptive_coarsening
        assert new_coarsening <= initial_coarsening + 0.05

    def test_negative_feedback_increases_coarsening(self):
        pyramid = ResolutionPyramid()
        initial = pyramid._adaptive_coarsening

        for _ in range(5):
            pyramid.record_dream_feedback(
                entropy_delta=0.01, dreams_created=0, dreams_reintegrated=0
            )

        assert pyramid._adaptive_coarsening >= initial

    def test_query_feedback_tracking(self):
        pyramid = ResolutionPyramid()
        pyramid.record_query_feedback(level=1, hit=True)
        pyramid.record_query_feedback(level=1, hit=True)
        pyramid.record_query_feedback(level=1, hit=False)

        status = pyramid.get_adaptive_status()
        assert "1" in status["query_hit_rates"]
        assert abs(status["query_hit_rates"]["1"] - 0.6667) < 0.01

    def test_pyramid_builds_with_adaptive_params(self):
        mesh = TetraMesh()
        for i in range(50):
            mesh.store(f"mem {i}", seed_point=np.random.randn(3))

        pyramid = ResolutionPyramid()
        pyramid._adaptive_max_levels = 3
        result = pyramid.build(mesh)
        assert result["levels"] >= 1

    def test_closed_loop_feedback_chain(self):
        pyramid = ResolutionPyramid()
        for i in range(10):
            delta = 0.2 if i < 5 else 0.01
            pyramid.record_dream_feedback(
                entropy_delta=delta,
                dreams_created=max(0, int(delta * 10)),
                dreams_reintegrated=0,
            )

        status = pyramid.get_adaptive_status()
        assert status["feedback_count"] == 10


# ─── GAP-3: Eternity Principle Strict Audit ───


class TestEternityAudit:
    def test_record_store_creates_entry(self):
        audit = EternityAudit()
        audit.record_store("tid1", "hello world")
        status = audit.get_status()
        assert status["total_entries"] == 1
        assert status["total_tracked_ids"] == 1

    def test_verify_all_present(self):
        mesh = TetraMesh()
        audit = EternityAudit()

        ids = []
        for i in range(10):
            tid = mesh.store(f"mem {i}", seed_point=np.random.randn(3))
            audit.record_store(tid, f"mem {i}")
            ids.append(tid)

        result = audit.verify(mesh)
        assert result["verified"] is True
        assert result["total_stored"] == 10
        assert result["total_violations"] == 0

    def test_verify_detects_missing(self):
        mesh = TetraMesh()
        audit = EternityAudit()

        tid1 = mesh.store("keep", seed_point=np.random.randn(3))
        audit.record_store(tid1, "keep")

        audit.record_store("phantom_id", "I was never in mesh")

        result = audit.verify(mesh)
        assert result["verified"] is False
        assert result["total_violations"] == 1
        assert "phantom_id" in result["violations"]

    def test_merge_preserves_content(self):
        mesh = TetraMesh()
        audit = EternityAudit()

        tid1 = mesh.store("alpha", seed_point=np.random.randn(3))
        tid2 = mesh.store("beta", seed_point=np.random.randn(3))
        audit.record_store(tid1, "alpha")
        audit.record_store(tid2, "beta")

        merged_id = mesh.edge_contraction(tid1, tid2)
        if merged_id:
            audit.record_merge([tid1, tid2], merged_id, "merged content")

        result = audit.verify(mesh)
        assert result["verified"] is True

    def test_preservation_chain(self):
        audit = EternityAudit()
        audit.record_store("s1", "original")
        audit.record_merge(["s1"], "m1", "merged")
        audit.record_transform("m1", "t1", "integrate", "transformed")

        chain = audit.get_preservation_chain("s1")
        assert "s1" in chain
        assert "m1" in chain
        assert "t1" in chain

    def test_audit_trail(self):
        audit = EternityAudit()
        audit.record_store("x1", "hello")
        audit.record_dream("d1", "dream content", ["x1"])
        audit.record_reintegration("d1")

        trail = audit.get_audit_trail("x1")
        assert len(trail) >= 2
        operations = [t["operation"] for t in trail]
        assert "store" in operations
        assert "dream_create" in operations

    def test_dream_audit_preserves_origin(self):
        mesh = TetraMesh()
        audit = EternityAudit()

        s1 = mesh.store("source1", seed_point=np.random.randn(3))
        s2 = mesh.store("source2", seed_point=np.random.randn(3))
        audit.record_store(s1, "source1")
        audit.record_store(s2, "source2")

        dream_id = mesh.store(
            "[dream] synthesized",
            seed_point=np.random.randn(3),
            labels=["__dream__"],
        )
        audit.record_dream(dream_id, "[dream] synthesized", [s1, s2])

        result = audit.verify(mesh)
        assert result["verified"] is True

        chain = audit.get_preservation_chain(s1)
        assert dream_id in chain


class TestEternityViaGeoMemoryBody:
    def test_body_verify_eternity(self):
        body = GeoMemoryBody()
        for i in range(5):
            body.store(f"memory {i}")

        result = body.verify_eternity()
        assert result["verified"] is True

    def test_body_eternity_status(self):
        body = GeoMemoryBody()
        body.store("test")
        status = body.get_eternity_status()
        assert status["total_entries"] >= 1

    def test_body_eternity_trail(self):
        body = GeoMemoryBody()
        tid = body.store("test memory")
        trail = body.get_eternity_trail(tid)
        assert len(trail) >= 1
        assert trail[0]["operation"] == "store"
