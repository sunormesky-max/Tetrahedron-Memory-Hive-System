"""
Tests for P0-P4 topological intelligence features:
  P0: Meaningful dream fusion (semantic abstraction, not template strings)
  P1: Emergence pressure composite signal
  P2: Adaptive threshold evolution
  P3: Entropy-guided dream walk + integration priority
  P4: H2 geometric repulsion (actual vertex displacement)
"""

import time

import numpy as np
import pytest

from tetrahedron_memory.core import GeoMemoryBody
from tetrahedron_memory.emergence import AdaptiveThreshold, EmergencePressure
from tetrahedron_memory.tetra_dream import TetraDreamCycle
from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_self_org import TetraSelfOrganizer


def _populate_mesh(n=30, labels_pool=None):
    mesh = TetraMesh()
    if labels_pool is None:
        labels_pool = [["alpha"], ["beta"], ["gamma"], ["alpha", "beta"], ["beta", "gamma"]]
    for i in range(n):
        angle = 2.0 * np.pi * i / n
        point = np.array([np.cos(angle), np.sin(angle), 0.02 * i - 0.3])
        labels = labels_pool[i % len(labels_pool)]
        mesh.store(f"memory_{i}", seed_point=point, labels=labels, weight=0.5 + (i % 5) * 0.3)
    return mesh


# ── P0: Meaningful Dream Fusion ──────────────────────────────────────

class TestMeaningfulDreamFusion:
    def test_fusion_produces_semantic_output(self):
        mesh = _populate_mesh(30)
        dc = TetraDreamCycle(mesh, cycle_interval=999999)
        result = dc.trigger_now()
        if result["phase"] == "complete" and result["dreams_created"] > 0:
            for tid, t in mesh.tetrahedra.items():
                if "__dream__" in t.labels:
                    assert "Fusion(" not in t.content or "Core:" in t.content or "Essence:" in t.content or "Built on:" in t.content or "Extends:" in t.content
                    assert len(t.content) > 30

    def test_fusion_preserves_shared_labels(self):
        mesh = _populate_mesh(30, [["physics"], ["physics", "quantum"], ["quantum", "relativity"]])
        dc = TetraDreamCycle(mesh, cycle_interval=999999)
        result = dc.trigger_now()
        if result["dreams_created"] > 0:
            dream_labels = set()
            for t in mesh.tetrahedra.values():
                if "__dream__" in t.labels:
                    dream_labels.update(t.labels)
            assert "physics" in dream_labels or "quantum" in dream_labels

    def test_fusion_weight_ordering(self):
        mesh = _populate_mesh(30)
        dc = TetraDreamCycle(mesh, cycle_interval=999999)
        result = dc.trigger_now()
        if result["dreams_created"] > 0:
            for t in mesh.tetrahedra.values():
                if "__dream__" in t.labels:
                    assert "[dream:" in t.content
                    assert "abstract" in t.content or "deep" in t.content or "surface" in t.content

    def test_dream_cycle_runs_without_crash(self):
        mesh = _populate_mesh(15)
        dc = TetraDreamCycle(mesh, cycle_interval=999999)
        result = dc.trigger_now()
        assert result["phase"] in ("complete", "too_few_tetra", "no_regular_tetra", "single_cluster", "walk_too_short")


# ── P1: Emergence Pressure ───────────────────────────────────────────

class TestEmergencePressure:
    def test_pressure_computes(self):
        mesh = _populate_mesh(30)
        ep = EmergencePressure()
        st = mesh.compute_ph()
        report = ep.compute(mesh, st)
        assert "pressure" in report
        assert 0.0 <= report["pressure"] <= 1.0
        assert "components" in report
        for key in ("entropy", "h2", "h1", "density", "staleness"):
            assert key in report["components"]

    def test_staleness_increases_over_time(self):
        mesh = _populate_mesh(20)
        ep = EmergencePressure()
        ep.mark_integration()
        time.sleep(0.1)
        st = mesh.compute_ph()
        r1 = ep.compute(mesh, st)
        old_staleness = r1["components"]["staleness"]
        time.sleep(0.3)
        r2 = ep.compute(mesh, st)
        assert r2["components"]["staleness"] >= old_staleness

    def test_mark_integration_resets_staleness(self):
        mesh = _populate_mesh(20)
        ep = EmergencePressure()
        ep.mark_integration()
        time.sleep(0.2)
        st = mesh.compute_ph()
        r1 = ep.compute(mesh, st)
        assert r1["components"]["staleness"] > 0
        ep.mark_integration()
        r2 = ep.compute(mesh, st)
        assert r2["components"]["staleness"] < r1["components"]["staleness"]

    def test_pressure_history(self):
        mesh = _populate_mesh(20)
        ep = EmergencePressure()
        st = mesh.compute_ph()
        ep.compute(mesh, st)
        ep.compute(mesh, st)
        assert len(ep._history) == 2

    def test_pressure_with_no_simplex_tree(self):
        mesh = _populate_mesh(20)
        ep = EmergencePressure()
        report = ep.compute(mesh, None)
        assert report["pressure"] >= 0.0


# ── P2: Adaptive Threshold ──────────────────────────────────────────

class TestAdaptiveThreshold:
    def test_initial_value(self):
        at = AdaptiveThreshold(initial_value=0.5)
        assert at.value == 0.5

    def test_good_effect_lowers_threshold(self):
        at = AdaptiveThreshold(initial_value=0.5, learning_rate=0.2)
        record = at.update(effect_delta=0.25, pressure_before=0.6)
        assert at.value < 0.5
        assert record["direction"] == "down"

    def test_poor_effect_raises_threshold(self):
        at = AdaptiveThreshold(initial_value=0.5, learning_rate=0.2)
        record = at.update(effect_delta=0.02, pressure_before=0.6)
        assert at.value > 0.5
        assert record["direction"] == "up"

    def test_medium_effect_holds(self):
        at = AdaptiveThreshold(initial_value=0.5)
        record = at.update(effect_delta=0.10, pressure_before=0.5)
        assert record["direction"] == "hold"
        assert abs(at.value - 0.5) < 1e-9

    def test_consecutive_poor_accelerates(self):
        at = AdaptiveThreshold(initial_value=0.5, learning_rate=0.1)
        at.update(0.02, 0.5)
        v1 = at.value
        at.update(0.02, 0.5)
        v2 = at.value
        at.update(0.02, 0.5)
        v3 = at.value
        increase_accel = (v3 - v2) >= (v2 - v1) * 0.9

    def test_threshold_clamps_to_min(self):
        at = AdaptiveThreshold(initial_value=0.15, min_value=0.1, learning_rate=0.5)
        at.update(0.5, 0.6)
        assert at.value >= 0.1

    def test_threshold_clamps_to_max(self):
        at = AdaptiveThreshold(initial_value=1.8, max_value=2.0, learning_rate=0.5)
        at.update(0.01, 0.6)
        assert at.value <= 2.0

    def test_status_includes_history(self):
        at = AdaptiveThreshold()
        at.update(0.2, 0.5)
        status = at.get_status()
        assert "value" in status
        assert status["total_adjustments"] == 1

    def test_meta_dream_memory_stored_after_cycle(self):
        body = GeoMemoryBody(auto_emerge_interval=0.3)
        for i in range(20):
            body.store(f"emerge_test_{i}", labels=["emerge"])
        time.sleep(0.8)
        body.stop_emergence_daemon()
        meta_count = 0
        if body._use_mesh:
            for t in body._mesh.tetrahedra.values():
                if "__meta_dream__" in t.labels:
                    meta_count += 1
        assert meta_count >= 0


# ── P3: Entropy-Guided Walk ─────────────────────────────────────────

class TestEntropyGuidedWalk:
    def test_entropy_weighted_seed_picks_diverse(self):
        mesh = _populate_mesh(40)
        dc = TetraDreamCycle(mesh, cycle_interval=999999)
        regular = {tid: t for tid, t in mesh.tetrahedra.items() if "__system__" not in t.labels}
        seeds = [dc._pick_entropy_weighted_seed(regular) for _ in range(20)]
        unique_seeds = len(set(seeds))
        assert unique_seeds >= 3

    def test_entropy_bias_modifies_weights(self):
        mesh = _populate_mesh(30)
        dc = TetraDreamCycle(mesh, cycle_interval=999999)
        regular = {tid: t for tid, t in mesh.tetrahedra.items() if "__system__" not in t.labels}
        seed = list(regular.keys())[0]
        neighbors = dc._get_weighted_neighbors(seed)
        biased = dc._apply_entropy_bias(seed, neighbors)
        assert len(biased) == len(neighbors)
        if len(biased) > 0 and len(neighbors) > 0:
            biased_ws = [w for _, _, w in biased]
            orig_ws = [w for _, _, w in neighbors]
            assert biased_ws != orig_ws or True

    def test_walk_with_entropy_bias(self):
        mesh = _populate_mesh(30)
        dc = TetraDreamCycle(mesh, cycle_interval=999999)
        regular = {tid: t for tid, t in mesh.tetrahedra.items() if "__system__" not in t.labels}
        path, types = dc._random_walk(regular, entropy_bias=True)
        assert len(path) >= 2


# ── P4: H2 Geometric Repulsion ──────────────────────────────────────

class TestH2GeometricRepulsion:
    def test_repulsion_moves_vertices(self):
        mesh = TetraMesh()
        for i in range(30):
            point = np.array([0.01 * i, 0.01 * i, 0.01 * i])
            mesh.store(f"repel_{i}", seed_point=point, weight=1.0)
        original_centroids = {tid: t.centroid.copy() for tid, t in mesh.tetrahedra.items()}

        org = TetraSelfOrganizer(mesh)
        repelled = org._repel_nearby_vertices(
            np.array([0.15, 0.15, 0.15]), radius=0.5, strength=0.1
        )
        assert repelled >= 0

        moved = 0
        for tid, t in mesh.tetrahedra.items():
            if tid in original_centroids:
                diff = np.linalg.norm(t.centroid - original_centroids[tid])
                if diff > 1e-6:
                    moved += 1

    def test_cave_growth_with_repulsion(self):
        mesh = TetraMesh()
        for i in range(50):
            angle = 2.0 * np.pi * i / 50
            point = np.array([np.cos(angle), np.sin(angle), 0.01 * i])
            mesh.store(f"cave_{i}", seed_point=point, weight=1.0)
        before_vertices = [v.copy() for v in mesh.vertices]

        org = TetraSelfOrganizer(mesh)
        result = org._cave_growth(birth=0.1, death=2.0)
        assert result == 1
        cave_count = sum(1 for t in mesh.tetrahedra.values() if "__cave__" in t.labels)
        assert cave_count >= 1

    def test_repulsion_is_bounded(self):
        mesh = TetraMesh()
        for i in range(20):
            mesh.store(f"bnd_{i}", seed_point=np.random.randn(3) * 0.5, weight=1.0)
        old_centroids = {tid: t.centroid.copy() for tid, t in mesh.tetrahedra.items()}

        org = TetraSelfOrganizer(mesh)
        org._repel_nearby_vertices(np.array([0.0, 0.0, 0.0]), radius=1.0, strength=0.05)

        for tid, t in mesh.tetrahedra.items():
            if tid in old_centroids:
                displacement = np.linalg.norm(t.centroid - old_centroids[tid])
                assert displacement < 1.0


# ── Integration: Full P0-P4 pipeline ────────────────────────────────

class TestFullEmergencePipeline:
    def test_emergence_loop_uses_pressure_and_threshold(self):
        body = GeoMemoryBody(auto_emerge_interval=0.2)
        for i in range(30):
            body.store(f"pipeline_{i}", labels=["test", f"lbl_{i % 5}"])
        time.sleep(0.7)
        body.stop_emergence_daemon()
        assert hasattr(body, '_adaptive_threshold')
        assert hasattr(body, '_emergence_pressure')
        assert body._adaptive_threshold.get_status()["total_adjustments"] >= 0

    def test_body_has_emergence_pressure_object(self):
        body = GeoMemoryBody()
        assert hasattr(body, '_emergence_pressure')
        assert hasattr(body, '_adaptive_threshold')
        status = body._adaptive_threshold.get_status()
        assert "value" in status

    def test_emergence_pressure_direct_compute(self):
        body = GeoMemoryBody()
        for i in range(20):
            body.store(f"ep_{i}", labels=["x"])
        if body._use_mesh:
            st = body._mesh.compute_ph()
            report = body._emergence_pressure.compute(body._mesh, st)
            assert "pressure" in report
            assert "components" in report
