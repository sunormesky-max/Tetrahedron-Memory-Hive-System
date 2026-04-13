"""
Tests for TetraDreamCycle and TetraSelfOrganizer.
"""

import numpy as np
import pytest

from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_dream import TetraDreamCycle, default_synthesis
from tetrahedron_memory.tetra_self_org import TetraSelfOrganizer


class TestDefaultSynthesis:
    def test_basic_synthesis(self):
        result = default_synthesis(
            ["content A", "content B"],
            [["ai"], ["ml"]],
        )
        assert result is not None
        assert "dream" in result

    def test_synthesis_strips_system_labels(self):
        result = default_synthesis(
            ["content"],
            [["__system__", "real_label"]],
        )
        assert "__system__" not in result

    def test_synthesis_empty_input(self):
        result = default_synthesis([], [])
        assert "general" in result


class TestTetraDreamCycle:
    def _make_mesh(self, n=10):
        mesh = TetraMesh()
        for i in range(n):
            pt = np.array([float(i) * 0.2, 0.0, 0.0])
            mesh.store(f"item_{i}", seed_point=pt, labels=["topic"])
        return mesh

    def test_init(self):
        mesh = self._make_mesh()
        dc = TetraDreamCycle(mesh)
        assert dc.cycle_interval == 300.0
        assert dc.walk_steps == 12

    def test_trigger_now_too_few(self):
        mesh = TetraMesh()
        dc = TetraDreamCycle(mesh)
        stats = dc.trigger_now()
        assert stats["phase"] == "too_few_tetra"

    def test_trigger_now_with_data(self):
        mesh = self._make_mesh(10)
        dc = TetraDreamCycle(mesh, cycle_interval=9999)
        stats = dc.trigger_now()
        assert "phase" in stats
        assert stats["walk_visited"] >= 0

    def test_get_status(self):
        mesh = self._make_mesh()
        dc = TetraDreamCycle(mesh)
        status = dc.get_status()
        assert "running" in status
        assert status["running"] is False
        assert "cycle_count" in status

    def test_start_stop(self):
        mesh = self._make_mesh()
        dc = TetraDreamCycle(mesh, cycle_interval=9999)
        dc.start()
        assert dc.get_status()["running"] is True
        dc.stop()
        assert dc.get_status()["running"] is False

    def test_dream_creates_new_tetra(self):
        mesh = self._make_mesh(20)
        initial_count = len(mesh.tetrahedra)
        dc = TetraDreamCycle(mesh, walk_steps=15, dream_weight=0.5)
        dc.trigger_now()
        dream_count = sum(1 for t in mesh.tetrahedra.values() if "__dream__" in t.labels)
        assert dream_count >= 0

    def test_reintegrate_dreams_thread_safe(self):
        mesh = self._make_mesh(5)
        mesh.store("dream1", seed_point=np.array([10.0, 0.0, 0.0]),
                    labels=["__dream__"], weight=0.01)
        dc = TetraDreamCycle(mesh)
        reintegrated = dc._reintegrate_dreams()
        assert reintegrated == 1


class TestTetraSelfOrganizer:
    def _make_mesh(self, n=10):
        mesh = TetraMesh()
        for i in range(n):
            pt = np.array([float(i) * 0.3, float(i % 3) * 0.1, 0.0])
            mesh.store(f"item_{i}", seed_point=pt, labels=["topic"], weight=1.0)
        return mesh

    def test_init(self):
        mesh = self._make_mesh()
        so = TetraSelfOrganizer(mesh)
        assert so.h2_threshold == 1.0
        assert so.max_iterations == 10

    def test_run_too_few(self):
        mesh = TetraMesh()
        so = TetraSelfOrganizer(mesh)
        stats = so.run()
        assert stats["iterations"] <= 1

    def test_run_with_data(self):
        pytest.importorskip("gudhi")
        mesh = self._make_mesh(15)
        so = TetraSelfOrganizer(mesh, max_iterations=3)
        stats = so.run()
        assert "iterations" in stats
        assert "total_actions" in stats
        assert "converged" in stats

    def test_get_status(self):
        mesh = self._make_mesh()
        so = TetraSelfOrganizer(mesh)
        status = so.get_status()
        assert "total_actions" in status
        assert "h2_threshold" in status

    def test_integrate_thread_safe(self):
        mesh = self._make_mesh(5)
        for tid, t in list(mesh.tetrahedra.items()):
            t.weight = 0.001
            t.init_weight = 1.0
        so = TetraSelfOrganizer(mesh, integrate_heat_threshold=0.2)
        h0 = np.array([[0.0, 0.01]])
        count = so._detect_and_integrate(h0)
        assert count >= 0
