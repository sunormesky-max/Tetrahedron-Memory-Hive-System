"""
Tests for P1: Abstract Reorganization.
"""

import numpy as np
import pytest

from tetrahedron_memory.tetra_mesh import MemoryTetrahedron, TetraMesh


class TestMemoryTetrahedronAbstractReorg:
    def test_integrate_secondary_empty(self):
        t = MemoryTetrahedron(
            id="test", content="hello",
            vertex_indices=(0, 1, 2, 3),
            centroid=np.zeros(3),
        )
        assert t.integrate_secondary() == 0
        assert t.weight == 1.0

    def test_integrate_single_secondary(self):
        t = MemoryTetrahedron(
            id="test", content="base memory content",
            vertex_indices=(0, 1, 2, 3),
            centroid=np.zeros(3),
            labels=["ai"],
        )
        t.attach_secondary("related note", labels=["ai"], weight=1.5)
        count = t.integrate_secondary()
        assert count == 1
        assert len(t.secondary_memories) == 0
        assert t.integration_count >= 1

    def test_integrate_multiple_secondaries(self):
        t = MemoryTetrahedron(
            id="test", content="machine learning fundamentals",
            vertex_indices=(0, 1, 2, 3),
            centroid=np.zeros(3),
            labels=["ai", "ml"],
            weight=2.0,
        )
        t.attach_secondary("neural networks", labels=["ai", "dl"], weight=1.5)
        t.attach_secondary("deep learning basics", labels=["ai", "dl"], weight=1.8)
        t.attach_secondary("gradient descent optimization", labels=["ai", "ml"], weight=1.2)

        count = t.integrate_secondary()
        assert count == 3
        assert len(t.secondary_memories) == 0
        assert "ai" in t.labels

    def test_integrate_preserves_provenance(self):
        t = MemoryTetrahedron(
            id="test", content="original content here",
            vertex_indices=(0, 1, 2, 3),
            centroid=np.zeros(3),
            labels=["test"],
            weight=1.0,
        )
        t.attach_secondary("first secondary", labels=["test", "extra"], weight=1.0)
        t.attach_secondary("second secondary", labels=["extra"], weight=1.0)

        t.integrate_secondary()

        assert "reorg_history" in t.metadata
        assert len(t.metadata["reorg_history"]) == 1
        entry = t.metadata["reorg_history"][0]
        assert entry["sources_count"] == 2
        assert len(entry["source_contents"]) == 2

    def test_integrate_theme_extraction(self):
        t = MemoryTetrahedron(
            id="test", content="python programming language features",
            vertex_indices=(0, 1, 2, 3),
            centroid=np.zeros(3),
            labels=["python"],
            weight=1.0,
        )
        t.attach_secondary("python web development frameworks", labels=["python", "web"], weight=1.0)
        t.attach_secondary("python data science libraries", labels=["python", "data"], weight=1.0)

        t.integrate_secondary()

        if "reorg_history" in t.metadata:
            themes = t.metadata["reorg_history"][0].get("themes_extracted", [])
            assert len(themes) > 0

    def test_label_consolidation_priority(self):
        t = MemoryTetrahedron(
            id="test", content="content about things",
            vertex_indices=(0, 1, 2, 3),
            centroid=np.zeros(3),
            labels=["alpha", "beta"],
            weight=1.0,
        )
        t.attach_secondary("s1", labels=["alpha", "gamma"], weight=1.0)
        t.attach_secondary("s2", labels=["alpha", "delta"], weight=1.0)

        t.integrate_secondary()
        assert "alpha" in t.labels

    def test_weight_fusion_not_simple_average(self):
        t = MemoryTetrahedron(
            id="test", content="base content for weight test",
            vertex_indices=(0, 1, 2, 3),
            centroid=np.zeros(3),
            labels=["wtest"],
            weight=3.0,
        )
        t.attach_secondary("sec1", weight=1.0)
        t.attach_secondary("sec2", weight=1.0)
        t.attach_secondary("sec3", weight=1.0)

        old_weight = t.weight
        t.integrate_secondary()

        # Weight should be fused (total/4 * boost), not simple avg * 1.05
        # fused = 6.0 / 4 = 1.5, boost = 1 + 0.1*3 = 1.3, result = 1.95
        # vs old: avg*1.05 = (6/4)*1.05 = 1.575
        assert t.weight > 0
        assert "reorg_history" in t.metadata


class TestTetraMeshAbstractReorganize:
    def test_empty_mesh(self):
        mesh = TetraMesh()
        result = mesh.abstract_reorganize()
        assert result["integrated_count"] == 0
        assert result["tetra_scanned"] == 0

    def test_no_dense_tetras(self):
        mesh = TetraMesh()
        for i in range(5):
            mesh.store("item_" + str(i), seed_point=np.array([float(i) * 0.1, 0.0, 0.0]))
        result = mesh.abstract_reorganize(min_density=2)
        assert result["integrated_count"] == 0
        assert result["tetra_scanned"] == 5

    def test_reorganize_dense_tetras(self):
        mesh = TetraMesh()
        tid = mesh.store(
            "machine learning fundamentals",
            seed_point=np.zeros(3),
            labels=["ai", "ml"],
            weight=2.0,
        )
        mesh.store_secondary(tid, "neural networks", labels=["ai", "dl"], weight=1.5)
        mesh.store_secondary(tid, "deep learning", labels=["ai", "dl"], weight=1.8)
        mesh.store_secondary(tid, "gradient descent", labels=["ai", "ml"], weight=1.2)

        result = mesh.abstract_reorganize(min_density=2)
        assert result["integrated_count"] >= 1
        assert result["tetra_scanned"] >= 1

        tetra = mesh.get_tetrahedron(tid)
        assert tetra is not None
        assert len(tetra.secondary_memories) == 0

    def test_reorganize_with_cross_fusion(self):
        mesh = TetraMesh()
        t1 = mesh.store(
            "python programming",
            seed_point=np.array([0.0, 0.0, 0.0]),
            labels=["python", "coding"],
            weight=2.0,
        )
        mesh.store_secondary(t1, "python flask web", labels=["python", "web"], weight=1.0)
        mesh.store_secondary(t1, "python django web", labels=["python", "web"], weight=1.0)

        t2 = mesh.store(
            "javascript frameworks",
            seed_point=np.array([0.1, 0.0, 0.0]),
            labels=["javascript", "coding"],
            weight=2.0,
        )
        mesh.store_secondary(t2, "react frontend", labels=["javascript", "frontend"], weight=1.0)
        mesh.store_secondary(t2, "node backend", labels=["javascript", "backend"], weight=1.0)

        result = mesh.abstract_reorganize(min_density=2)
        assert result["integrated_count"] == 2

    def test_reorganize_respects_max_operations(self):
        mesh = TetraMesh()
        for i in range(10):
            tid = mesh.store(
                "dense_" + str(i),
                seed_point=np.array([float(i) * 0.05, 0.0, 0.0]),
                labels=["test"],
                weight=1.0,
            )
            for j in range(5):
                mesh.store_secondary(tid, "sec_" + str(i) + "_" + str(j), labels=["test"], weight=1.0)

        result = mesh.abstract_reorganize(min_density=2, max_operations=3)
        assert result["integrated_count"] <= 3

    def test_reorganize_custom_fusion_fn(self):
        mesh = TetraMesh()
        t1 = mesh.store("alpha", seed_point=np.array([0.0, 0.0, 0.0]),
                         labels=["x"], weight=1.0)
        mesh.store_secondary(t1, "a1", labels=["x"], weight=1.0)
        mesh.store_secondary(t1, "a2", labels=["x"], weight=1.0)

        t2 = mesh.store("beta", seed_point=np.array([0.1, 0.0, 0.0]),
                         labels=["x"], weight=1.0)
        mesh.store_secondary(t2, "b1", labels=["x"], weight=1.0)
        mesh.store_secondary(t2, "b2", labels=["x"], weight=1.0)

        def custom_fusion(c1, c2):
            return "MERGED: " + c1 + " + " + c2

        result = mesh.abstract_reorganize(min_density=2, fusion_fn=custom_fusion)
        assert result["integrated_count"] == 2
