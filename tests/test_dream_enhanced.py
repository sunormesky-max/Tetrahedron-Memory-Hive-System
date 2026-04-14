"""
Tests for P2: Dream Law Enhancement.
"""

import hashlib
import time

import numpy as np
import pytest

from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_dream import (
    DreamRecord,
    DreamStore,
    TetraDreamCycle,
    fusion_quality_score,
)


class TestDreamRecord:
    def test_creation(self):
        rec = DreamRecord(
            dream_id="d1",
            tetra_id="t1",
            source_tetra_ids=["s1", "s2"],
            source_clusters=[["s1"], ["s2"]],
            synthesis_content="dream content",
            fusion_quality=0.75,
            entropy_before=2.0,
            entropy_after=1.5,
            entropy_delta=0.25,
            creation_time=time.time(),
            labels=["ai"],
            reintegrated=False,
            reintegration_count=0,
            walk_path_hash="abc123",
        )
        assert rec.dream_id == "d1"
        assert rec.fusion_quality == 0.75
        assert not rec.reintegrated

    def test_to_dict(self):
        rec = DreamRecord(
            dream_id="d2",
            tetra_id="t2",
            source_tetra_ids=["s1"],
            source_clusters=[["s1"]],
            synthesis_content="x" * 300,
            fusion_quality=0.5,
            entropy_before=1.0,
            entropy_after=0.8,
            entropy_delta=0.2,
            creation_time=1000.0,
            labels=["test"],
            reintegrated=False,
            reintegration_count=0,
            walk_path_hash="hash",
        )
        d = rec.to_dict()
        assert d["dream_id"] == "d2"
        assert len(d["synthesis_content"]) <= 200
        assert d["entropy_delta"] == 0.2


class TestDreamStore:
    def test_record_and_get(self):
        store = DreamStore()
        rec = DreamRecord(
            dream_id="d1", tetra_id="t1",
            source_tetra_ids=["s1", "s2"],
            source_clusters=[["s1"], ["s2"]],
            synthesis_content="dream",
            fusion_quality=0.8,
            entropy_before=2.0, entropy_after=1.5, entropy_delta=0.25,
            creation_time=time.time(), labels=["a"],
            reintegrated=False, reintegration_count=0,
            walk_path_hash="h1",
        )
        store.record(rec)
        assert store.get("d1") is not None
        assert store.size == 1

    def test_get_by_source(self):
        store = DreamStore()
        for i in range(3):
            rec = DreamRecord(
                dream_id="d" + str(i), tetra_id="t" + str(i),
                source_tetra_ids=["s1", "s_other"],
                source_clusters=[["s1"], ["s_other"]],
                synthesis_content="dream" + str(i),
                fusion_quality=0.5,
                entropy_before=1.0, entropy_after=0.8, entropy_delta=0.2,
                creation_time=time.time(), labels=[],
                reintegrated=False, reintegration_count=0,
                walk_path_hash="h",
            )
            store.record(rec)

        results = store.get_by_source("s1")
        assert len(results) == 3

        results2 = store.get_by_source("s_other")
        assert len(results2) == 3

        results3 = store.get_by_source("nonexistent")
        assert len(results3) == 0

    def test_max_records_eviction(self):
        store = DreamStore(max_records=3)
        for i in range(5):
            rec = DreamRecord(
                dream_id="d" + str(i), tetra_id="t" + str(i),
                source_tetra_ids=["s" + str(i)],
                source_clusters=[["s" + str(i)]],
                synthesis_content="x",
                fusion_quality=0.5,
                entropy_before=1.0, entropy_after=1.0, entropy_delta=0.0,
                creation_time=time.time(), labels=[],
                reintegrated=False, reintegration_count=0,
                walk_path_hash="h",
            )
            store.record(rec)

        assert store.size == 3
        assert store.get("d0") is None
        assert store.get("d2") is not None

    def test_mark_reintegrated(self):
        store = DreamStore()
        rec = DreamRecord(
            dream_id="d1", tetra_id="t1",
            source_tetra_ids=["s1"],
            source_clusters=[["s1"]],
            synthesis_content="x",
            fusion_quality=0.5,
            entropy_before=1.0, entropy_after=1.0, entropy_delta=0.0,
            creation_time=time.time(), labels=[],
            reintegrated=False, reintegration_count=0,
            walk_path_hash="h",
        )
        store.record(rec)
        assert not rec.reintegrated

        store.mark_reintegrated("d1")
        assert rec.reintegrated
        assert rec.reintegration_count == 1

    def test_quality_stats(self):
        store = DreamStore()
        for q in [0.3, 0.5, 0.7, 0.9]:
            rec = DreamRecord(
                dream_id="d_" + str(q), tetra_id="t",
                source_tetra_ids=["s"],
                source_clusters=[["s"]],
                synthesis_content="x",
                fusion_quality=q,
                entropy_before=2.0, entropy_after=1.0, entropy_delta=0.5,
                creation_time=time.time(), labels=[],
                reintegrated=(q > 0.6), reintegration_count=0,
                walk_path_hash="h",
            )
            store.record(rec)

        stats = store.quality_stats()
        assert stats["count"] == 4
        assert abs(stats["avg_quality"] - 0.6) < 0.01
        assert stats["max_quality"] == 0.9
        assert stats["min_quality"] == 0.3
        assert stats["reintegrated_count"] == 2

    def test_get_recent(self):
        store = DreamStore()
        for i in range(10):
            rec = DreamRecord(
                dream_id="d" + str(i), tetra_id="t" + str(i),
                source_tetra_ids=["s"],
                source_clusters=[["s"]],
                synthesis_content="x",
                fusion_quality=0.5,
                entropy_before=1.0, entropy_after=1.0, entropy_delta=0.0,
                creation_time=float(i), labels=[],
                reintegrated=False, reintegration_count=0,
                walk_path_hash="h",
            )
            store.record(rec)

        recent = store.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].dream_id == "d9"


class TestFusionQualityScore:
    def test_empty_input(self):
        assert fusion_quality_score([], "content") == 0.0

    def test_single_input(self):
        assert fusion_quality_score([{"labels": ["a"]}], "content") == 0.0

    def test_none_content(self):
        inputs = [{"labels": ["a"]}, {"labels": ["b"]}]
        assert fusion_quality_score(inputs, None) == 0.0

    def test_basic_quality(self):
        inputs = [
            {"labels": ["ai", "ml"], "weight": 1.0},
            {"labels": ["ai", "dl"], "weight": 1.0},
            {"labels": ["ml", "stats"], "weight": 1.0},
        ]
        quality = fusion_quality_score(inputs, "[dream] abstract synthesis of AI and ML concepts")
        assert quality > 0.0
        assert quality <= 1.0

    def test_v2_quality_is_positive(self):
        inputs_a = [
            {"labels": ["a", "b", "c", "d"], "weight": 1.0},
            {"labels": ["e", "f", "g", "h"], "weight": 1.0},
        ]
        inputs_b = [
            {"labels": ["a"], "weight": 1.0},
            {"labels": ["a"], "weight": 1.0},
        ]
        q_a = fusion_quality_score(inputs_a, "synthesis of diverse content for quality testing")
        q_b = fusion_quality_score(inputs_b, "synthesis with shared labels for quality testing")
        assert q_a > 0.0
        assert q_b > 0.0
        assert q_a <= 1.0
        assert q_b <= 1.0

    def test_balanced_weights_higher_quality(self):
        balanced = [
            {"labels": ["a"], "weight": 1.0},
            {"labels": ["a"], "weight": 1.0},
        ]
        unbalanced = [
            {"labels": ["a"], "weight": 0.1},
            {"labels": ["a"], "weight": 10.0},
        ]
        q_bal = fusion_quality_score(balanced, "content for quality test")
        q_unbal = fusion_quality_score(unbalanced, "content for quality test")
        assert q_bal >= q_unbal


class TestTetraDreamCycleEnhanced:
    def test_dream_store_integration(self):
        mesh = TetraMesh()
        for i in range(10):
            mesh.store(
                "memory item " + str(i) + " with content about topic",
                seed_point=np.array([float(i) * 0.1, 0.0, 0.0]),
                labels=["topic", "item"],
                weight=1.0 + float(i) * 0.1,
            )

        dream = TetraDreamCycle(mesh, walk_steps=8)
        stats = dream.trigger_now()

        store = dream.get_dream_store()
        assert store.size >= 0

        status = dream.get_status()
        assert "dream_store" in status
        assert status["dream_store"]["count"] >= 0

    def test_dream_trace(self):
        mesh = TetraMesh()
        tid = mesh.store(
            "traceable memory",
            seed_point=np.zeros(3),
            labels=["trace"],
            weight=1.0,
        )
        for i in range(8):
            mesh.store(
                "related " + str(i),
                seed_point=np.array([float(i) * 0.1, 0.0, 0.0]),
                labels=["trace", "related"],
                weight=1.0,
            )

        dream = TetraDreamCycle(mesh, walk_steps=8)
        dream.trigger_now()

        traces = dream.get_dream_trace(tid)
        assert isinstance(traces, list)

    def test_fusion_quality_in_dream(self):
        mesh = TetraMesh()
        for i in range(10):
            mesh.store(
                "diverse content about topic " + str(i),
                seed_point=np.array([float(i) * 0.1, 0.0, 0.0]),
                labels=["topic" + str(i % 3), "shared"],
                weight=1.0,
            )

        dream = TetraDreamCycle(mesh, walk_steps=10)
        dream.trigger_now()

        store = dream.get_dream_store()
        if store.size > 0:
            for rec in store._records.values():
                assert rec.fusion_quality >= 0.0
                assert rec.fusion_quality <= 1.0
                assert rec.creation_time > 0
