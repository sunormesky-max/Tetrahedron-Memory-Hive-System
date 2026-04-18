"""PCNN Honeycomb Neural Field tests for TetraMem-XL v4.1"""
import pytest
import time
import threading


def _create_field(resolution=3, spacing=1.0):
    from tetrahedron_memory.honeycomb_neural_field import HoneycombNeuralField
    field = HoneycombNeuralField(resolution=resolution, spacing=spacing)
    field.initialize()
    return field


class TestBCClattice:
    def test_lattice_builds(self):
        field = _create_field(resolution=2)
        stats = field.stats()
        assert stats["total_nodes"] > 0
        assert stats["face_edges"] > 0

    def test_node_positions_unique(self):
        field = _create_field(resolution=2)
        positions = set()
        for nid, node in field._nodes.items():
            pos_tuple = tuple(node.position.tolist())
            positions.add(pos_tuple)
        assert len(positions) == len(field._nodes)


class TestMemoryStore:
    def test_store_and_query(self):
        field = _create_field(resolution=2)
        mid = field.store("Hello PCNN", labels=["test"], weight=2.0)
        assert mid is not None
        results = field.query("Hello", k=5)
        assert len(results) > 0
        assert any("PCNN" in r["content"] for r in results)

    def test_store_duplicate_reinforces(self):
        field = _create_field(resolution=2)
        mid1 = field.store("duplicate content", weight=1.0)
        mid2 = field.store("duplicate content", weight=1.0)
        assert mid1 == mid2

    def test_weight_preserved(self):
        field = _create_field(resolution=2)
        mid = field.store("weight test", weight=5.0)
        node = field.get_node(mid)
        assert node is not None
        assert node["weight"] == 5.0


class TestPCNNEngine:
    def test_pulse_emission(self):
        field = _create_field(resolution=2)
        field.store("pulse source", weight=2.0)
        from tetrahedron_memory.honeycomb_neural_field import PulseType
        field._emit_pulse(
            list(field._nodes.keys())[0],
            strength=0.5,
            pulse_type=PulseType.EXPLORATORY,
        )
        assert field._pulse_count >= 1

    def test_hebbian_recording(self):
        field = _create_field(resolution=2)
        field.store("node A", labels=["test"], weight=2.0)
        field.store("node B", labels=["test"], weight=2.0)
        path = list(field._nodes.keys())[:2]
        field._hebbian.record_path(path, success=True, strength=0.5)
        stats = field._hebbian.stats()
        assert stats["success_count"] >= 1

    def test_self_check_engine(self):
        field = _create_field(resolution=2)
        field.store("check me", weight=1.0)
        from tetrahedron_memory.honeycomb_neural_field import SelfCheckEngine
        engine = SelfCheckEngine(field)
        result = engine.run_full_check()
        assert result is not None
        d = result.to_dict()
        assert "anomalies_found" in d
        assert "pulse_triggered" in d


class TestDuplicateDetection:
    def test_finds_duplicates(self):
        field = _create_field(resolution=3)
        field.store("The quick brown fox jumps over the lazy dog", weight=1.0)
        field.store("The quick brown fox jumped over the lazy dogs", weight=1.0)
        dupes = field.detect_duplicates()
        assert len(dupes) > 0
        assert dupes[0]["similarity"] >= 0.7

    def test_no_false_positives(self):
        field = _create_field(resolution=3)
        field.store("Completely different content about quantum physics", weight=1.0)
        field.store("Today is a sunny day for walking in the park", weight=1.0)
        dupes = field.detect_duplicates()
        assert len(dupes) == 0


class TestIsolatedDetection:
    def test_detects_isolated(self):
        field = _create_field(resolution=2)
        mid = field.store("isolated memory", weight=1.0)
        isolated = field.detect_isolated()
        assert isinstance(isolated, list)

    def test_connected_not_isolated(self):
        field = _create_field(resolution=3)
        for i in range(10):
            field.store(f"cluster memory {i}", labels=["cluster"], weight=1.0)
        isolated = field.detect_isolated()
        assert len(isolated) == 0


class TestTimeline:
    def test_timeline_newest(self):
        field = _create_field(resolution=2)
        field.store("old", weight=1.0)
        time.sleep(0.01)
        field.store("new", weight=1.0)
        items = field.browse_timeline(direction="newest", limit=5)
        assert len(items) >= 2
        assert items[0]["content"] == "new"

    def test_timeline_label_filter(self):
        field = _create_field(resolution=2)
        field.store("filtered", labels=["special"], weight=1.0)
        field.store("normal", labels=["normal"], weight=1.0)
        items = field.browse_timeline(labels=["special"])
        assert all("special" in i["labels"] for i in items)


class TestStats:
    def test_stats_structure(self):
        field = _create_field(resolution=2)
        stats = field.stats()
        assert "total_nodes" in stats
        assert "occupied_nodes" in stats
        assert "face_edges" in stats
        assert "pcnn_config" in stats
        assert "self_check" in stats
