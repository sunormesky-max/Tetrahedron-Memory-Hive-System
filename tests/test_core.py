"""
Tests for Tetrahedron Memory System core functionality.
"""

import numpy as np
import pytest

from tetrahedron_memory import RayController
from tetrahedron_memory.core import GeoMemoryBody, MemoryNode, QueryResult
from tetrahedron_memory.geometry import GeometryPrimitives, TextToGeometryMapper
from tetrahedron_memory.multimodal import PixHomology
from tetrahedron_memory.partitioning import BoundingBox, M3NOPartitioner, Octree
from tetrahedron_memory.persistence import ParquetPersistence


class TestMemoryNode:
    """Tests for MemoryNode dataclass."""

    def test_memory_node_creation(self):
        """Test creating a MemoryNode."""
        node = MemoryNode(
            id="test_id",
            content="Test content",
            geometry=np.array([1.0, 0.0, 0.0]),
            timestamp=1234567890.0,
            weight=1.5,
            labels=["label1", "label2"],
            metadata={"key": "value"},
        )
        assert node.id == "test_id"
        assert node.content == "Test content"
        assert node.weight == 1.5
        assert len(node.labels) == 2


class TestGeoMemoryBody:
    """Tests for GeoMemoryBody class."""

    @pytest.fixture
    def memory_body(self):
        """Create a fresh memory body for each test."""
        return GeoMemoryBody(dimension=3, precision="fast")

    def test_initialization(self, memory_body):
        """Test memory body initialization."""
        assert memory_body.dimension == 3
        assert memory_body.precision == "fast"
        assert len(memory_body._nodes) == 0

    def test_store_memory(self, memory_body):
        """Test storing a memory."""
        memory_id = memory_body.store(
            content="Test memory content",
            labels=["test", "example"],
            metadata={"source": "test"},
            weight=2.0,
        )
        assert memory_id is not None
        assert len(memory_body._nodes) == 1
        assert memory_id in memory_body._nodes

    def test_store_multiple_memories(self, memory_body):
        """Test storing multiple memories."""
        ids = []
        for i in range(5):
            memory_id = memory_body.store(content=f"Memory content {i}", labels=[f"label_{i}"])
            ids.append(memory_id)

        assert len(memory_body._nodes) == 5
        assert len(set(ids)) == 5

    def test_query_empty_memory(self, memory_body):
        """Test querying empty memory body."""
        results = memory_body.query("test query", k=5)
        assert results == []

    def test_query_with_memories(self, memory_body):
        """Test querying memories."""
        memory_body.store("First memory")
        memory_body.store("Second memory")
        memory_body.store("Third memory")

        results = memory_body.query("memory", k=2)
        assert len(results) <= 2
        assert all(isinstance(r, QueryResult) for r in results)

    def test_query_by_label(self, memory_body):
        """Test querying by label."""
        memory_body.store("Memory A", labels=["category_a"])
        memory_body.store("Memory B", labels=["category_b"])
        memory_body.store("Memory C", labels=["category_a"])

        results = memory_body.query_by_label("category_a")
        assert len(results) == 2

    def test_associate_memories(self, memory_body):
        """Test finding associated memories."""
        id1 = memory_body.store("First memory", weight=1.0)
        memory_body.store("Second memory", weight=1.0)
        memory_body.store("Third memory", weight=1.0)

        associations = memory_body.associate(id1, max_depth=2)
        assert isinstance(associations, list)

    def test_update_weight_ema(self, memory_body):
        """Test weight update with EMA."""
        memory_id = memory_body.store("Test memory", weight=1.0)
        initial_weight = memory_body._nodes[memory_id].weight

        memory_body.update_weight(memory_id, delta=2.0, use_ema=True, alpha=0.5)
        new_weight = memory_body._nodes[memory_id].weight

        assert new_weight != initial_weight
        assert 0.1 <= new_weight <= 10.0

    def test_update_weight_direct(self, memory_body):
        """Test direct weight update."""
        memory_id = memory_body.store("Test memory", weight=1.0)

        memory_body.update_weight(memory_id, delta=0.5, use_ema=False)
        new_weight = memory_body._nodes[memory_id].weight

        assert new_weight == pytest.approx(1.5, rel=1e-5)

    def test_detect_conflicts(self, memory_body):
        """Test conflict detection."""
        memory_body.store("Memory A")
        memory_body.store("Memory B")

        conflicts = memory_body.detect_conflicts()
        assert isinstance(conflicts, list)

    def test_get_statistics(self, memory_body):
        """Test getting statistics."""
        memory_body.store("Memory 1", weight=1.5)
        memory_body.store("Memory 2", weight=2.5)

        stats = memory_body.get_statistics()
        assert stats["total_memories"] == 2
        assert stats["dimension"] == 3
        assert stats["avg_weight"] == pytest.approx(2.0, rel=1e-5)

    def test_clear_memory(self, memory_body):
        """Test clearing all memories."""
        memory_body.store("Memory 1")
        memory_body.store("Memory 2")
        assert len(memory_body._nodes) == 2

        memory_body.clear()
        assert len(memory_body._nodes) == 0

    def test_deterministic_geometry(self, memory_body):
        """Test that same text produces same geometry."""
        geom1 = memory_body._text_to_geometry("test content")
        geom2 = memory_body._text_to_geometry("test content")

        np.testing.assert_array_almost_equal(geom1, geom2)

    def test_different_text_different_geometry(self, memory_body):
        """Test that different text produces different geometry."""
        geom1 = memory_body._text_to_geometry("content A")
        geom2 = memory_body._text_to_geometry("content B")

        distance = np.linalg.norm(geom1 - geom2)
        assert distance > 0.01


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self):
        """Test creating a QueryResult."""
        node = MemoryNode(id="test", content="Test", geometry=np.array([1, 0, 0]), timestamp=0.0)
        result = QueryResult(
            node=node, distance=0.5, persistence_score=0.8, association_type="geometric"
        )

        assert result.distance == 0.5
        assert result.persistence_score == 0.8
        assert result.association_type == "geometric"


class TestGeometryPrimitives:
    """Tests for GeometryPrimitives class."""

    def test_tetrahedron_volume(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        volume = GeometryPrimitives.tetrahedron_volume(vertices)
        assert volume == pytest.approx(1.0 / 6.0, rel=1e-5)

    def test_shared_vertices(self):
        tet1 = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        tet2 = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        shared = GeometryPrimitives.shared_vertices(tet1, tet2)
        assert shared == 3

    def test_jaccard_index(self):
        tet1 = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        tet2 = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        jaccard = GeometryPrimitives.jaccard_index(tet1, tet2)
        assert jaccard == pytest.approx(1.0, rel=1e-5)

    def test_centroid(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        centroid = GeometryPrimitives.centroid(vertices)
        expected = np.array([0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(centroid, expected)


class TestTextToGeometryMapper:
    """Tests for TextToGeometryMapper class."""

    def test_deterministic_mapping(self):
        mapper = TextToGeometryMapper()
        geom1 = mapper.map_text("test")
        geom2 = mapper.map_text("test")
        np.testing.assert_array_almost_equal(geom1, geom2)

    def test_different_text_different_geometry(self):
        mapper = TextToGeometryMapper()
        geom1 = mapper.map_text("text A")
        geom2 = mapper.map_text("text B")
        assert not np.allclose(geom1, geom2)


class TestOctree:
    """Tests for Octree class."""

    def test_insert_and_query(self):
        bounds = BoundingBox(
            np.array([-1.0, -1.0, -1.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        tree = Octree(bounds)
        tree.insert(np.array([0.0, 0.0, 0.0]), "node1")
        tree.insert(np.array([0.5, 0.5, 0.5]), "node2")

        results = tree.query_nearest(np.array([0.0, 0.0, 0.0]), k=2)
        assert len(results) == 2


class TestM3NOPartitioner:
    """Tests for M3NOPartitioner class."""

    def test_initialization(self):
        partitioner = M3NOPartitioner()
        assert not partitioner.is_initialized()

    def test_add_point(self):
        partitioner = M3NOPartitioner()
        partitioner.add_point(np.array([0.0, 0.0, 0.0]), "node1")
        assert partitioner.is_initialized()


class TestParquetPersistence:
    """Tests for ParquetPersistence class."""

    def test_save_and_load(self):
        persistence = ParquetPersistence()
        persistence.save_snapshot(
            node_id="test",
            content="test content",
            geometry=np.array([1.0, 0.0, 0.0]),
            timestamp=1234567890.0,
            weight=1.0,
            labels=["test"],
            metadata={},
        )
        snapshots = persistence.load_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].id == "test"


class TestRayController:
    """Tests for RayController class."""

    def test_initialization(self):
        controller = RayController()
        assert not controller.is_initialized()

    def test_initialize(self):
        controller = RayController()
        controller.initialize()
        assert controller.is_initialized()
        assert controller.is_local_mode()


class TestPixHomology:
    """Tests for PixHomology class."""

    def test_image_to_geometry(self):
        pix_homology = PixHomology()
        image = np.random.rand(32, 32)
        geometry = pix_homology.image_to_geometry(image)
        assert geometry.shape == (3,)
        assert np.linalg.norm(geometry) == pytest.approx(1.0, rel=1e-5)
