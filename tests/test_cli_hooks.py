"""
Tests for CLI and hooks modules.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_store_and_query(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            code = main(["store", "test memory content", "-l", "test", "-w", "1.5"])
            self.assertEqual(code, 0)

    def test_stats(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            code = main(["stats"])
            self.assertEqual(code, 0)

    def test_clear(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            code = main(["clear"])
            self.assertEqual(code, 0)

    def test_persist(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            code = main(["persist"])
            self.assertEqual(code, 0)

    def test_status(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            code = main(["status"])
            self.assertEqual(code, 0)

    def test_dream(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            from tetrahedron_memory.core import GeoMemoryBody
            body = GeoMemoryBody(dimension=3, precision="fast")
            for i in range(5):
                body.store(f"mem_{i}", labels=[f"l{i}"])
            from tetrahedron_memory.persistence import MemoryPersistence
            persistence = MemoryPersistence(storage_dir=self.tmpdir)
            persistence.save_nodes(body._nodes)

            code = main(["dream", "-n", "1"])
            self.assertEqual(code, 0)

    def test_self_org(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            code = main(["self-org"])
            self.assertEqual(code, 0)

    def test_catalyze(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            code = main(["catalyze"])
            self.assertEqual(code, 0)

    def test_mquery(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            from tetrahedron_memory.core import GeoMemoryBody
            body = GeoMemoryBody(dimension=3, precision="fast")
            for i in range(5):
                body.store(f"mem_{i}", labels=[f"l{i}"])
            from tetrahedron_memory.persistence import MemoryPersistence
            persistence = MemoryPersistence(storage_dir=self.tmpdir)
            persistence.save_nodes(body._nodes)

            code = main(["mquery", "mem", "-k", "3"])
            self.assertEqual(code, 0)

    def test_build_pyramid(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            from tetrahedron_memory.core import GeoMemoryBody
            body = GeoMemoryBody(dimension=3, precision="fast")
            for i in range(10):
                body.store(f"mem_{i}")
            from tetrahedron_memory.persistence import MemoryPersistence
            persistence = MemoryPersistence(storage_dir=self.tmpdir)
            persistence.save_nodes(body._nodes)

            code = main(["build-pyramid"])
            self.assertEqual(code, 0)

    def test_pyquery(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            from tetrahedron_memory.core import GeoMemoryBody
            body = GeoMemoryBody(dimension=3, precision="fast")
            for i in range(10):
                body.store(f"mem_{i}")
            from tetrahedron_memory.persistence import MemoryPersistence
            persistence = MemoryPersistence(storage_dir=self.tmpdir)
            persistence.save_nodes(body._nodes)

            code = main(["pyquery", "mem", "-k", "3"])
            self.assertEqual(code, 0)

    def test_zigzag(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            from tetrahedron_memory.core import GeoMemoryBody
            body = GeoMemoryBody(dimension=3, precision="fast")
            for i in range(5):
                body.store(f"mem_{i}")
            from tetrahedron_memory.persistence import MemoryPersistence
            persistence = MemoryPersistence(storage_dir=self.tmpdir)
            persistence.save_nodes(body._nodes)

            code = main(["zigzag"])
            self.assertEqual(code, 0)

    def test_predict(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            code = main(["predict"])
            self.assertEqual(code, 0)

    def test_no_command(self):
        from tetrahedron_memory.cli import main
        code = main([])
        self.assertEqual(code, 1)

    def test_label_query(self):
        from tetrahedron_memory.cli import main
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": self.tmpdir}):
            from tetrahedron_memory.core import GeoMemoryBody
            body = GeoMemoryBody(dimension=3, precision="fast")
            body.store("labeled memory", labels=["test_label"])
            from tetrahedron_memory.persistence import MemoryPersistence
            persistence = MemoryPersistence(storage_dir=self.tmpdir)
            persistence.save_nodes(body._nodes)

            code = main(["label", "test_label"])
            self.assertEqual(code, 0)


class TestHooks(unittest.TestCase):
    def test_get_memory_creates_instance(self):
        import importlib
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": tempfile.mkdtemp()}):
            import tetrahedron_memory.hooks as hooks
            importlib.reload(hooks)
            hooks._memory = None
            mem = hooks.get_memory()
            self.assertIsNotNone(mem)
            self.assertEqual(len(mem._nodes), 0)
            hooks._memory = None

    def test_get_memory_loads_persisted(self):
        import importlib
        tmpdir = tempfile.mkdtemp()
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": tmpdir}):
            from tetrahedron_memory.core import GeoMemoryBody
            body = GeoMemoryBody(dimension=3, precision="fast")
            body.store("persisted memory", labels=["test"], weight=2.0)
            from tetrahedron_memory.persistence import MemoryPersistence
            persistence = MemoryPersistence(storage_dir=tmpdir)
            persistence.save_nodes(body._nodes)

            import tetrahedron_memory.hooks as hooks
            importlib.reload(hooks)
            hooks._memory = None
            mem = hooks.get_memory()
            self.assertEqual(len(mem._nodes), 1)
            hooks._memory = None

    def test_get_memory_idempotent(self):
        import importlib
        with patch.dict(os.environ, {"TETRAMEM_STORAGE": tempfile.mkdtemp()}):
            import tetrahedron_memory.hooks as hooks
            importlib.reload(hooks)
            hooks._memory = None
            m1 = hooks.get_memory()
            m2 = hooks.get_memory()
            self.assertIs(m1, m2)
            hooks._memory = None

    def test_storage_dir_default(self):
        default = os.path.expanduser("~/.tetramem_data")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TETRAMEM_STORAGE", None)
            import tetrahedron_memory.hooks as hooks
            import importlib
            importlib.reload(hooks)
            self.assertEqual(hooks.STORAGE_DIR, default)
            hooks._memory = None


if __name__ == "__main__":
    unittest.main()
