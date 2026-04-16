import pytest
import threading
import time
import sys
sys.path.insert(0, "/root/.openclaw/workspace/sunorm-space-memory")


@pytest.fixture(autouse=True)
def fresh_api(tmp_path, monkeypatch):
    monkeypatch.setenv("TETRAMEM_STORAGE", str(tmp_path / "test_data"))
    import importlib
    import start_api_v2
    importlib.reload(start_api_v2)
    start_api_v2.init_state()
    yield start_api_v2
    start_api_v2._flush_save()


class TestStoreAndQuery:
    def test_store_returns_id(self, fresh_api):
        req = fresh_api.StoreReq(content="hello world", labels=["greet"])
        resp = fresh_api.store(req)
        assert resp["id"] is not None
        assert len(resp["id"]) > 0

    def test_query_returns_stored(self, fresh_api):
        fresh_api.store(fresh_api.StoreReq(content="alpha beta", labels=["test"]))
        fresh_api.store(fresh_api.StoreReq(content="gamma delta", labels=["test"]))
        resp = fresh_api.query(fresh_api.QueryReq(query="alpha beta", k=5))
        assert len(resp["results"]) >= 1
        assert any("alpha" in r["content"] for r in resp["results"])

    def test_query_with_labels(self, fresh_api):
        fresh_api.store(fresh_api.StoreReq(content="python code", labels=["programming"]))
        fresh_api.store(fresh_api.StoreReq(content="java code", labels=["programming"]))
        fresh_api.store(fresh_api.StoreReq(content="baking bread", labels=["cooking"]))
        resp = fresh_api.query(fresh_api.QueryReq(query="code", k=10, labels=["programming"]))
        assert len(resp["results"]) >= 1

    def test_query_empty_mesh(self, fresh_api):
        resp = fresh_api.query(fresh_api.QueryReq(query="nothing here", k=5))
        assert resp["results"] == []

    def test_store_with_metadata(self, fresh_api):
        req = fresh_api.StoreReq(
            content="test meta", labels=["meta"],
            metadata={"source": "unit-test", "priority": 1},
        )
        resp = fresh_api.store(req)
        assert resp["id"]
        t = fresh_api._mesh.get_tetrahedron(resp["id"])
        assert t is not None
        assert t.metadata.get("source") == "unit-test"

    def test_store_weight_bounds(self, fresh_api):
        with pytest.raises(Exception):
            fresh_api.store(fresh_api.StoreReq(content="bad", weight=0.0))
        with pytest.raises(Exception):
            fresh_api.store(fresh_api.StoreReq(content="bad", weight=11.0))


class TestAssociate:
    def test_associate_returns_neighbors(self, fresh_api):
        t1 = fresh_api.store(fresh_api.StoreReq(content="node A"))
        t2 = fresh_api.store(fresh_api.StoreReq(content="node B"))
        resp = fresh_api.associate(fresh_api.AssociateReq(tetra_id=t1["id"], max_depth=2))
        assert "associations" in resp
        assert isinstance(resp["associations"], list)

    def test_associate_invalid_id(self, fresh_api):
        resp = fresh_api.associate(fresh_api.AssociateReq(tetra_id="nonexistent", max_depth=1))
        assert resp["associations"] == []


class TestDream:
    def test_dream_trigger(self, fresh_api):
        for i in range(5):
            fresh_api.store(fresh_api.StoreReq(content=f"memory item {i}", labels=[f"label{i}"]))
        resp = fresh_api.dream(fresh_api.DreamReq(force=True))
        assert "result" in resp
        assert resp["result"]["phase"] in ("complete", "too_few_tetra", "walk_too_short", "single_cluster", "no_regular_tetra")


class TestSelfOrganize:
    def test_self_organize_runs(self, fresh_api):
        for i in range(5):
            fresh_api.store(fresh_api.StoreReq(content=f"org item {i}", labels=["org"]))
        resp = fresh_api.self_organize(fresh_api.SelfOrgReq(max_iterations=3))
        assert "stats" in resp


class TestAbstractReorganize:
    def test_abstract_reorganize_runs(self, fresh_api):
        for i in range(3):
            fresh_api.store(fresh_api.StoreReq(content=f"reorg item {i}"))
        resp = fresh_api.abstract_reorganize(fresh_api.AbstractReorgReq(min_density=1, max_operations=5))
        assert "result" in resp


class TestNavigate:
    def test_navigate_returns_path(self, fresh_api):
        t1 = fresh_api.store(fresh_api.StoreReq(content="start node"))
        for i in range(5):
            fresh_api.store(fresh_api.StoreReq(content=f"nav node {i}"))
        resp = fresh_api.navigate(fresh_api.NavigateReq(seed_id=t1["id"], max_steps=10, strategy="bfs"))
        assert "path" in resp
        assert isinstance(resp["path"], list)


class TestSeedByLabel:
    def test_seed_by_label_finds_existing(self, fresh_api):
        fresh_api.store(fresh_api.StoreReq(content="python stuff", labels=["python"]))
        resp = fresh_api.seed_by_label(fresh_api.SeedByLabelReq(labels=["python"]))
        assert resp["id"] is not None

    def test_seed_by_label_missing(self, fresh_api):
        resp = fresh_api.seed_by_label(fresh_api.SeedByLabelReq(labels=["nonexistent"]))
        assert resp["id"] is None


class TestStats:
    def test_stats_structure(self, fresh_api):
        fresh_api.store(fresh_api.StoreReq(content="stat item"))
        resp = fresh_api.stats()
        assert isinstance(resp, dict)
        assert "total_tetrahedra" in resp

    def test_health(self, fresh_api):
        resp = fresh_api.health()
        assert resp["status"] == "ok"
        assert resp["version"] == "2.5.0"
        assert resp["uptime_seconds"] >= 0


class TestExport:
    def test_export(self, fresh_api):
        fresh_api.store(fresh_api.StoreReq(content="export me"))
        resp = fresh_api.export()
        assert resp["status"] == "ok"
        assert resp["size"] > 0


class TestClosedLoop:
    def test_closed_loop_runs(self, fresh_api):
        for i in range(5):
            fresh_api.store(fresh_api.StoreReq(content=f"loop item {i}"))
        resp = fresh_api.closed_loop()
        assert "result" in resp
        assert isinstance(resp["result"], dict)


class TestTopologyHealth:
    def test_topology_health(self, fresh_api):
        resp = fresh_api.topology_health()
        assert "result" in resp


class TestThreadSafety:
    def test_concurrent_stores(self, fresh_api):
        n = 50
        results = {"ok": 0, "err": 0}
        lock = threading.Lock()

        def worker(wid):
            for i in range(n):
                try:
                    req = fresh_api.StoreReq(content=f"concurrent_{wid}_{i}", labels=["concurrent"])
                    resp = fresh_api.store(req)
                    if resp.get("id"):
                        with lock:
                            results["ok"] += 1
                except Exception:
                    with lock:
                        results["err"] += 1

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["ok"] == 200
        assert results["err"] == 0

    def test_concurrent_store_and_query(self, fresh_api):
        for i in range(10):
            fresh_api.store(fresh_api.StoreReq(content=f"preload_{i}"))

        errors = []
        store_count = {"n": 0}
        query_count = {"n": 0}

        def store_worker():
            for i in range(30):
                try:
                    req = fresh_api.StoreReq(content=f"sq_store_{i}")
                    fresh_api.store(req)
                    store_count["n"] += 1
                except Exception as e:
                    errors.append(str(e))

        def query_worker():
            for i in range(30):
                try:
                    req = fresh_api.QueryReq(query=f"sq_query_{i}", k=3)
                    fresh_api.query(req)
                    query_count["n"] += 1
                except Exception as e:
                    errors.append(str(e))

        ts = [threading.Thread(target=store_worker) for _ in range(2)]
        tq = [threading.Thread(target=query_worker) for _ in range(2)]
        for t in ts + tq:
            t.start()
        for t in ts + tq:
            t.join()

        assert not errors, f"Errors: {errors[:3]}"
        assert store_count["n"] == 60
        assert query_count["n"] == 60


class TestPersistence:
    def test_save_and_reload(self, fresh_api, tmp_path):
        import json
        fresh_api.store(fresh_api.StoreReq(content="persist me", labels=["persist"]))
        fresh_api._flush_save()

        import importlib
        import start_api_v2
        importlib.reload(start_api_v2)
        start_api_v2.init_state()
        assert len(start_api_v2._mesh.tetrahedra) >= 1
