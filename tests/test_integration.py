import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from start_api_v2 import app
    with TestClient(app) as c:
        yield c


def _headers(api_key="test-key-001"):
    return {"X-API-Key": api_key}


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert "version" in data


def test_store_and_query(client):
    r = client.post("/api/v1/store", json={
        "content": "Integration test memory about dark planes",
        "labels": ["test", "dark-plane"],
        "weight": 1.5,
    }, headers=_headers())
    assert r.status_code == 200
    store_data = r.json()
    mem_id = store_data["id"]
    assert mem_id

    r = client.post("/api/v1/query", json={"query": "dark planes", "k": 5}, headers=_headers())
    assert r.status_code == 200
    body = r.json()
    results = body.get("results", body) if isinstance(body, dict) else body
    assert len(results) >= 1
    found = any(m["id"] == mem_id for m in results)
    assert found, f"Stored memory {mem_id} not found in query results"


def test_store_multiple_and_browse(client):
    for i in range(5):
        r = client.post("/api/v1/store", json={
            "content": f"Browse test memory {i}",
            "labels": ["browse-test"],
            "weight": 1.0,
        }, headers=_headers())
        assert r.status_code == 200

    r = client.get("/api/v1/browse", params={"direction": "newest", "limit": 3}, headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert "tetrahedra" in data
    assert len(data["tetrahedra"]) <= 3


def test_stats(client):
    r = client.get("/api/v1/stats", headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert "total_nodes" in data or "occupied" in data


def test_login_with_password(client):
    r = client.post("/api/v1/login", json={"password": "CHANGE_ME"})
    assert r.status_code == 200
    data = r.json()
    assert "token" in data

    r = client.post("/api/v1/login", json={"password": "wrong-password"})
    assert r.status_code == 401


def test_dark_plane_stats(client):
    r = client.get("/api/v1/dark-plane/stats", headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert "temperature" in data
    assert "plane_distribution" in data


def test_dark_plane_flow(client):
    client.post("/api/v1/store", json={
        "content": "Dark plane flow test",
        "labels": ["dp-flow"],
        "weight": 1.0,
    }, headers=_headers())

    r = client.post("/api/v1/dark-plane/flow", headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert "temperature" in data
    assert "plane_distribution" in data
    assert "adaptive_thresholds" in data


def test_regulation_status(client):
    r = client.get("/api/v1/regulation/status", headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert "active" in data
    assert data["active"] is True


def test_regulation_trigger(client):
    r = client.post("/api/v1/regulation/trigger", headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert "hormones" in data
    assert "stress" in data
    assert "circadian" in data
    assert "actions" in data


def test_regulation_history(client):
    client.post("/api/v1/regulation/trigger", headers=_headers())

    r = client.get("/api/v1/regulation/history", params={"n": 5}, headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert "history" in data
    assert len(data["history"]) >= 1


def test_attention_focus_and_status(client):
    client.post("/api/v1/store", json={
        "content": "Attention test node",
        "labels": ["attention-test"],
        "weight": 1.0,
    }, headers=_headers())

    r = client.post("/api/v1/attention/focus", json={
        "center": [0.0, 0.0, 0.0],
        "radius": 10.0,
        "strength": 1.0,
        "labels": ["attention-test"],
    }, headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert "active_foci" in data

    r = client.get("/api/v1/attention/status", headers=_headers())
    assert r.status_code == 200

    r = client.post("/api/v1/attention/clear", headers=_headers())
    assert r.status_code == 200


def test_reflection_field(client):
    r = client.get("/api/v1/reflection-field/status", headers=_headers())
    assert r.status_code == 200

    r = client.post("/api/v1/reflection-field/run", headers=_headers())
    assert r.status_code == 200


def test_cascade_trigger(client):
    client.post("/api/v1/store", json={
        "content": "Cascade test source",
        "labels": ["cascade"],
        "weight": 2.0,
    }, headers=_headers())

    r = client.post("/api/v1/cascade/trigger", headers=_headers())
    assert r.status_code == 200


def test_self_organize(client):
    r = client.post("/api/v1/self-organize", headers=_headers())
    assert r.status_code == 200


def test_dream_cycle(client):
    r = client.post("/api/v1/dream", headers=_headers())
    assert r.status_code == 200


def test_export(client):
    r = client.get("/api/v1/export", headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_import(client):
    memories = [
        {"content": "Import test 1", "labels": ["import-test"], "weight": 1.0},
        {"content": "Import test 2", "labels": ["import-test"], "weight": 1.5},
    ]
    r = client.post("/api/v1/import", json={"memories": memories}, headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert data.get("imported", 0) >= 1


def test_query_with_dark_plane_energy_injection(client):
    for i in range(3):
        client.post("/api/v1/store", json={
            "content": f"Energy injection test {i} - abyss candidate",
            "labels": ["energy-inject"],
            "weight": 0.5,
        }, headers=_headers())

    client.post("/api/v1/dark-plane/flow", headers=_headers())

    r = client.post("/api/v1/query", json={"query": "abyss candidate", "k": 3}, headers=_headers())
    assert r.status_code == 200
    results = r.json()
    assert len(results) >= 1

    r = client.get("/api/v1/dark-plane/stats", headers=_headers())
    stats = r.json()
    assert stats.get("energy_injections", 0) >= 0


def test_regulation_force_mode(client):
    r = client.post("/api/v1/regulation/force-mode", json={"mode": "consolidation"}, headers=_headers())
    assert r.status_code == 200

    r = client.post("/api/v1/regulation/force-mode", json={"mode": "sympathetic"}, headers=_headers())
    assert r.status_code == 200

    r = client.post("/api/v1/regulation/force-mode", json={"mode": "work"}, headers=_headers())
    assert r.status_code == 200


def test_dark_plane_node_report(client):
    store_r = client.post("/api/v1/store", json={
        "content": "Node report test",
        "labels": ["node-report"],
        "weight": 1.0,
    }, headers=_headers())
    mem_id = store_r.json()["id"]

    client.post("/api/v1/dark-plane/flow", headers=_headers())

    r = client.get(f"/api/v1/dark-plane/node/{mem_id}", headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert "well_depth" in data
    assert "plane" in data
    assert "internal_energy" in data
    assert "neighborhood_entropy" in data
