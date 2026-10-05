"""REST API behaviour.

Skipped entirely when the optional `server` extra is not installed.
"""

import json

import pytest

fastapi = pytest.importorskip("fastapi", reason="requires the 'server' extra")

from fastapi.testclient import TestClient

from beamline_playground.server import API_PREFIX, ServerConfig, create_app

SCENE = {
    "name": "api test",
    "objs": [
        {"type": "source", "geometry": {"type": "segment", "pos_a": {"x": 0, "y": -1}, "pos_b": {"x": 0, "y": 1}}},
        {"type": "detector", "geometry": {"type": "segment", "pos_a": {"x": -1, "y": 3}, "pos_b": {"x": 1, "y": 3}}},
    ],
    "dag": [[], [0]],
    "wavelength": 5,
    "samples_per_wavelength": 10,
}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


class TestHealth:
    def test_reports_ok(self, client):
        response = client.get(f"{API_PREFIX}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_reports_a_version(self, client):
        assert client.get(f"{API_PREFIX}/health").json()["version"]


class TestSimulate:
    def test_returns_a_result_per_object(self, client):
        response = client.post(f"{API_PREFIX}/simulate", json=SCENE)
        assert response.status_code == 200
        assert len(response.json()["objs"]) == 2

    def test_result_carries_split_field_components(self, client):
        obj = client.post(f"{API_PREFIX}/simulate", json=SCENE).json()["objs"][0]
        assert {"field_re", "field_im", "intensity_y", "sample_x"} <= set(obj)
        assert all(isinstance(v, (int, float)) for v in obj["field_re"])

    def test_malformed_payload_is_rejected(self, client):
        assert client.post(f"{API_PREFIX}/simulate", json={"nope": 1}).status_code == 422

    def test_invalid_dag_length_is_a_422(self, client):
        bad = dict(SCENE, dag=[[], [0], [1]])
        assert client.post(f"{API_PREFIX}/simulate", json=bad).status_code == 422

    def test_cycle_is_a_422(self, client):
        bad = dict(SCENE, dag=[[1], [0]])
        assert client.post(f"{API_PREFIX}/simulate", json=bad).status_code == 422

    def test_unknown_geometry_is_a_422(self, client):
        bad = json.loads(json.dumps(SCENE))
        bad["objs"][0]["geometry"]["type"] = "triangle"
        assert client.post(f"{API_PREFIX}/simulate", json=bad).status_code == 422


class TestSceneCost:
    def test_reports_cost_and_limits(self, client):
        body = client.post(f"{API_PREFIX}/scene/cost", json=SCENE).json()
        assert body["total_elements"] > 0
        assert body["propagation_pairs"] > 0
        assert body["within_limits"] is True

    def test_does_not_run_the_simulation(self, client):
        """Cost must be cheap: it reports on a scene far too big to simulate."""
        huge = dict(SCENE, samples_per_wavelength=100000)
        body = client.post(f"{API_PREFIX}/scene/cost", json=huge).json()
        assert body["within_limits"] is False


class TestCostGuard:
    @pytest.fixture(scope="class")
    def strict(self):
        return TestClient(create_app(ServerConfig(cors_origins=(), max_propagation_pairs=5, max_total_elements=5)))

    def test_rejects_an_over_budget_scene(self, strict):
        response = strict.post(f"{API_PREFIX}/simulate", json=SCENE)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "budget" in detail["message"]
        assert detail["cost"]["within_limits"] is False

    def test_reports_the_limit_it_applied(self, strict):
        detail = strict.post(f"{API_PREFIX}/simulate", json=SCENE).json()["detail"]
        assert detail["cost"]["max_propagation_pairs"] == 5

    def test_generous_limits_allow_the_scene(self):
        relaxed = TestClient(
            create_app(ServerConfig(cors_origins=(), max_propagation_pairs=10**12, max_total_elements=10**9))
        )
        assert relaxed.post(f"{API_PREFIX}/simulate", json=SCENE).status_code == 200


class TestCors:
    def test_allows_a_configured_origin(self):
        client = TestClient(create_app(ServerConfig(cors_origins=("http://localhost:5173",))))
        response = client.options(
            f"{API_PREFIX}/simulate",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_rejects_an_unconfigured_origin(self):
        client = TestClient(create_app(ServerConfig(cors_origins=("http://localhost:5173",))))
        response = client.options(
            f"{API_PREFIX}/simulate",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert response.headers.get("access-control-allow-origin") is None

    def test_no_cors_header_when_disabled(self):
        client = TestClient(create_app(ServerConfig(cors_origins=())))
        response = client.post(f"{API_PREFIX}/simulate", json=SCENE, headers={"Origin": "http://localhost:5173"})
        assert response.headers.get("access-control-allow-origin") is None


class TestOpenApi:
    def test_has_metadata(self, client):
        info = client.get("/openapi.json").json()["info"]
        assert info["title"] == "Beamline Playground API"
        assert info["version"]

    def test_documents_every_route(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert set(paths) == {f"{API_PREFIX}/health", f"{API_PREFIX}/simulate", f"{API_PREFIX}/scene/cost"}


class TestConfig:
    def test_reads_cors_origins_from_env(self):
        config = ServerConfig.from_env({"BEAMLINE_CORS_ORIGINS": "http://a.test, http://b.test"})
        assert config.cors_origins == ("http://a.test", "http://b.test")

    def test_reads_limits_from_env(self):
        config = ServerConfig.from_env({"BEAMLINE_MAX_PROPAGATION_PAIRS": "123", "BEAMLINE_MAX_TOTAL_ELEMENTS": "45"})
        assert config.max_propagation_pairs == 123
        assert config.max_total_elements == 45

    def test_defaults_are_used_when_env_is_empty(self):
        config = ServerConfig.from_env({})
        assert config.cors_origins
        assert config.max_propagation_pairs > 0

    def test_blank_origins_disable_cors(self):
        assert ServerConfig.from_env({"BEAMLINE_CORS_ORIGINS": ""}).cors_origins == ()


class TestImportIsCheap:
    def test_importing_the_server_runs_no_simulation(self):
        """The module used to simulate a scene at import time."""
        import subprocess
        import sys
        import time

        start = time.monotonic()
        completed = subprocess.run(
            [sys.executable, "-c", "import beamline_playground.server"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.monotonic() - start
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == "", f"import printed to stdout: {completed.stdout!r}"
        assert elapsed < 30
