import asyncio
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mimir.client.http import MimirClient
from mimir.core.decisions import Choice, Estimate, Rank, YesNo
from mimir.core.errors import ArtifactError, AuthenticationError, InvalidRequestError
from mimir.runtime.engine import Mimir
from mimir.server.app import create_app, openapi_document
from mimir.server.metrics import Metrics
from mimir.server.model import ServedModel
from tests.conftest import load_engine

TEXT = "my card was charged twice"
TEAM = {"type": "choice", "question": "which team", "options": ["billing", "security", "shipping"]}
OPENAPI_FILE = Path(__file__).parents[4] / "openapi.json"


def _app(root: Path, **options: object) -> tuple[FastAPI, ServedModel, Metrics]:
    metrics = Metrics()
    served = ServedModel(lambda: load_engine(root), observer=metrics)
    route = served.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    keys = options.get("api_keys", frozenset())
    app = create_app(
        served,
        metrics=metrics,
        tools=[route],
        api_keys=keys if isinstance(keys, frozenset) else frozenset(),
        max_body_bytes=int(str(options.get("max_body_bytes", 1_000_000))),
        max_batch_items=int(str(options.get("max_batch_items", 3))),
    )
    return app, served, metrics


@contextmanager
def _serving(app: FastAPI, headers: dict[str, str] | None = None) -> Iterator[TestClient]:
    with TestClient(app, headers=headers) as client:
        for _ in range(500):
            if client.get("/readyz").json()["status"] != "loading":
                break
            time.sleep(0.01)
        yield client


def _without_latency(body: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in body.items() if key != "latency_ms"}


def test_probes_follow_the_load(release: Path) -> None:
    app, _, _ = _app(release)
    with _serving(app) as client:
        assert client.get("/healthz").json() == {"status": "ready", "error": None}
        assert client.get("/readyz").status_code == 200


def test_a_failed_load_fails_liveness_and_every_decision(tmp_path: Path) -> None:
    def load() -> Mimir:
        message = f"{tmp_path}: signature missing"
        raise ArtifactError(message)

    metrics = Metrics()
    app = create_app(ServedModel(load), metrics=metrics)
    with _serving(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 503
        assert health.json()["status"] == "failed"
        assert health.json()["error"] == f"ArtifactError: {tmp_path}: signature missing"
        refused = client.post("/v1/decide", json={"context": TEXT, "decision": TEAM})
        assert refused.status_code == 503
        assert refused.json()["error"]["type"] == "not_ready"
        assert "failed to load" in refused.json()["error"]["message"]


def test_decide_matches_the_engine(release: Path, engine: Mimir) -> None:
    app, _, _ = _app(release)
    with _serving(app) as client:
        response = client.post("/v1/decide", json={"context": TEXT, "decision": TEAM})
        uncertified = client.post(
            "/v1/decide/uncertified", json={"context": TEXT, "decision": TEAM}
        )
    spec = Choice("which team", ["billing", "security", "shipping"])
    assert response.status_code == 200
    expected = engine.decide(TEXT, spec).model_dump(mode="json")
    assert _without_latency(response.json()) == _without_latency(expected)
    assert uncertified.json()["certificate"] is None
    raw = engine.decide_uncertified(TEXT, spec).model_dump(mode="json")
    assert _without_latency(uncertified.json()) == _without_latency(raw)


def test_batches_keep_order_and_their_item_limit(release: Path) -> None:
    app, _, _ = _app(release)
    body = [
        {"context": TEXT, "decision": TEAM},
        {
            "context": {"state": {"invoice": {"late": True}}},
            "decision": YesNo("is it late").model_dump(mode="json"),
        },
        {
            "context": [TEXT, "refund"],
            "decision": Rank("best", ["refund", "order"]).model_dump(mode="json"),
        },
    ]
    with _serving(app) as client:
        response = client.post("/v1/decide/batch", json={"items": body})
        over = client.post("/v1/decide/batch", json={"items": [*body, body[0]]})
    assert [result["type"] for result in response.json()["results"]] == [
        "choice",
        "yes_no",
        "rank",
    ]
    assert over.status_code == 422
    assert over.json()["error"] == {
        "type": "input_limit",
        "message": "batch items is 4; the release is tested up to 3",
    }


def test_configured_tools_take_only_the_context(release: Path) -> None:
    app, _, _ = _app(release)
    with _serving(app) as client:
        response = client.post("/v1/tools/route_ticket", json={"context": TEXT})
        unknown = client.post("/v1/tools/nope", json={"context": TEXT})
        extra = client.post("/v1/tools/route_ticket", json={"context": TEXT, "question": "q"})
    assert response.status_code == 200
    assert set(response.json()["probabilities"]) == {"billing", "security"}
    assert unknown.status_code == 404
    assert (
        unknown.json()["error"]["message"]
        == "no tool named 'nope'; this server has ['route_ticket']"
    )
    assert extra.status_code == 422
    assert "body.question: Extra inputs are not permitted" in extra.json()["error"]["message"]


def test_systemone_answers_in_jev_format(release: Path, engine: Mimir) -> None:
    app, _, _ = _app(release)
    request = {
        "state": TEXT,
        "model": "systemone",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "which team",
                "criteria": {"billing": "", "security": ""},
            },
            "late": {"type": "noul", "instructions": "is it late"},
        },
    }
    with _serving(app) as client:
        response = client.post("/v1/systemone", json=request)
    body = response.json()
    assert response.status_code == 200
    assert body["model"] == engine.info().model
    assert body["answers"]["team"]["type"] == "choice"
    assert body["answers"]["team"]["choice"] in {"billing", "security"}
    assert 0 <= body["answers"]["late"]["noul"] <= 1
    expected_tokens = engine.count_tokens(TEXT, Choice("which team", ["billing", "security"]))
    expected_tokens += engine.count_tokens(
        TEXT, Choice("is it late", {"false": "false", "true": "true"})
    )
    assert body["usage"] == {"input_tokens": expected_tokens, "output_tokens": 0}


def test_requests_the_release_cannot_take_are_refused_by_name(release: Path) -> None:
    app, _, _ = _app(release)
    options = [f"option {index}" for index in range(9)]
    with _serving(app) as client:
        limit = client.post(
            "/v1/decide",
            json={"context": TEXT, "decision": {**TEAM, "options": options}},
        )
        risk = client.post("/v1/decide", json={"context": TEXT, "decision": TEAM, "risk": 0.3})
        spec = client.post(
            "/v1/decide", json={"context": TEXT, "decision": {**TEAM, "options": ["one"]}}
        )
    assert (limit.status_code, limit.json()["error"]["type"]) == (422, "input_limit")
    assert limit.json()["error"]["message"] == "options is 9; the release is tested up to 8"
    assert (risk.status_code, risk.json()["error"]["type"]) == (422, "risk_level")
    assert "choose one of [0.01]" in risk.json()["error"]["message"]
    assert (spec.status_code, spec.json()["error"]["type"]) == (422, "invalid_request")
    assert "options needs at least 2 entries" in spec.json()["error"]["message"]


def test_a_release_without_a_policy_serves_only_uncertified_decisions(
    release_builder: object,
) -> None:
    assert callable(release_builder)
    root = release_builder("bare", with_policy=False)
    app, _, _ = _app(root)
    with _serving(app) as client:
        certified = client.post("/v1/decide", json={"context": TEXT, "decision": TEAM})
        uncertified = client.post(
            "/v1/decide/uncertified", json={"context": TEXT, "decision": TEAM}
        )
        info = client.get("/v1/models").json()
    assert (certified.status_code, certified.json()["error"]["type"]) == (409, "no_policy")
    assert uncertified.status_code == 200
    assert (info["certification"], info["risk_levels"]) == ("none", [])


def test_keys_guard_everything_but_the_probes(release: Path) -> None:
    app, _, _ = _app(release, api_keys=frozenset({"secret"}))
    with _serving(app) as client:
        assert client.get("/v1/models").status_code == 401
        assert client.get("/metrics").status_code == 401
        assert client.get("/healthz").status_code == 200
        authorized = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        assert authorized.status_code == 200


def test_bodies_over_the_limit_are_refused(release: Path) -> None:
    app, _, _ = _app(release, max_body_bytes=100)
    with _serving(app) as client:
        response = client.post("/v1/decide", json={"context": "x" * 200, "decision": TEAM})
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "too_large"


def test_metrics_count_requests_batches_and_statuses(release: Path) -> None:
    app, _, metrics = _app(release)
    with _serving(app) as client:
        for _ in range(2):
            client.post("/v1/decide", json={"context": TEXT, "decision": TEAM})
        client.post(
            "/v1/decide",
            json={"context": TEXT, "decision": Estimate("price", 0, 100).model_dump(mode="json")},
        )
        exposition = client.get("/metrics").text
    assert 'mimir_requests_total{code="200",route="/v1/decide"} 3.0' in exposition
    assert metrics.registry.get_sample_value("mimir_batch_requests_sum") == 3
    assert (
        metrics.registry.get_sample_value(
            "mimir_decisions_total", {"type": "estimate", "status": "deferred"}
        )
        == 1
    )


def test_the_client_round_trips_results_and_errors(release: Path, engine: Mimir) -> None:
    app, served, _ = _app(release, api_keys=frozenset({"secret"}))
    spec = Choice("which team", ["billing", "security", "shipping"])

    async def main() -> None:
        async with app.router.lifespan_context(app):
            while served.state == "loading":
                await asyncio.sleep(0.01)
            transport = httpx.ASGITransport(app=app)
            client = MimirClient("http://mimir", api_key="secret", async_transport=transport)
            result = await client.adecide(TEXT, spec)
            assert _without_latency(result.model_dump()) == _without_latency(
                engine.decide(TEXT, spec).model_dump()
            )
            with pytest.raises(InvalidRequestError, match=r"risk 0\.3 is not certified"):
                await client.adecide(TEXT, spec, risk=0.3)
            wrong = MimirClient("http://mimir", api_key="nope", async_transport=transport)
            with pytest.raises(AuthenticationError, match="HTTP 401"):
                await wrong.adecide(TEXT, spec)
            await client.aclose()
            await wrong.aclose()

    asyncio.run(main())


def test_the_openapi_document_is_3_1_and_committed() -> None:
    document = openapi_document()
    assert str(document["openapi"]).startswith("3.1")
    paths = document["paths"]
    assert isinstance(paths, dict)
    assert set(paths) == {
        "/v1/decide",
        "/v1/decide/uncertified",
        "/v1/decide/batch",
        "/v1/tools/{name}",
        "/v1/systemone",
        "/v1/models",
        "/healthz",
        "/readyz",
    }
    committed = json.loads(OPENAPI_FILE.read_text(encoding="utf-8"))
    assert committed == document, "openapi.json is stale: run `make openapi`"
