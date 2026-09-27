from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mimir.core.decisions import Choice, YesNo
from mimir.core.results import Status
from mimir.server.metrics import CONTENT_TYPE, Metrics, RequestMetrics
from tests.conftest import result_for


def _sample(metrics: Metrics, name: str, labels: dict[str, str]) -> float | None:
    return metrics.registry.get_sample_value(name, labels)


def _ok(_: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _broken(_: Request) -> PlainTextResponse:
    return PlainTextResponse("no", status_code=503)


def test_batches_count_requests_and_statuses() -> None:
    metrics = Metrics()
    choice = Choice("q", ["a", "b"])
    metrics.observe_batch(
        [result_for(choice), result_for(choice, Status.DEFERRED), result_for(YesNo("q"))]
    )
    metrics.observe_batch([result_for(choice)])
    assert _sample(metrics, "mimir_batch_requests_count", {}) == 2
    assert _sample(metrics, "mimir_batch_requests_sum", {}) == 4
    assert _sample(metrics, "mimir_decisions_total", {"type": "choice", "status": "decided"}) == 2
    assert _sample(metrics, "mimir_decisions_total", {"type": "choice", "status": "deferred"}) == 1
    assert _sample(metrics, "mimir_decisions_total", {"type": "yes_no", "status": "decided"}) == 1


def test_requests_are_labelled_by_route_template_or_unmatched() -> None:
    metrics = Metrics()
    routes = [Route("/v1/tools/{name}", _ok), Route("/broken", _broken)]
    client = TestClient(RequestMetrics(Starlette(routes=routes), metrics))
    client.get("/v1/tools/route_ticket")
    client.get("/v1/tools/other")
    client.get("/broken")
    client.get("/nowhere")
    tool = {"route": "/v1/tools/{name}", "code": "200"}
    assert _sample(metrics, "mimir_requests_total", tool) == 2
    assert _sample(metrics, "mimir_requests_total", {"route": "/broken", "code": "503"}) == 1
    assert _sample(metrics, "mimir_requests_total", {"route": "unmatched", "code": "404"}) == 1
    assert _sample(metrics, "mimir_request_seconds_count", {"route": "/v1/tools/{name}"}) == 2


def test_each_server_has_its_own_registry() -> None:
    first, second = Metrics(), Metrics()
    first.observe_request("/v1/decide", 200, 0.1)
    assert _sample(second, "mimir_requests_total", {"route": "/v1/decide", "code": "200"}) is None
    assert b'mimir_requests_total{code="200",route="/v1/decide"} 1.0' in first.exposition()
    assert CONTENT_TYPE.startswith("text/plain")
