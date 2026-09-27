import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from mimir.client.http import MimirClient, response_error, retry_delay_s
from mimir.core.decisions import Choice, DecisionSpec, YesNo
from mimir.core.errors import (
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    ServerConnectionError,
    ServerError,
    ServerResponseError,
    ServerTimeoutError,
)
from mimir.core.results import ChoiceResult, YesNoResult
from mimir.core.wire import BatchRequest, BatchResponse, DecideRequest, UncertifiedRequest
from tests.conftest import RecordingDecider, result_for

Handler = Callable[[httpx.Request], httpx.Response]


def serving(requests: list[httpx.Request]) -> Handler:
    """A handler that answers every route the client uses, recording each request."""

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, content=RecordingDecider().info().model_dump_json())
        if path == "/v1/decide":
            body = DecideRequest.model_validate_json(request.content)
            return httpx.Response(200, content=result_for(body.decision).model_dump_json())
        if path == "/v1/decide/uncertified":
            body_uncertified = UncertifiedRequest.model_validate_json(request.content)
            return httpx.Response(
                200, content=result_for(body_uncertified.decision).model_dump_json()
            )
        if path == "/v1/decide/batch":
            batch = BatchRequest.model_validate_json(request.content)
            results = [result_for(item.decision) for item in batch.items]
            return httpx.Response(200, content=BatchResponse(results=results).model_dump_json())
        return httpx.Response(404, json={"error": {"type": "NotFound", "message": path}})

    return handle


def client(handler: Handler, *, max_retries: int = 2) -> MimirClient:
    return MimirClient(
        "https://mimir.test",
        api_key="secret",
        backoff_s=0.0,
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(handler),
        max_retries=max_retries,
    )


def test_single_decisions_use_decide_with_bearer_auth() -> None:
    seen: list[httpx.Request] = []
    with client(serving(seen)) as remote:
        result = remote.decide("ticket", Choice("q", ["a", "b"]), risk=0.05, alpha=0.2)
    assert isinstance(result, ChoiceResult)
    assert [request.url.path for request in seen] == ["/v1/decide"]
    assert seen[0].headers["authorization"] == "Bearer secret"
    body = json.loads(seen[0].content)
    assert (body["risk"], body["alpha"], body["decision"]["type"]) == (0.05, 0.2, "choice")


def test_many_decisions_are_sent_in_batches_of_batch_size() -> None:
    seen: list[httpx.Request] = []
    specs: list[DecisionSpec] = [YesNo(f"q{index}") for index in range(5)]
    with client(serving(seen)) as remote:
        results = remote.decide_many([("x", spec) for spec in specs], batch_size=2)
    assert [request.url.path for request in seen] == ["/v1/decide/batch"] * 3
    assert [len(json.loads(request.content)["items"]) for request in seen] == [2, 2, 1]
    assert all(isinstance(result, YesNoResult) for result in results)


def test_uncertified_and_info_routes() -> None:
    seen: list[httpx.Request] = []
    with client(serving(seen)) as remote:
        remote.decide_uncertified("x", YesNo("q"))
        info = remote.info()
    assert [request.url.path for request in seen] == ["/v1/decide/uncertified", "/v1/models"]
    assert info.model == "recording"


def test_api_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMIR_API_KEY", "from-env")
    seen: list[httpx.Request] = []
    remote = MimirClient("https://mimir.test", transport=httpx.MockTransport(serving(seen)))
    remote.info()
    assert seen[0].headers["authorization"] == "Bearer from-env"
    monkeypatch.delenv("MIMIR_API_KEY")
    anonymous = MimirClient("https://mimir.test", transport=httpx.MockTransport(serving(seen)))
    anonymous.info()
    assert "authorization" not in seen[1].headers


def test_retryable_statuses_are_retried_then_succeed() -> None:
    attempts: list[int] = []
    succeed = serving([])

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, headers={"Retry-After": "0"})
        return succeed(request)

    with client(flaky, max_retries=2) as remote:
        assert remote.info().model == "recording"
    assert len(attempts) == 3


def test_retries_are_bounded() -> None:
    attempts: list[int] = []

    def overloaded(_: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(529, text="overloaded")

    with client(overloaded, max_retries=1) as remote, pytest.raises(ServerError, match="529"):
        remote.info()
    assert len(attempts) == 2


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectTimeout("slow"), ServerTimeoutError),
        (httpx.ConnectError("down"), ServerConnectionError),
    ],
)
def test_transport_failures_raise_after_retries(
    error: Exception, expected: type[Exception]
) -> None:
    def failing(_: httpx.Request) -> httpx.Response:
        raise error

    with client(failing, max_retries=1) as remote, pytest.raises(expected, match="/v1/models"):
        remote.info()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, InvalidRequestError),
        (422, InvalidRequestError),
        (401, AuthenticationError),
        (403, AuthenticationError),
        (404, NotFoundError),
        (429, RateLimitError),
        (500, ServerError),
        (418, ServerResponseError),
    ],
)
def test_error_statuses_map_to_exceptions(status: int, expected: type[Exception]) -> None:
    response = httpx.Response(
        status, json={"error": {"type": "InputLimitError", "message": "options is 200"}}
    )
    error = response_error(response)
    assert type(error) is expected
    assert error.status == status
    assert "options is 200" in str(error)


def test_error_without_a_json_body_keeps_the_text() -> None:
    error = response_error(httpx.Response(502, text="bad gateway page"))
    assert str(error) == "HTTP 502: bad gateway page"
    assert error.body == "bad gateway page"


def test_retry_delay_honours_retry_after_and_caps_backoff() -> None:
    assert retry_delay_s(httpx.Response(429, headers={"Retry-After": "2.5"}), 0, 0.5) == 2.5
    assert retry_delay_s(httpx.Response(429, headers={"Retry-After": "soon"}), 1, 0.5) == 1.0
    assert retry_delay_s(None, 3, 0.5) == 4.0
    assert retry_delay_s(None, 20, 0.5) == 30.0


def test_negative_retries_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        MimirClient("https://mimir.test", max_retries=-1)


def test_async_methods_use_the_async_transport() -> None:
    seen: list[httpx.Request] = []
    remote = client(serving(seen))

    async def run() -> None:
        await remote.adecide("x", YesNo("q"))
        await remote.adecide_many([("x", YesNo("a")), ("y", YesNo("b"))])
        await remote.adecide_uncertified("x", YesNo("q"))
        await remote.aclose()

    asyncio.run(run())
    assert [request.url.path for request in seen] == [
        "/v1/decide",
        "/v1/decide/batch",
        "/v1/decide/uncertified",
    ]
