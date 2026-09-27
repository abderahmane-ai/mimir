import asyncio
from collections.abc import Iterator

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from mimir.core.wire import MAX_BODY_BYTES
from mimir.server.limits import BodyLimit

LIMIT = 10


async def _echo(request: Request) -> JSONResponse:
    body = await request.body()
    return JSONResponse({"bytes": len(body), "text": body.decode()})


def _client(limit: int = LIMIT) -> TestClient:
    return TestClient(BodyLimit(Starlette(routes=[Route("/", _echo, methods=["POST"])]), limit))


def _chunks(*parts: bytes) -> Iterator[bytes]:
    yield from parts


def test_a_declared_length_over_the_limit_is_refused_unread() -> None:
    response = _client().post("/", content=b"x" * (LIMIT + 1))
    assert response.status_code == 413
    assert response.json() == {
        "error": {"type": "too_large", "message": "request body is 11 bytes; the limit is 10"}
    }


def test_bodies_at_the_limit_pass_whole() -> None:
    response = _client().post("/", content=b"0123456789")
    assert response.json() == {"bytes": 10, "text": "0123456789"}


def test_a_streamed_body_is_counted_and_refused_past_the_limit() -> None:
    response = _client().post("/", content=_chunks(b"abcdef", b"ghijkl"))
    assert response.status_code == 413
    assert response.json()["error"]["message"] == (
        "request body is at least 12 bytes; the limit is 10"
    )


def test_a_streamed_body_within_the_limit_reaches_the_app_whole() -> None:
    response = _client().post("/", content=_chunks(b"ab", b"", b"cd"))
    assert response.json() == {"bytes": 4, "text": "abcd"}


def test_a_malformed_content_length_is_refused() -> None:
    received: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive, send
        pytest.fail("the app must not run")

    async def send(message: Message) -> None:
        received.append(message)

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [(b"content-length", b"-1")],
    }
    asyncio.run(BodyLimit(app, LIMIT)(scope, receive, send))
    assert received[0]["status"] == 400


def test_the_default_limit_and_its_validation() -> None:
    assert BodyLimit(Starlette()).max_bytes == MAX_BODY_BYTES == 4 * 1024 * 1024
    with pytest.raises(ValueError, match="at least 1; got 0"):
        BodyLimit(Starlette(), 0)
