import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mimir.server.auth import (
    BearerKeys,
    api_keys_from_environment,
    check_exposure,
    is_authorized,
    is_loopback,
)

KEYS = frozenset({"key-one", "key-two"})


def _ok(_: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _client(keys: frozenset[str]) -> TestClient:
    routes = [Route(path, _ok) for path in ("/v1/models", "/healthz", "/readyz", "/metrics")]
    return TestClient(BearerKeys(Starlette(routes=routes), keys))


def test_keys_are_read_from_the_environment_without_blanks() -> None:
    environ = {"MIMIR_API_KEYS": " key-one, ,key-two,, "}
    assert api_keys_from_environment(environ) == KEYS
    assert api_keys_from_environment({}) == frozenset()
    assert api_keys_from_environment({"MIMIR_API_KEYS": " , "}) == frozenset()


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.8.9.1", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", False),
        ("::", False),
        ("10.0.0.5", False),
        ("mimir.internal", False),
    ],
)
def test_loopback_hosts(host: str, expected: bool) -> None:
    assert is_loopback(host) is expected


def test_a_public_host_without_keys_needs_the_explicit_flag() -> None:
    with pytest.raises(ValueError, match=r"--host 0\.0\.0\.0 .*MIMIR_API_KEYS holds no key"):
        check_exposure("0.0.0.0", frozenset(), allow_no_auth=False)
    check_exposure("0.0.0.0", frozenset(), allow_no_auth=True)
    check_exposure("0.0.0.0", KEYS, allow_no_auth=False)
    check_exposure("127.0.0.1", frozenset(), allow_no_auth=False)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer key-one", True),
        ("bearer key-two", True),
        ("Bearer  key-one ", True),
        ("Bearer key-three", False),
        ("Bearer key-on", False),
        ("Bearer ", False),
        ("Basic key-one", False),
        ("key-one", False),
        ("", False),
        (None, False),
    ],
)
def test_bearer_tokens(header: str | None, expected: bool) -> None:
    assert is_authorized(header, KEYS) is expected


def test_the_middleware_refuses_missing_and_wrong_keys_with_401() -> None:
    client = _client(KEYS)
    for headers in ({}, {"Authorization": "Bearer nope"}):
        response = client.get("/v1/models", headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json() == {
            "error": {"type": "unauthorized", "message": "missing or invalid bearer API key"}
        }
    assert client.get("/metrics").status_code == 401
    authorized = client.get("/v1/models", headers={"Authorization": "Bearer key-two"})
    assert (authorized.status_code, authorized.text) == (200, "ok")


def test_probes_stay_open_and_no_keys_means_no_check() -> None:
    client = _client(KEYS)
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert _client(frozenset()).get("/v1/models").status_code == 200
