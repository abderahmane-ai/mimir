import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from mimir.core.decisions import Choice
from mimir.core.errors import (
    ContextError,
    InputLimitError,
    MimirError,
    PolicyError,
    RiskLevelError,
)
from mimir.server.errors import install_error_handlers
from mimir.server.model import NotReadyError


class Body(BaseModel):
    count: int


def _client(error: Exception) -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/raise")
    async def raise_it() -> None:
        raise error

    @app.post("/body")
    async def body(value: Body) -> Body:
        return value

    return TestClient(app, raise_server_exceptions=False)


def _choice_error() -> Exception:
    try:
        Choice("q", ["only"])
    except ValueError as error:
        return error
    pytest.fail("a one-option choice must not construct")


@pytest.mark.parametrize(
    ("error", "status", "kind"),
    [
        (InputLimitError(limit="options", value=9, maximum=8), 422, "input_limit"),
        (RiskLevelError("risk 0.3 is not certified"), 422, "risk_level"),
        (ContextError("field 'x': unsupported"), 422, "invalid_context"),
        (PolicyError("no policy for variant fp32"), 409, "no_policy"),
        (NotReadyError("the model is still loading"), 503, "not_ready"),
    ],
)
def test_errors_map_to_their_status_and_type(error: Exception, status: int, kind: str) -> None:
    response = _client(error).get("/raise")
    assert response.status_code == status
    assert response.json() == {"error": {"type": kind, "message": str(error)}}


def test_not_ready_asks_the_client_to_retry() -> None:
    response = _client(NotReadyError("loading")).get("/raise")
    assert response.headers["retry-after"] == "5"


def test_spec_validation_is_an_invalid_request() -> None:
    response = _client(_choice_error()).get("/raise")
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request"
    assert "options needs at least 2 entries" in response.json()["error"]["message"]


def test_body_validation_names_the_field() -> None:
    response = _client(MimirError()).post("/body", json={"count": "many"})
    assert response.status_code == 422
    assert response.json()["error"] == {
        "type": "invalid_request",
        "message": "body.count: Input should be a valid integer, unable to parse string as an "
        "integer",
    }


def test_unknown_routes_and_methods() -> None:
    client = _client(MimirError())
    assert client.get("/nowhere").json() == {"error": {"type": "not_found", "message": "Not Found"}}
    assert client.get("/body").status_code == 405
    assert client.get("/body").json()["error"]["type"] == "method_not_allowed"


def test_a_defect_is_logged_and_its_message_kept_off_the_wire(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="mimir.server.errors"):
        response = _client(KeyError("secret internals")).get("/raise")
    assert response.status_code == 500
    assert response.json() == {"error": {"type": "internal", "message": "internal server error"}}
    assert "request failed method=GET path=/raise" in caplog.text
    assert "secret internals" in caplog.text
