import pytest
from pydantic import ValidationError

from mimir.core.context import Context, JsonState
from mimir.core.decisions import Choice, YesNo
from mimir.core.wire import BatchRequest, DecideRequest, ErrorBody, UncertifiedRequest


def test_decide_request_defaults_and_parsing() -> None:
    request = DecideRequest.model_validate(
        {"context": "text", "decision": {"type": "yes_no", "question": "q"}}
    )
    assert request.risk == 0.01
    assert request.alpha is None
    assert request.decision == YesNo("q")
    assert request.context == "text"


def test_decide_request_accepts_structured_states() -> None:
    request = DecideRequest.model_validate(
        {"context": {"state": {"a": 1}}, "decision": {"type": "yes_no", "question": "q"}}
    )
    assert request.context == JsonState(state={"a": 1})


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_must_be_strictly_between_zero_and_one(alpha: float) -> None:
    with pytest.raises(ValidationError, match="alpha"):
        DecideRequest(context="x", decision=YesNo("q"), alpha=alpha)


def test_batch_request_needs_items() -> None:
    with pytest.raises(ValidationError, match="items"):
        BatchRequest(items=[])


def test_requests_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        UncertifiedRequest.model_validate(
            {"context": "x", "decision": {"type": "yes_no", "question": "q"}, "risk": 0.01}
        )


def test_request_round_trip() -> None:
    request = DecideRequest(
        context=Context.coerce("hello"), decision=Choice("q", ["a", "b"]), risk=0.05, alpha=0.2
    )
    assert DecideRequest.model_validate_json(request.model_dump_json()) == request


def test_error_body_shape() -> None:
    body = ErrorBody.model_validate({"error": {"type": "InputLimitError", "message": "too big"}})
    assert body.error.message == "too big"
