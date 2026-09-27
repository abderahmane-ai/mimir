import json
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from mimir.compat.systemone.v1 import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    as_text,
    humanize,
    jev_confidence,
    respond,
    translate,
)
from mimir.core.context import Context, Field, Passage
from mimir.core.decisions import Choice, Rate
from mimir.core.results import ChoiceResult, DecisionResult, RateResult, Status

FIXTURE: Final = json.loads(
    (Path(__file__).parent / "fixtures" / "jev_v1.json").read_text(encoding="utf-8")
)


def choice_result(probabilities: dict[str, float], abstain: float = 0.0) -> ChoiceResult:
    return ChoiceResult(
        status=Status.DEFERRED,
        confidence=max(probabilities.values()),
        relevant_context=(),
        deferral=None,
        certificate=None,
        latency_ms=1.0,
        answer=max(probabilities, key=lambda key: probabilities[key]),
        probabilities=probabilities,
        abstain_probability=abstain,
        prediction_set=None,
    )


def rate_result(probabilities: dict[str, float]) -> RateResult:
    return RateResult(
        status=Status.DEFERRED,
        confidence=max(probabilities.values()),
        relevant_context=(),
        deferral=None,
        certificate=None,
        latency_ms=1.0,
        answer=max(probabilities, key=lambda key: probabilities[key]),
        probabilities=probabilities,
        prediction_set=None,
    )


def test_documented_requests_translate_to_training_phrasings() -> None:
    noul, choice, score = (SystemOneRequest.model_validate(item) for item in FIXTURE["requests"])
    context, questions = translate(noul)
    assert context == Context(
        passages=(Passage(text="Help! My payouts have been failing for 3 days."),)
    )
    assert questions["is_urgent"].kind == "noul"
    assert questions["is_urgent"].spec == Choice(
        "Does this convey urgency?",
        {"false": "False: No urgency expressed", "true": "True: Explicitly time-sensitive"},
    )
    assert translate(choice)[1]["department"].spec == Choice(
        "Which team should handle this?",
        {
            "billing": "Billing: Payments, invoicing, refunds",
            "technical": "Technical: Bugs, outages, integrations",
            "sales": "Sales: Pricing, upgrades, new accounts",
        },
    )
    translated = translate(score)[1]["frustration"]
    assert translated.spec == Rate(
        "How frustrated is the customer?", {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    )
    assert translated.legend == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}


def test_documented_responses_validate_and_round_trip() -> None:
    for item in FIXTURE["responses"]:
        response = SystemOneResponse.model_validate(item)
        assert json.loads(response.model_dump_json()) == item


def test_responses_have_the_documented_shape() -> None:
    documented = {
        next(iter(item["answers"].values()))["type"]: set(next(iter(item["answers"].values())))
        for item in FIXTURE["responses"]
    }
    requests = [SystemOneRequest.model_validate(item) for item in FIXTURE["requests"]]
    results: list[dict[str, DecisionResult]] = [
        {"is_urgent": choice_result({"false": 0.2, "true": 0.6}, abstain=0.2)},
        {"department": choice_result({"billing": 0.7, "technical": 0.2, "sales": 0.1})},
        {"frustration": rate_result({"0": 0.1, "1": 0.8, "2": 0.1})},
    ]
    for request, result in zip(requests, results, strict=True):
        response = respond(translate(request)[1], result, "mimir-1", 42)
        body = json.loads(response.model_dump_json())
        answer = next(iter(body["answers"].values()))
        assert set(answer) == documented[answer["type"]]
        assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}
        assert body["model"] == "mimir-1"


def test_answers_renormalise_over_options_and_compute_jev_confidence() -> None:
    request = SystemOneRequest.model_validate(FIXTURE["requests"][0])
    noul = respond(
        translate(request)[1],
        {"is_urgent": choice_result({"false": 0.2, "true": 0.6}, 0.2)},
        "m",
        0,
    ).answers["is_urgent"]
    assert isinstance(noul, NoulAnswer)
    assert noul.noul == pytest.approx(0.75)
    request = SystemOneRequest.model_validate(FIXTURE["requests"][1])
    choice = respond(
        translate(request)[1],
        {"department": choice_result({"billing": 0.45, "technical": 0.3, "sales": 0.15}, 0.1)},
        "m",
        0,
    ).answers["department"]
    assert isinstance(choice, ChoiceAnswer)
    assert choice.choice == "billing"
    assert sum(choice.probabilities.values()) == pytest.approx(1.0)
    assert choice.confidence == pytest.approx((3 * 0.5 - 1) / 2)
    request = SystemOneRequest.model_validate(FIXTURE["requests"][2])
    score = respond(
        translate(request)[1],
        {"frustration": rate_result({"0": 0.0, "1": 0.95, "2": 0.05})},
        "m",
        0,
    ).answers["frustration"]
    assert isinstance(score, ScoreAnswer)
    assert score.score == pytest.approx(1.05)


def test_two_level_scores_become_binary_choices() -> None:
    request = SystemOneRequest.model_validate(
        {
            "state": "x",
            "model": "jev-latest",
            "questions": {"q": {"type": "score", "instructions": "i", "criteria": ["no", "yes"]}},
        }
    )
    translated = translate(request)[1]["q"]
    assert translated.spec == Choice("i", {"0": "no", "1": "yes"})
    answer = respond({"q": translated}, {"q": choice_result({"0": 0.25, "1": 0.75})}, "m", 0)
    score = answer.answers["q"]
    assert isinstance(score, ScoreAnswer)
    assert score.score == pytest.approx(0.75)


def test_structured_state_and_instructions() -> None:
    request = SystemOneRequest.model_validate(
        {
            "state": {"ticket": {"id": 7}},
            "model": "jev-latest",
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": {"question": "Is `ticket` open?", "open": ["a", "b"]},
                }
            },
        }
    )
    context, questions = translate(request)
    assert context.fields == (Field(key="ticket.id", kind="number", value=7.0),)
    assert (
        questions["q"].spec.model_question
        == '{"question": "Is `ticket` open?", "open": ["a", "b"]}'
    )
    assert questions["q"].spec.option_texts == ("false", "true")


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "i", "criteria": {}},
        {"type": "choice", "instructions": "i", "criteria": {f"o{i}": None for i in range(256)}},
        {"type": "score", "instructions": "i", "criteria": ["only"]},
        {"type": "score", "instructions": "i", "criteria": [str(i) for i in range(11)]},
        {"type": "noul", "instructions": "i", "labels": {"true": "A", "false": "B"}},
        {"type": "guess", "instructions": "i"},
    ],
)
def test_requests_outside_the_documented_format_are_rejected(question: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "x", "model": "m", "questions": {"q": question}})


@pytest.mark.parametrize(
    ("label", "phrase"),
    [
        ("billing", "Billing"),
        ("needs_review", "Needs review"),
        ("customerService", "Customer service"),
        ("ABC", "ABC"),
        ("", ""),
    ],
)
def test_humanize(label: str, phrase: str) -> None:
    assert humanize(label) == phrase


def test_as_text_and_confidence_edges() -> None:
    assert as_text("plain") == "plain"
    assert as_text({"a": [1, 2]}) == '{"a": [1, 2]}'
    assert jev_confidence({"a": 1.0}) == 1.0
    assert jev_confidence({"a": 0.5, "b": 0.5}) == 0.0
