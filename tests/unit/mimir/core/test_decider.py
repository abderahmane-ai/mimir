import asyncio
from collections.abc import Sequence

import pytest

from mimir.core.context import Context, Passage
from mimir.core.decisions import (
    DEFAULT_CHOICE_QUESTION,
    Choice,
    DecisionSpec,
    Estimate,
    Rank,
    Rate,
    Verify,
    YesNo,
)
from mimir.core.errors import RiskLevelError
from mimir.core.results import ChoiceResult, DecisionResult, EstimateResult, Status, YesNoResult
from mimir.core.wire import DEFAULT_RISK, Mode
from tests.conftest import RecordingDecider

TEAMS = ["billing", "technical", "sales"]


class ChoiceDecider(RecordingDecider):
    """A `RecordingDecider` answering every `Choice` with `answer` at `status`."""

    def __init__(self, *, answer: str | None, status: Status) -> None:
        super().__init__()
        self.answer = answer
        self.status = status

    def _run(
        self,
        requests: Sequence[tuple[Context, DecisionSpec]],
        *,
        mode: Mode,
        min_confidence: float | None,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        results = super()._run(
            requests,
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
            batch_size=batch_size,
        )
        update = {
            "answer": self.answer,
            "status": self.status,
            "actionable": self.status in (Status.DECIDED, Status.ABSTAINED),
        }
        return [
            ChoiceResult.model_validate({**result.model_dump(), **update})
            if isinstance(result, ChoiceResult)
            else result
            for result in results
        ]


def test_decide_coerces_the_context_and_passes_risk_and_alpha() -> None:
    decider = RecordingDecider()
    result = decider.decide("ticket", Choice("q", ["a", "b"]), risk=0.05, alpha=0.2)
    assert isinstance(result, ChoiceResult)
    call = decider.calls[0]
    assert call.requests == (
        (Context(passages=(Passage(text="ticket"),)), Choice("q", ["a", "b"])),
    )
    assert (call.risk, call.alpha, call.batch_size) == (0.05, 0.2, None)


def test_decide_rejects_a_missing_risk() -> None:
    decider = RecordingDecider()
    with pytest.raises(RiskLevelError, match="decide_uncertified"):
        decider.decide("x", YesNo("q"), risk=None)
    assert decider.calls == []


def test_decide_uncertified_runs_without_risk() -> None:
    decider = RecordingDecider()
    assert isinstance(decider.decide_uncertified("x", YesNo("q")), YesNoResult)
    assert decider.calls[0].risk is None


def test_decide_uncertified_many_runs_one_batch_without_risk() -> None:
    decider = RecordingDecider()
    items = [("a", YesNo("one")), ("b", Rank("two", ["x", "y"]))]
    results = decider.decide_uncertified_many(items, batch_size=3)
    assert [result.type for result in results] == ["yes_no", "rank"]
    assert len(decider.calls) == 1
    call = decider.calls[0]
    assert (call.risk, call.alpha, call.batch_size) == (None, None, 3)
    assert [context for context, _ in call.requests] == [
        Context(passages=(Passage(text="a"),)),
        Context(passages=(Passage(text="b"),)),
    ]


def test_decide_many_keeps_order_and_batch_size() -> None:
    decider = RecordingDecider()
    items = [("a", YesNo("one")), ({"k": 1}, Estimate("two", 0.0, 1.0))]
    results = decider.decide_many(items, batch_size=7)
    assert [type(result) for result in results] == [YesNoResult, EstimateResult]
    assert decider.calls[0].batch_size == 7
    assert [spec for _, spec in decider.calls[0].requests] == [spec for _, spec in items]


def test_shortcuts_build_the_expected_specs() -> None:
    decider = RecordingDecider()
    decider.choose("x", "q", ["a", "b"])
    decider.yes_no("x", "q")
    decider.verify("x", "claim")
    decider.rank("x", "q", ["a", "b"])
    decider.rate("x", "q", ["l", "m", "h"])
    decider.estimate("x", "q", 0, 10, unit="kg")
    specs = [call.requests[0][1] for call in decider.calls]
    assert specs == [
        Choice("q", ["a", "b"]),
        YesNo("q"),
        Verify("claim"),
        Rank("q", ["a", "b"]),
        Rate("q", ["l", "m", "h"]),
        Estimate("q", 0, 10, "kg"),
    ]


def test_async_methods_match_the_sync_ones() -> None:
    decider = RecordingDecider()

    async def run() -> None:
        await decider.adecide("x", YesNo("q"))
        await decider.adecide_many([("x", YesNo("q"))])
        await decider.adecide_uncertified("x", YesNo("q"))
        await decider.adecide_uncertified_many([("x", YesNo("q"))], batch_size=2)
        await decider.achoose("x", "q", ["a", "b"])
        await decider.ayes_no("x", "q")
        await decider.averify("x", "c")
        await decider.arank("x", "q", ["a", "b"])
        await decider.arate("x", "q", ["l", "m", "h"])
        await decider.aestimate("x", "q", 0, 1)

    asyncio.run(run())
    assert [call.risk for call in decider.calls] == [0.05, 0.05, None, None, *[0.05] * 6]
    assert decider.calls[3].batch_size == 2


def test_async_decide_rejects_a_missing_risk() -> None:
    decider = RecordingDecider()
    with pytest.raises(RiskLevelError):
        asyncio.run(decider.adecide("x", YesNo("q"), risk=None))


def test_default_risk_is_the_95_percent_level() -> None:
    assert DEFAULT_RISK == 0.05


@pytest.mark.parametrize(
    ("second", "expected"),
    [
        (TEAMS, Choice(DEFAULT_CHOICE_QUESTION, TEAMS)),
        (
            {"b": "Billing", "t": "Technical"},
            Choice(DEFAULT_CHOICE_QUESTION, {"b": "Billing", "t": "Technical"}),
        ),
        (("a", "b"), Choice(DEFAULT_CHOICE_QUESTION, ["a", "b"])),
    ],
)
def test_choose_without_a_question_reads_the_second_argument_as_options(
    second: Sequence[str] | dict[str, str], expected: Choice
) -> None:
    decider = RecordingDecider()
    decider.choose("x", second)
    assert decider.calls[0].requests[0][1] == expected
    assert decider.calls[0].risk == DEFAULT_RISK


def test_choose_takes_options_by_keyword_with_or_without_a_question() -> None:
    decider = RecordingDecider()
    decider.choose("x", options=TEAMS)
    decider.choose("x", "Which team?", options=TEAMS)
    decider.choose("x", question="Which team?", options=TEAMS)
    assert [call.requests[0][1] for call in decider.calls] == [
        Choice(DEFAULT_CHOICE_QUESTION, TEAMS),
        Choice("Which team?", TEAMS),
        Choice("Which team?", TEAMS),
    ]


def test_choose_treats_a_text_second_argument_as_the_question() -> None:
    decider = RecordingDecider()
    with pytest.raises(TypeError, match=r"question='billing' and options=None"):
        decider.choose("x", "billing")
    assert decider.calls == []


def test_choose_rejects_options_given_twice_and_none_given() -> None:
    decider = RecordingDecider()
    with pytest.raises(TypeError, match=r"options twice: \['a', 'b'\] as the second argument"):
        decider.choose("x", ["a", "b"], ["c", "d"])
    with pytest.raises(TypeError, match=r"question=None and options=None"):
        decider.choose("x")
    assert decider.calls == []


def test_achoose_without_a_question_matches_choose() -> None:
    decider = RecordingDecider()

    async def run() -> None:
        await decider.achoose("x", TEAMS)
        await decider.achoose("x", options=TEAMS, risk=0.01)
        await decider.achoose("x", "Which team?", TEAMS)

    asyncio.run(run())
    assert [call.requests[0][1] for call in decider.calls] == [
        Choice(DEFAULT_CHOICE_QUESTION, TEAMS),
        Choice(DEFAULT_CHOICE_QUESTION, TEAMS),
        Choice("Which team?", TEAMS),
    ]
    assert [call.risk for call in decider.calls] == [0.05, 0.01, 0.05]


@pytest.mark.parametrize(
    ("answer", "status", "expected"),
    [
        ("billing", Status.DECIDED, "billing"),
        (None, Status.ABSTAINED, None),
        ("billing", Status.DEFERRED, None),
    ],
)
def test_pick_returns_the_answer_only_when_decided(
    answer: str | None, status: Status, expected: str | None
) -> None:
    decider = ChoiceDecider(answer=answer, status=status)

    async def run() -> str | None:
        return await decider.apick("x", TEAMS)

    assert decider.pick("x", TEAMS) == expected
    assert asyncio.run(run()) == expected


def test_pick_passes_question_mode_floor_and_risk() -> None:
    decider = ChoiceDecider(answer="billing", status=Status.DECIDED)
    assert decider.pick("x", TEAMS) == "billing"
    assert (
        decider.pick(
            "x",
            TEAMS,
            question="Which team?",
            mode=Mode.THRESHOLD,
            min_confidence=0.7,
            risk=0.01,
        )
        == "billing"
    )
    first, second = decider.calls
    assert first.requests[0][1] == Choice(DEFAULT_CHOICE_QUESTION, TEAMS)
    assert (first.mode, first.min_confidence, first.risk) == (Mode.STANDARD, None, DEFAULT_RISK)
    assert second.requests[0][1] == Choice("Which team?", TEAMS)
    assert (second.mode, second.min_confidence, second.risk) == (Mode.THRESHOLD, 0.7, 0.01)
