import asyncio

import pytest

from mimir.core.context import Context, Passage
from mimir.core.decisions import Choice, Estimate, Rank, Rate, Verify, YesNo
from mimir.core.errors import RiskLevelError
from mimir.core.results import ChoiceResult, EstimateResult, YesNoResult
from tests.conftest import RecordingDecider


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
    assert [call.risk for call in decider.calls] == [0.01, 0.01, None, None, *[0.01] * 6]
    assert decider.calls[3].batch_size == 2


def test_async_decide_rejects_a_missing_risk() -> None:
    decider = RecordingDecider()
    with pytest.raises(RiskLevelError):
        asyncio.run(decider.adecide("x", YesNo("q"), risk=None))
