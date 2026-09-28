from examples.usecases import (
    estimate_cost,
    gate_tool_call,
    rank_vendors,
    rate_severity,
    route_ticket,
    verify_claim,
)
from mimir import Permission, Status
from tests.conftest import RecordingDecider


def test_route_ticket_picks_a_team() -> None:
    decider = RecordingDecider()
    result = route_ticket.route(decider, "I was charged twice for order 4412.")
    assert (result.status, result.answer) == (Status.DECIDED, "billing")
    [(context, spec)] = decider.calls[0].requests
    assert context.passages[0].text == "I was charged twice for order 4412."
    assert spec.option_ids == tuple(route_ticket.TEAMS)


def test_verify_claim_judges_the_claim() -> None:
    result = verify_claim.check(RecordingDecider(), "Exports fail nightly.", ["Exports fail."])
    assert (result.status, result.answer) == (Status.DECIDED, "supported")


def test_rank_vendors_orders_best_first() -> None:
    result = rank_vendors.rank(RecordingDecider())
    assert result.status is Status.DECIDED
    assert tuple(result.answer) == tuple(rank_vendors.VENDORS)


def test_rate_severity_picks_a_level() -> None:
    result = rate_severity.rate(RecordingDecider(), "Checkout is down.")
    assert (result.status, result.answer) == (Status.DECIDED, "low")


def test_estimate_cost_stays_in_range() -> None:
    result = estimate_cost.estimate(RecordingDecider(), "Replace a screen.")
    assert result.status is Status.DECIDED
    assert result.answer == 2500.0
    assert result.interval == (0.0, 5000.0)


def test_gate_denies_an_uncertified_refund() -> None:
    outcome = gate_tool_call.gate(RecordingDecider(), 900.0)
    assert outcome.permission is Permission.DENY
