import numpy as np
import pytest

from mimir.core.decisions import (
    Choice,
    DecisionSpec,
    Estimate,
    MultiChoice,
    Rank,
    Rate,
    Verify,
    YesNo,
)
from mimir.core.results import (
    ChoiceResult,
    DecisionResult,
    EstimateResult,
    MultiChoiceResult,
    OptionSet,
    RankResult,
    RateResult,
    Status,
    VerifyResult,
    YesNoResult,
)
from mimir.policy.assessment import Assessment
from mimir.policy.distributions import Decision
from mimir.policy.document import CertifiedThreshold
from mimir.runtime.answer import Outcome, Provenance, build_result, relevant_context
from mimir.runtime.rendering import PASSAGE, Part, Segment

PROVENANCE = Provenance(model="m", revision="r", variant="fp32", confidence=0.95, origin="release")
ENTRY = CertifiedThreshold(
    threshold=0.8,
    records=100,
    taken=40,
    errors=0,
    p_value=0.01,
    coverage=0.4,
    coverage_interval=(0.3, 0.5),
)


def outcome(
    probs: list[float],
    decision: Decision,
    *,
    passes_gate: bool = True,
    taken: bool = True,
    entry: CertifiedThreshold | None = ENTRY,
    prediction_set: tuple[int, ...] | None = None,
) -> Outcome:
    values = np.array(probs)
    found = Assessment(
        probabilities=values,
        decision=decision,
        gate_p_value=0.5 if passes_gate else 0.001,
        passes_gate=passes_gate,
        certificate=entry,
        taken=taken,
        prediction_set=prediction_set,
    )
    return Outcome(values, decision, found)


def build(spec: DecisionSpec, item: Outcome) -> DecisionResult:
    return build_result(spec, item, (), 0.01, PROVENANCE, 3.0)


def test_decided_choice_carries_ids_certificate_and_prediction_set() -> None:
    spec = Choice("q", {"b": "Billing", "s": "Security"})
    result = build(spec, outcome([0.9, 0.05, 0.05], Decision((0,), 0.9), prediction_set=(0, 2)))
    assert isinstance(result, ChoiceResult)
    assert (result.status, result.answer, result.confidence) == (Status.DECIDED, "b", 0.9)
    assert result.probabilities == {"b": 0.9, "s": 0.05}
    assert result.abstain_probability == 0.05
    assert result.prediction_set == OptionSet(options=("b",), abstain=True)
    assert result.certificate is not None
    assert (result.certificate.threshold, result.certificate.risk) == (0.8, 0.01)
    assert result.deferral is None
    assert result.latency_ms == 3.0


def test_abstention_taken_is_abstained() -> None:
    result = build(Choice("q", ["a", "b"]), outcome([0.02, 0.03, 0.95], Decision((), 0.95)))
    assert isinstance(result, ChoiceResult)
    assert (result.status, result.answer) == (Status.ABSTAINED, None)


DEFERRALS = [
    (outcome([0.6, 0.4, 0.0], Decision((0,), 0.6), entry=None), "no_certified_threshold", None),
    (
        outcome(
            [0.6, 0.4, 0.0], Decision((0,), 0.6), entry=ENTRY.model_copy(update={"threshold": None})
        ),
        "no_certified_threshold",
        None,
    ),
    (
        outcome([0.6, 0.4, 0.0], Decision((0,), 0.6), passes_gate=False, taken=False),
        "out_of_distribution",
        0.8,
    ),
    (outcome([0.6, 0.4, 0.0], Decision((0,), 0.6), taken=False), "below_threshold", 0.8),
]


@pytest.mark.parametrize(("item", "reason", "threshold"), DEFERRALS)
def test_deferral_reasons(item: Outcome, reason: str, threshold: float | None) -> None:
    result = build(YesNo("q"), item)
    assert isinstance(result, YesNoResult)
    assert result.status == Status.DEFERRED
    assert result.answer is False
    assert result.deferral is not None
    assert (result.deferral.reason, result.deferral.threshold) == (reason, threshold)


def test_verify_answers_with_verdicts() -> None:
    result = build(Verify("c"), outcome([0.1, 0.8, 0.05, 0.05], Decision((1,), 0.8)))
    assert isinstance(result, VerifyResult)
    assert result.answer == "contradicted"
    assert set(result.probabilities) == {"supported", "contradicted", "not_enough_information"}


def test_multi_choice_rank_and_rate() -> None:
    multi = build(
        MultiChoice("q", ["a", "b", "c"]), outcome([0.9, 0.2, 0.7], Decision((0, 2), 0.5))
    )
    assert isinstance(multi, MultiChoiceResult)
    assert multi.answer == ("a", "c")
    rank = build(Rank("q", ["a", "b", "c"]), outcome([0.2, 0.5, 0.3], Decision((1,), 0.5)))
    assert isinstance(rank, RankResult)
    assert rank.answer == ("b", "c", "a")
    rate = build(
        Rate("q", ["l", "m", "h"]),
        outcome([0.1, 0.6, 0.3], Decision((1,), 0.6), prediction_set=(1, 2)),
    )
    assert isinstance(rate, RateResult)
    assert (rate.answer, rate.prediction_set) == ("m", ("m", "h"))


def test_estimate_value_is_the_mean_and_interval_spans_bins() -> None:
    probs = [0.0, 0.5, 0.5, 0.0]
    result = build(
        Estimate("q", 0.0, 8.0, "kg"),
        outcome(probs, Decision((1,), None), entry=None, prediction_set=(1, 2)),
    )
    assert isinstance(result, EstimateResult)
    assert result.answer == pytest.approx(4.0)
    assert result.interval == (2.0, 6.0)
    assert result.unit == "kg"
    assert result.confidence is None
    empty = build(
        Estimate("q", 0.0, 8.0), outcome(probs, Decision((1,), None), entry=None, prediction_set=())
    )
    assert isinstance(empty, EstimateResult)
    assert empty.interval == ()


def test_uncertified_results_have_no_policy_fields() -> None:
    values = np.array([0.7, 0.2, 0.1])
    result = build_result(
        Choice("q", ["a", "b"]), Outcome(values, Decision((0,), 0.7), None), (), None, None, 1.0
    )
    assert isinstance(result, ChoiceResult)
    assert (result.status, result.certificate, result.deferral, result.prediction_set) == (
        Status.DECIDED,
        None,
        None,
        None,
    )


def test_relevant_context_is_positive_parts_highest_first() -> None:
    state = (
        Segment(PASSAGE, 0, "first", Part("passage", 0, None)),
        Segment(PASSAGE, 1, "second", Part("passage", 1, None)),
        Segment(PASSAGE, 2, "third", Part("passage", 2, None)),
    )
    found = relevant_context(state, np.array([0.2, 0.0, 0.7]))
    assert [(item.index, item.relevance, item.text) for item in found] == [
        (2, 0.7, "third"),
        (0, 0.2, "first"),
    ]
