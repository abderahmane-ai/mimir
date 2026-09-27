"""Building result models from model outputs and policy assessments."""

from dataclasses import dataclass
from typing import Final, Literal, TypedDict

import numpy as np
import numpy.typing as npt

from mimir.core.decisions import (
    VERDICTS,
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
    Certificate,
    ChoiceResult,
    ContextRelevance,
    DecisionResult,
    Deferral,
    DeferralReason,
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
from mimir.policy.distributions import CHOICE_TYPES, Decision
from mimir.runtime.rendering import Segment

Floats = npt.NDArray[np.float64]

YES: Final = "yes"


class _Common(TypedDict):
    status: Status
    confidence: float | None
    relevant_context: tuple[ContextRelevance, ...]
    deferral: Deferral | None
    certificate: Certificate | None
    latency_ms: float


@dataclass(frozen=True, slots=True)
class Provenance:
    """Model identity copied into certificates."""

    model: str
    revision: str
    variant: str
    confidence: float
    origin: Literal["release", "custom"]


@dataclass(frozen=True, slots=True)
class Outcome:
    """Distribution and decision for one request, and the policy assessment if applied."""

    probabilities: Floats
    decision: Decision
    assessment: Assessment | None


def relevant_context(state: tuple[Segment, ...], relevance: Floats) -> tuple[ContextRelevance, ...]:
    """Return context parts with positive relevance, highest first."""
    order = sorted(
        (index for index in range(len(state)) if relevance[index] > 0),
        key=lambda index: -relevance[index],
    )
    return tuple(
        ContextRelevance(
            kind=state[index].part.kind,
            index=state[index].part.index,
            row=state[index].part.row,
            relevance=float(relevance[index]),
            text=state[index].text,
        )
        for index in order
    )


def _status(outcome: Outcome, is_abstain: bool) -> tuple[Status, Deferral | None]:
    assessment = outcome.assessment
    decided = Status.ABSTAINED if is_abstain else Status.DECIDED
    if assessment is None:
        return decided, None
    entry = assessment.certificate
    threshold = None if entry is None else entry.threshold
    reason: DeferralReason
    if threshold is None:
        reason = "no_certified_threshold"
    elif not assessment.passes_gate:
        reason = "out_of_distribution"
    elif not assessment.taken:
        reason = "below_threshold"
    else:
        return decided, None
    return Status.DEFERRED, Deferral(
        reason=reason, threshold=threshold, gate_p_value=assessment.gate_p_value
    )


def _certificate(
    outcome: Outcome, risk: float | None, provenance: Provenance | None
) -> Certificate | None:
    assessment = outcome.assessment
    entry = None if assessment is None else assessment.certificate
    if entry is None or entry.threshold is None or risk is None or provenance is None:
        return None
    return Certificate(
        risk=risk,
        confidence=provenance.confidence,
        threshold=entry.threshold,
        records=entry.records,
        taken=entry.taken,
        errors=entry.errors,
        p_value=entry.p_value,
        coverage=entry.coverage,
        coverage_interval=entry.coverage_interval,
        model=provenance.model,
        revision=provenance.revision,
        variant=provenance.variant,
        origin=provenance.origin,
    )


def _option_set(indices: tuple[int, ...] | None, ids: tuple[str, ...]) -> OptionSet | None:
    if indices is None:
        return None
    count = len(ids)
    return OptionSet(
        options=tuple(ids[index] for index in indices if index < count),
        abstain=count in indices,
    )


def _estimate_value(probs: Floats, low: float, high: float) -> float:
    bins = probs.shape[0]
    centres = (np.arange(bins) + 0.5) / bins
    return low + (high - low) * float(probs @ centres)


def _estimate_interval(
    indices: tuple[int, ...] | None, bins: int, low: float, high: float
) -> tuple[float, float] | tuple[()] | None:
    if indices is None:
        return None
    if not indices:
        return ()
    width = (high - low) / bins
    return (low + width * indices[0], low + width * (indices[-1] + 1))


def build_result(
    spec: DecisionSpec,
    outcome: Outcome,
    relevance: tuple[ContextRelevance, ...],
    risk: float | None,
    provenance: Provenance | None,
    latency_ms: float,
) -> DecisionResult:
    """Build the typed result for `spec`, with option indices mapped to the caller's ids."""
    probs = outcome.probabilities
    chosen = outcome.decision.chosen
    ids = spec.option_ids
    is_abstain = spec.model_type in CHOICE_TYPES and not chosen
    status, deferral = _status(outcome, is_abstain)
    prediction = None if outcome.assessment is None else outcome.assessment.prediction_set
    common = _Common(
        status=status,
        confidence=outcome.decision.score,
        relevant_context=relevance,
        deferral=deferral,
        certificate=_certificate(outcome, risk, provenance),
        latency_ms=latency_ms,
    )
    count = len(ids)
    by_id = {option: float(probs[index]) for index, option in enumerate(ids)}
    answer = ids[chosen[0]] if spec.model_type in CHOICE_TYPES and chosen else None
    match spec:
        case Choice():
            return ChoiceResult(
                answer=answer,
                probabilities=by_id,
                abstain_probability=float(probs[count]),
                prediction_set=_option_set(prediction, ids),
                **common,
            )
        case YesNo():
            return YesNoResult(
                answer=None if answer is None else answer == YES,
                probabilities=by_id,
                abstain_probability=float(probs[count]),
                prediction_set=_option_set(prediction, ids),
                **common,
            )
        case Verify():
            return VerifyResult(
                answer=None if answer is None else VERDICTS[chosen[0]],
                probabilities=by_id,
                abstain_probability=float(probs[count]),
                prediction_set=_option_set(prediction, ids),
                **common,
            )
        case MultiChoice():
            return MultiChoiceResult(
                answer=tuple(ids[index] for index in chosen), probabilities=by_id, **common
            )
        case Rank():
            order = sorted(range(count), key=lambda index: -probs[index])
            return RankResult(
                answer=tuple(ids[index] for index in order), probabilities=by_id, **common
            )
        case Rate():
            return RateResult(
                answer=ids[chosen[0]],
                probabilities=by_id,
                prediction_set=None if prediction is None else tuple(ids[i] for i in prediction),
                **common,
            )
        case Estimate():
            return EstimateResult(
                answer=_estimate_value(probs, spec.low, spec.high),
                unit=spec.unit,
                interval=_estimate_interval(prediction, probs.shape[0], spec.low, spec.high),
                **common,
            )
