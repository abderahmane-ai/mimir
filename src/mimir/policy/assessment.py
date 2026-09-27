"""Applying a policy to one request's graph outputs."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mimir.policy.conformal import conformal_quantile, interval_set, lac_set
from mimir.policy.distributions import (
    CHOICE_TYPES,
    Decision,
    Readout,
    decide,
    probabilities,
)
from mimir.policy.document import CertifiedThreshold, Policy, risk_key
from mimir.policy.gate import nearest_distance, p_value

Floats = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Assessment:
    """Policy outcome for one request at a given risk level and alpha.

    Attributes:
        certificate: The policy entry for this decision type and risk, or None.
        taken: The entry has a threshold, the gate passes and the score reaches the threshold.
        prediction_set: Label indices in the conformal set (abstain is index K for binary and
            categorical; intervals list every level or bin they cover), or None if the policy
            has no conformal scores for this type.
    """

    probabilities: Floats
    decision: Decision
    gate_p_value: float
    passes_gate: bool
    certificate: CertifiedThreshold | None
    taken: bool
    prediction_set: tuple[int, ...] | None


def assess(readout: Readout, policy: Policy, risk: float, alpha: float) -> Assessment:
    kind = readout.model_type
    type_policy = policy.type_policy(kind)
    scaling = type_policy.scaling_of(readout.count)
    probs = probabilities(readout, scaling.parameters)
    decision = decide(kind, probs)
    arrays = policy.arrays
    distance = nearest_distance(
        readout.workspace.astype(np.float64),
        arrays[f"gate/{kind}/centroids"],
        arrays[f"gate/{kind}/precision"],
    )
    gate = p_value(arrays[f"gate/{kind}/reference"], distance)
    passes_gate = gate > policy.document.gate_level
    certificate = type_policy.thresholds.get(risk_key(risk))
    threshold = None if certificate is None else certificate.threshold
    taken = (
        threshold is not None
        and passes_gate
        and decision.score is not None
        and decision.score >= threshold
    )
    scores = policy.conformal_scores(kind, scaling.bucket)
    prediction_set: tuple[int, ...] | None = None
    if scores is not None:
        quantile = conformal_quantile(scores, alpha)
        if kind in CHOICE_TYPES:
            prediction_set = lac_set(probs, quantile)
        else:
            bounds = interval_set(probs, quantile)
            prediction_set = () if bounds is None else tuple(range(bounds[0], bounds[1] + 1))
    return Assessment(
        probabilities=probs,
        decision=decision,
        gate_p_value=gate,
        passes_gate=passes_gate,
        certificate=certificate,
        taken=taken,
        prediction_set=prediction_set,
    )
