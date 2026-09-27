"""Custom policies certified on labelled decisions (`mimir calibrate`).

A custom policy copies the release policy's calibration, gate and conformal scores and replaces
its thresholds with ones certified on the caller's data at a single risk level. Each decision
type is one test, and the tests split `1 - confidence` equally (Bonferroni). The fingerprint is
set to the local model, runtime and hardware, so the policy loads only in that configuration.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mimir.core.decisions import ModelType
from mimir.core.errors import PolicyError
from mimir.core.intervals import wilson_interval
from mimir.core.labels import LabelledDecision, is_correct
from mimir.policy.certification import certify_threshold, fewest_trials
from mimir.policy.document import (
    CertifiedThreshold,
    Configuration,
    Fingerprint,
    Policy,
    PolicyDocument,
    risk_key,
)
from mimir.runtime.engine import Mimir

CERTIFIABLE: tuple[ModelType, ...] = ("binary", "categorical", "multilabel", "ranking", "ordinal")


@dataclass(frozen=True, slots=True)
class TypeCalibration:
    """Certification outcome for one model decision type.

    `needed` is the fewest accepted decisions that could certify the risk with zero errors.
    """

    model_type: ModelType
    records: int
    passed_gate: int
    certified: CertifiedThreshold
    needed: int


@dataclass(frozen=True, slots=True)
class Calibration:
    policy: Policy
    types: tuple[TypeCalibration, ...]


def _fingerprint(engine: Mimir) -> Fingerprint:
    runtime = engine.runtime
    snapshot = engine.snapshot
    return Fingerprint(
        model=snapshot.model,
        revision=snapshot.revision,
        variant=runtime.variant,
        graph_sha256=runtime.graph_sha256,
        weights_sha256=runtime.weights_sha256,
        opset=runtime.opset,
        onnxruntime=runtime.onnxruntime,
        configurations=(
            Configuration(
                provider=runtime.provider,
                options_sha256=runtime.options_sha256,
                hardware=runtime.hardware,
            ),
        ),
    )


def calibrate(
    engine: Mimir,
    labelled: Sequence[LabelledDecision],
    *,
    risk: float,
    confidence: float,
    batch_size: int | None = None,
) -> Calibration:
    """Certify thresholds on `labelled` and return the custom policy with a per-type report.

    Raises:
        PolicyError: the engine has no release policy to start from, or no labelled decision
            is of a certifiable type.
    """
    base = engine.policy
    if base is None:
        message = "calibration starts from the release policy, and none is loaded"
        raise PolicyError(message)
    if not (0 < risk < 1 and 0 < confidence < 1):
        message = f"risk {risk} and confidence {confidence} must both be in (0, 1)"
        raise ValueError(message)
    evaluations = engine.assess_many(
        [(item.context, item.decision) for item in labelled], batch_size=batch_size
    )
    present = [
        kind for kind in CERTIFIABLE if any(item.decision.model_type == kind for item in labelled)
    ]
    if not present:
        message = "no labelled decision is of a certifiable type (continuous cannot be certified)"
        raise PolicyError(message)
    delta = (1 - confidence) / len(present)
    reports: list[TypeCalibration] = []
    thresholds: dict[ModelType, dict[str, CertifiedThreshold]] = {}
    for kind in present:
        rows = [
            (evaluation, item)
            for evaluation, item in zip(evaluations, labelled, strict=True)
            if item.decision.model_type == kind
        ]
        scores: list[float] = []
        outcomes: list[bool] = []
        for evaluation, item in rows:
            assessment = evaluation.assessment
            if assessment is None or not assessment.passes_gate:
                continue
            if assessment.decision.score is None:
                message = f"{kind} decision without a score"
                raise PolicyError(message)
            scores.append(assessment.decision.score)
            outcomes.append(is_correct(evaluation.result, item.label))
        found = certify_threshold(
            np.array(scores, dtype=np.float64), np.array(outcomes, dtype=np.bool_), risk, delta
        )
        entry = CertifiedThreshold(
            threshold=found.threshold,
            records=len(rows),
            taken=found.taken,
            errors=found.errors,
            p_value=found.p_value,
            coverage=found.taken / len(rows),
            coverage_interval=wilson_interval(found.taken, len(rows)),
        )
        thresholds[kind] = {risk_key(risk): entry}
        reports.append(
            TypeCalibration(
                model_type=kind,
                records=len(rows),
                passed_gate=len(scores),
                certified=entry,
                needed=fewest_trials(risk, delta),
            )
        )
    source = base.document
    document = PolicyDocument(
        format_version=source.format_version,
        origin="custom",
        fingerprint=_fingerprint(engine),
        confidence=confidence,
        gate_level=source.gate_level,
        label_taken=source.label_taken,
        decision_types={
            kind: policy.model_copy(update={"thresholds": thresholds.get(kind, {})})
            for kind, policy in source.decision_types.items()
        },
    )
    return Calibration(Policy(document, base.arrays), tuple(reports))
