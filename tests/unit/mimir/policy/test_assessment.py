import numpy as np
import pytest

from mimir.core.decisions import ModelType
from mimir.policy.assessment import assess
from mimir.policy.distributions import Readout
from mimir.policy.document import Configuration, Fingerprint
from mimir.runtime.session import torch_version
from tests.conftest import build_policy

FINGERPRINT = Fingerprint(
    model="m",
    revision="r",
    variant="fp32",
    weights_sha256="b" * 64,
    torch=torch_version(),
    configurations=(Configuration(device="cpu", hardware="h"),),
)


def readout(kind: ModelType, utilities: list[float]) -> Readout:
    return Readout(
        model_type=kind,
        utilities=np.array(utilities),
        thresholds=np.linspace(-1.0, 1.0, max(len(utilities) - 1, 0)),
        abstain=-5.0,
        ordinal_score=0.0,
        histogram=np.zeros(8),
        workspace=np.zeros(4),
    )


def test_a_confident_decision_is_certified() -> None:
    found = assess(readout("categorical", [5.0, 0.0, 0.0]), build_policy(FINGERPRINT), 0.01, 0.1)
    assert found.decision.chosen == (0,)
    assert found.decision.score is not None
    assert found.decision.score > 0.5
    assert found.certified
    assert found.certificate is not None


def test_a_low_score_is_not_certified() -> None:
    found = assess(readout("categorical", [0.0, 0.0, 0.0]), build_policy(FINGERPRINT), 0.01, 0.1)
    assert not found.certified


def test_no_threshold_at_the_risk_means_no_certificate() -> None:
    policy = build_policy(FINGERPRINT)
    assert assess(readout("categorical", [5.0, 0.0]), policy, 0.05, 0.1).certificate is None
    uncertified = build_policy(FINGERPRINT, threshold=None)
    found = assess(readout("categorical", [5.0, 0.0]), uncertified, 0.01, 0.1)
    assert found.certificate is not None
    assert found.certificate.threshold is None
    assert not found.certified


def test_prediction_sets_per_type() -> None:
    policy = build_policy(FINGERPRINT)
    choice = assess(readout("categorical", [5.0, 0.0, 0.0]), policy, 0.01, 0.5)
    assert choice.prediction_set == (0,)
    ordinal = assess(readout("ordinal", [0.0, 0.0, 0.0]), policy, 0.01, 0.1)
    assert ordinal.prediction_set is not None
    assert list(ordinal.prediction_set) == list(
        range(ordinal.prediction_set[0], ordinal.prediction_set[-1] + 1)
    )
    ranking = assess(readout("ranking", [1.0, 0.0]), policy, 0.01, 0.1)
    assert ranking.prediction_set is None


def test_continuous_is_never_certified() -> None:
    found = assess(readout("continuous", []), build_policy(FINGERPRINT), 0.01, 0.1)
    assert found.decision.score is None
    assert not found.certified
    assert found.probabilities.sum() == pytest.approx(1.0)
