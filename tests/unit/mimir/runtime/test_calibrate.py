from collections.abc import Callable
from pathlib import Path

import pytest

from mimir.core.decisions import Estimate, MultiChoice
from mimir.core.errors import PolicyError
from mimir.core.labels import LabelledDecision
from mimir.core.results import Status
from mimir.policy.document import write_policy
from mimir.runtime.calibrate import calibrate
from tests.conftest import load_engine

# Scripted outputs give each option utility 2 and abstain K / 4, so both options of this spec
# are chosen with probability sigmoid(1.5) and the decision scores sigmoid(1.5)^2 = 0.67.
SPEC = MultiChoice("which apply", ["refund", "order"])


def labelled(count: int, label: list[str]) -> list[LabelledDecision]:
    return [
        LabelledDecision(context=f"card {index}", decision=SPEC, label=label)
        for index in range(count)
    ]


def test_correct_decisions_certify_a_threshold_that_loads_back(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = load_engine(release, monkeypatch)
    found = calibrate(engine, labelled(400, ["refund", "order"]), risk=0.01, confidence=0.95)
    (report,) = found.types
    assert (report.model_type, report.records, report.needed) == ("multilabel", 400, 299)
    assert report.certified.threshold is not None
    assert (report.certified.taken, report.certified.errors) == (400, 0)
    document = found.policy.document
    assert document.origin == "custom"
    assert document.fingerprint.configurations[0].hardware == engine.runtime.hardware
    assert document.decision_types["categorical"].thresholds == {}
    write_policy(found.policy, tmp_path / "custom.json", tmp_path / "custom.npz")
    custom = load_engine(release, monkeypatch, policy=tmp_path / "custom.json")
    result = custom.decide("card", SPEC)
    assert result.status == Status.DECIDED
    assert result.certificate is not None
    assert result.certificate.origin == "custom"


def test_wrong_decisions_certify_nothing(release: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    found = calibrate(
        load_engine(release, monkeypatch), labelled(400, []), risk=0.01, confidence=0.95
    )
    certified = found.types[0].certified
    assert (certified.threshold, certified.taken, certified.coverage) == (None, 0, 0.0)


def test_too_few_decisions_report_how_many_are_needed(
    release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = calibrate(
        load_engine(release, monkeypatch),
        labelled(50, ["refund", "order"]),
        risk=0.01,
        confidence=0.95,
    )
    assert found.types[0].certified.threshold is None
    assert found.types[0].needed == 299


def test_calibration_needs_a_release_policy_and_a_certifiable_type(
    release: Path, release_builder: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    bare = load_engine(release_builder("bare", with_policy=False), monkeypatch)
    with pytest.raises(PolicyError, match="none is loaded"):
        calibrate(bare, labelled(5, []), risk=0.01, confidence=0.95)
    only_continuous = [
        LabelledDecision(context="x", decision=Estimate("q", 0, 1), label=0.5) for _ in range(3)
    ]
    with pytest.raises(PolicyError, match="certifiable"):
        calibrate(load_engine(release, monkeypatch), only_continuous, risk=0.01, confidence=0.95)
    with pytest.raises(ValueError, match="confidence"):
        calibrate(load_engine(release, monkeypatch), labelled(5, []), risk=0.01, confidence=1.0)
