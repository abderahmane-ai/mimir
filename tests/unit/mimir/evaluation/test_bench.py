import pytest

from mimir.core.decisions import Choice, Estimate, YesNo
from mimir.core.labels import Label
from mimir.core.results import DecisionResult, Status
from mimir.evaluation.bench import bench, rate
from tests.conftest import result_for


def test_rates_accuracy_coverage_and_risk_per_type() -> None:
    choice = Choice("q", ["a", "b"])
    results: list[DecisionResult] = [
        result_for(choice, Status.DECIDED),
        result_for(choice, Status.DECIDED),
        result_for(choice, Status.DEFERRED),
        result_for(choice, Status.DEFERRED),
        result_for(YesNo("q"), Status.DECIDED),
    ]
    labels: list[Label] = ["a", "b", "a", "b", False]
    report = bench(results, labels)
    assert report.records == 5
    choice_report = report.types["choice"]
    assert (choice_report.accuracy.count, choice_report.accuracy.total) == (2, 4)
    assert (choice_report.coverage.count, choice_report.coverage.total) == (2, 4)
    assert (choice_report.risk.count, choice_report.risk.total) == (1, 2)
    assert choice_report.risk.value == 0.5
    assert choice_report.mean_absolute_error is None
    assert report.types["yes_no"].accuracy.value == 1.0


def test_estimates_report_mean_absolute_error() -> None:
    spec = Estimate("q", 0.0, 10.0)
    report = bench([result_for(spec), result_for(spec)], [4.0, 7.0])
    estimate = report.types["estimate"]
    assert estimate.mean_absolute_error == pytest.approx(1.5)
    assert estimate.accuracy.value == 1.0


def test_no_decision_taken_gives_no_risk_value() -> None:
    report = bench([result_for(YesNo("q"), Status.DEFERRED)], [True])
    assert report.types["yes_no"].risk.value is None
    assert report.types["yes_no"].risk.interval is None


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="1 results for 2 labels"):
        bench([result_for(YesNo("q"))], [True, False])


def test_rate_has_a_wilson_interval() -> None:
    found = rate(5, 10)
    assert found.value == 0.5
    assert found.interval is not None
    assert found.interval[0] < 0.5 < found.interval[1]
