import math

import numpy as np
import pytest

from mimir.core.decisions import ModelType
from mimir.policy.distributions import (
    Readout,
    decide,
    identity,
    option_count_bucket,
    probabilities,
    sigmoid,
    softmax,
)


def readout(
    kind: ModelType,
    utilities: list[float],
    *,
    thresholds: list[float] | None = None,
    abstain: float = 0.0,
    ordinal_score: float = 0.0,
    histogram: list[float] | None = None,
) -> Readout:
    return Readout(
        model_type=kind,
        utilities=np.array(utilities, dtype=np.float64),
        thresholds=np.array(thresholds or [], dtype=np.float64),
        abstain=abstain,
        ordinal_score=ordinal_score,
        histogram=np.array(histogram or [0.0] * 4, dtype=np.float64),
        workspace=np.zeros(4),
    )


def test_choice_softmax_includes_abstain_last_and_scales_by_temperature() -> None:
    found = probabilities(readout("categorical", [1.0, 2.0, 0.0], abstain=0.5), (2.0,))
    expected = softmax(np.array([0.5, 1.0, 0.0, 0.25]))
    np.testing.assert_allclose(found, expected, rtol=0, atol=1e-15)
    assert found.sum() == pytest.approx(1.0, abs=1e-15)


def test_ranking_is_a_softmax_over_candidates_only() -> None:
    found = probabilities(readout("ranking", [0.0, math.log(3.0)]), (1.0,))
    np.testing.assert_allclose(found, [0.25, 0.75], atol=1e-15)


def test_ordinal_levels_come_from_cumulative_sigmoids() -> None:
    found = probabilities(
        readout("ordinal", [0.0, 0.0, 0.0], thresholds=[-1.0, 1.0], ordinal_score=0.0), (1.0,)
    )
    below_first = 1 / (1 + math.e)
    np.testing.assert_allclose(
        found, [below_first, 1 - 2 * below_first, below_first], rtol=0, atol=1e-15
    )
    assert found.sum() == pytest.approx(1.0, abs=1e-15)


def test_multilabel_is_an_independent_platt_sigmoid_per_option() -> None:
    found = probabilities(readout("multilabel", [2.0, -1.0], abstain=1.0), (2.0, 0.5))
    np.testing.assert_allclose(found, sigmoid(np.array([2.5, -3.5])), atol=1e-15)


def test_continuous_is_a_softmax_over_bins() -> None:
    found = probabilities(
        readout("continuous", [], histogram=[0.0, 0.0, 0.0, math.log(5.0)]), (1.0,)
    )
    np.testing.assert_allclose(found, [0.125, 0.125, 0.125, 0.625], atol=1e-15)


def test_identity_scaling_leaves_the_distribution_unchanged() -> None:
    item = readout("multilabel", [0.3, -0.7], abstain=0.1)
    np.testing.assert_array_equal(
        probabilities(item, identity("multilabel")), sigmoid(np.array([0.3, -0.7]) - 0.1)
    )
    assert identity("binary") == (1.0,)


def test_decide_per_type() -> None:
    abstained = decide("categorical", np.array([0.2, 0.1, 0.7]))
    assert (abstained.chosen, abstained.score) == ((), 0.7)
    categorical = decide("categorical", np.array([0.7, 0.2, 0.1]))
    assert (categorical.chosen, categorical.score) == ((0,), 0.7)
    multilabel = decide("multilabel", np.array([0.9, 0.2, 0.5]))
    assert multilabel.chosen == (0,)
    assert multilabel.score == pytest.approx(0.9 * 0.8 * 0.5)
    ordinal = decide("ordinal", np.array([0.1, 0.6, 0.3]))
    assert (ordinal.chosen, ordinal.score) == ((1,), 0.6)
    continuous = decide("continuous", np.array([0.1, 0.9]))
    assert (continuous.chosen, continuous.score) == ((1,), None)


def test_decide_breaks_ties_towards_the_first_option() -> None:
    assert decide("ranking", np.array([0.5, 0.5])).chosen == (0,)


@pytest.mark.parametrize(
    ("count", "bucket"), [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4), (150, 8)]
)
def test_option_count_bucket_is_ceil_log2(count: int, bucket: int) -> None:
    assert option_count_bucket(count) == bucket
    if count > 1:
        assert bucket == math.ceil(math.log2(count))


def test_softmax_is_stable_for_large_logits() -> None:
    found = softmax(np.array([1000.0, 1000.0]))
    np.testing.assert_array_equal(found, [0.5, 0.5])
