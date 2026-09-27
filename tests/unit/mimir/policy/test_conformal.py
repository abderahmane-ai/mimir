import numpy as np
import pytest

from mimir.policy.conformal import conformal_quantile, grown_intervals, interval_set, lac_set


def test_quantile_takes_the_finite_sample_rank() -> None:
    scores = np.arange(1, 20, dtype=np.float64)
    # n = 19, alpha = 0.1: rank ceil(20 * 0.9) = 18.
    assert conformal_quantile(scores, 0.1) == 18.0
    assert conformal_quantile(scores, 0.5) == 10.0


def test_quantile_is_none_when_too_few_scores() -> None:
    assert conformal_quantile(np.array([0.1, 0.2]), 0.1) is None
    assert conformal_quantile(np.array([]), 0.5) is None


def test_lac_set_includes_labels_within_the_quantile() -> None:
    probs = np.array([0.6, 0.3, 0.1])
    assert lac_set(probs, 0.75) == (0, 1)
    assert lac_set(probs, 0.4) == (0,)
    assert lac_set(probs, 0.0) == ()
    assert lac_set(probs, None) == (0, 1, 2)


def test_grown_intervals_take_the_heavier_neighbour_and_the_lower_on_ties() -> None:
    steps = grown_intervals(np.array([0.1, 0.2, 0.4, 0.2, 0.1]))
    assert [(left, right) for left, right, _ in steps] == [
        (2, 2),
        (1, 2),
        (1, 3),
        (0, 3),
        (0, 4),
    ]
    assert [mass for _, _, mass in steps] == pytest.approx([0.4, 0.6, 0.8, 0.9, 1.0])


def test_interval_set() -> None:
    probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    assert interval_set(probs, 0.85) == (1, 3)
    assert interval_set(probs, None) == (0, 4)
    assert interval_set(probs, 0.3) is None


def test_interval_at_the_edge_grows_inwards() -> None:
    steps = grown_intervals(np.array([0.7, 0.2, 0.1]))
    assert [(left, right) for left, right, _ in steps] == [(0, 0), (0, 1), (0, 2)]
