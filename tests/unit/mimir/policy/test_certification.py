import math

import numpy as np
import pytest

from mimir.policy.certification import (
    LAMBDA_GRID,
    binomial_p_value,
    certify_threshold,
    fewest_trials,
    fixed_sequence,
    taken_and_wrong,
)


def test_binomial_p_value_matches_the_exact_sum() -> None:
    expected = sum(math.comb(100, k) * 0.05**k * 0.95 ** (100 - k) for k in range(3))
    assert binomial_p_value(2, 100, 0.05) == pytest.approx(expected, rel=1e-12)
    assert binomial_p_value(0, 0, 0.01) == 1.0
    assert binomial_p_value(5, 5, 0.01) == 1.0


def test_fewest_trials_is_where_zero_errors_first_pass() -> None:
    needed = fewest_trials(0.01, 0.05)
    assert needed == 299
    assert binomial_p_value(0, needed, 0.01) <= 0.05 < binomial_p_value(0, needed - 1, 0.01)


def test_taken_and_wrong_counts_decisions_at_each_threshold() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, False, True, True])
    taken, wrong = taken_and_wrong(score, correct, np.array([0.95, 0.8, 0.65, 0.5]))
    assert taken == [0, 2, 3, 4]
    assert wrong == [0, 1, 1, 1]


def test_fixed_sequence_stops_at_the_first_failure() -> None:
    tested, last = fixed_sequence([400, 500, 600, 700], [0, 1, 20, 0], 0.01, 0.05)
    assert [item.rejected for item in tested] == [True, True, False]
    assert last == 1


def test_fixed_sequence_with_an_immediate_failure_certifies_nothing() -> None:
    tested, last = fixed_sequence([10], [5], 0.01, 0.05)
    assert last is None
    assert len(tested) == 1


def test_grid_descends_from_near_one_to_one_half() -> None:
    assert LAMBDA_GRID[0] == pytest.approx(1 - 1e-4)
    assert LAMBDA_GRID[-1] == pytest.approx(0.5)
    assert np.all(np.diff(LAMBDA_GRID) < 0)


def test_certify_threshold_on_clean_decisions_takes_them_all() -> None:
    rng = np.random.default_rng(7)
    score = rng.uniform(0.6, 1.0, size=2000)
    correct = np.ones(2000, dtype=np.bool_)
    found = certify_threshold(score, correct, 0.01, 0.05)
    assert found.threshold == pytest.approx(LAMBDA_GRID[-1])
    assert (found.taken, found.errors, found.records) == (2000, 0, 2000)
    assert found.p_value <= 0.05


def test_certify_threshold_needs_enough_decisions() -> None:
    found = certify_threshold(np.full(100, 0.99), np.ones(100, dtype=np.bool_), 0.01, 0.05)
    assert found.threshold is None
    assert (found.taken, found.errors, found.p_value) == (0, 0, 1.0)


def test_certify_threshold_stops_before_errors_accumulate() -> None:
    score = np.concatenate([np.linspace(0.999, 0.9, 1000), np.linspace(0.89, 0.5, 1000)])
    correct = np.concatenate([np.ones(1000, dtype=np.bool_), np.zeros(1000, dtype=np.bool_)])
    found = certify_threshold(score, correct, 0.01, 0.05)
    assert found.threshold is not None
    assert (found.taken, found.errors) == (1004, 4)
    assert found.p_value <= 0.05
    assert not found.tested[-1].rejected
    assert found.tested[-1].errors > found.errors
