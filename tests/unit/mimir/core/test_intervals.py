import pytest

from mimir.core.intervals import wilson_interval


def test_no_trials_has_no_interval() -> None:
    assert wilson_interval(0, 0) is None


def test_known_value() -> None:
    # 5 successes in 10 trials: centre 0.5, half-width 0.2837 (Wilson 1927).
    interval = wilson_interval(5, 10)
    assert interval is not None
    assert interval[0] == pytest.approx(0.2366, abs=1e-4)
    assert interval[1] == pytest.approx(0.7634, abs=1e-4)


@pytest.mark.parametrize(("successes", "trials"), [(0, 1), (0, 1000), (1000, 1000), (3, 7)])
def test_interval_contains_the_rate_and_stays_in_unit_range(successes: int, trials: int) -> None:
    interval = wilson_interval(successes, trials)
    assert interval is not None
    low, high = interval
    assert 0.0 <= low <= successes / trials <= high <= 1.0


def test_interval_narrows_with_more_trials() -> None:
    small, large = wilson_interval(5, 10), wilson_interval(500, 1000)
    assert small is not None
    assert large is not None
    assert large[1] - large[0] < small[1] - small[0]
