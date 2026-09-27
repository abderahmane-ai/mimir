"""Certification of selective thresholds by fixed-sequence testing.

Candidate thresholds run from `1 - 1e-4` down to `0.5`, evenly spaced in `log(1 - lambda)`.
Each is tested at level `delta` with an exact binomial test on the error rate of the decisions
it accepts, stopping at the first failure (Learn then Test, arXiv 2110.01052, section 3.2).
The certified threshold is the last one that passed. Testing starts at the first threshold that
accepts enough decisions for zero errors to pass; that choice uses counts only, not labels.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float64]
Bools = npt.NDArray[np.bool_]

LAMBDA_GRID: Final = 1 - np.geomspace(1e-4, 0.5, 1000)


@dataclass(frozen=True, slots=True)
class Tested:
    trials: int
    errors: int
    p_value: float
    rejected: bool


@dataclass(frozen=True, slots=True)
class Certified:
    """Outcome of certification.

    `threshold` is None when no candidate passed. `taken`, `errors` and `p_value` describe the
    certified threshold on the calibration records.
    """

    threshold: float | None
    records: int
    taken: int
    errors: int
    p_value: float
    tested: tuple[Tested, ...]


def binomial_p_value(errors: int, trials: int, rate: float) -> float:
    """Return `P(Binomial(trials, rate) <= errors)`."""
    if trials == 0 or errors >= trials:
        return 1.0
    steps = np.arange(errors, dtype=np.float64)
    log_ratios = np.log((trials - steps) / (steps + 1)) + math.log(rate / (1 - rate))
    log_mass = trials * math.log1p(-rate) + np.concatenate([[0.0], np.cumsum(log_ratios)])
    top = float(log_mass.max())
    return min(1.0, math.exp(top) * float(np.exp(log_mass - top).sum()))


def fewest_trials(rate: float, delta: float) -> int:
    """Return the fewest decisions for which zero errors pass the test at `rate` and `delta`."""
    return math.ceil(math.log(delta) / math.log1p(-rate))


def taken_and_wrong(score: Floats, correct: Bools, grid: Floats) -> tuple[list[int], list[int]]:
    """Return the accepted decisions and their errors at each threshold in `grid`."""
    order = np.argsort(-score, kind="stable")
    wrong = np.concatenate([[0], np.cumsum(~correct[order])])
    taken = np.searchsorted(-score[order], -grid, side="right")
    return taken.tolist(), wrong[taken].tolist()


def fixed_sequence(
    trials: Sequence[int], errors: Sequence[int], rate: float, delta: float
) -> tuple[tuple[Tested, ...], int | None]:
    """Test hypotheses in order; return the tests run and the index of the last rejection."""
    tested: list[Tested] = []
    last: int | None = None
    for index, (count, wrong) in enumerate(zip(trials, errors, strict=True)):
        found = binomial_p_value(int(wrong), int(count), rate)
        rejected = found <= delta
        tested.append(Tested(int(count), int(wrong), found, rejected))
        if not rejected:
            break
        last = index
    return tuple(tested), last


def certify_threshold(score: Floats, correct: Bools, rate: float, delta: float) -> Certified:
    """Certify a threshold whose accepted decisions have error rate at most `rate`, with
    probability at least `1 - delta`."""
    taken, wrong = taken_and_wrong(score, correct, LAMBDA_GRID)
    needed = fewest_trials(rate, delta)
    start = next((index for index, count in enumerate(taken) if count >= needed), len(taken))
    tested, last = fixed_sequence(taken[start:], wrong[start:], rate, delta)
    if last is None:
        return Certified(None, int(score.shape[0]), 0, 0, 1.0, tested)
    chosen = tested[last]
    return Certified(
        threshold=float(LAMBDA_GRID[start + last]),
        records=int(score.shape[0]),
        taken=chosen.trials,
        errors=chosen.errors,
        p_value=chosen.p_value,
        tested=tested,
    )
