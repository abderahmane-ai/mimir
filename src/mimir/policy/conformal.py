"""Split-conformal prediction sets at any alpha, from sorted calibration scores.

- Binary and categorical: LAC (Sadinle et al., arXiv 1609.00451), with abstain as a label.
  A label is included when `1 - p(label)` is at most the quantile.
- Ordinal and continuous: intervals grown from the mode towards the larger neighbour
  (arXiv 2207.02238, arXiv 2105.08747). The set is the largest interval whose mass is at most
  the quantile.
"""

import math

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float64]


def conformal_quantile(sorted_scores: Floats, alpha: float) -> float | None:
    """Return the `ceil((n + 1) * (1 - alpha))`-th smallest score.

    Returns None when n is too small for that rank; the set then contains every label.
    """
    rank = math.ceil((sorted_scores.shape[0] + 1) * (1 - alpha))
    return float(sorted_scores[rank - 1]) if rank <= sorted_scores.shape[0] else None


def lac_set(probs: Floats, quantile: float | None) -> tuple[int, ...]:
    """Return the labels with `1 - p <= quantile`, or every label if `quantile` is None."""
    if quantile is None:
        return tuple(range(probs.shape[0]))
    return tuple(int(index) for index in np.flatnonzero(1 - probs <= quantile))


def grown_intervals(probs: Floats) -> list[tuple[int, int, float]]:
    """Return `(left, right, mass)` after each growth step, starting from the mode.

    Each step adds the larger neighbour (the lower one on ties) until all levels are covered.
    """
    width = probs.shape[0]
    left = right = int(probs.argmax())
    mass = float(probs[left])
    steps = [(left, right, mass)]
    for _ in range(width - 1):
        below = float(probs[left - 1]) if left > 0 else -1.0
        above = float(probs[right + 1]) if right < width - 1 else -1.0
        if below >= above and below >= 0:
            left -= 1
            mass += below
        elif above >= 0:
            right += 1
            mass += above
        steps.append((left, right, mass))
    return steps


def interval_set(probs: Floats, quantile: float | None) -> tuple[int, int] | None:
    """Return the largest grown interval with mass at most `quantile`.

    Returns every level if `quantile` is None, and None (an empty set) if the mode alone
    exceeds it.
    """
    if quantile is None:
        return (0, probs.shape[0] - 1)
    within = [(left, right) for left, right, mass in grown_intervals(probs) if mass <= quantile]
    return within[-1] if within else None
