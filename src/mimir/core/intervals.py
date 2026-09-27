"""Confidence intervals for rates."""

import math
from typing import Final

WILSON_Z: Final = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z) -> tuple[float, float] | None:
    """Return the Wilson score interval (95% by default), or None when `trials` is 0."""
    if trials == 0:
        return None
    rate = successes / trials
    centre = (rate + z**2 / (2 * trials)) / (1 + z**2 / trials)
    spread = (
        z / (1 + z**2 / trials) * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2))
    )
    # The bounds are exactly 0 and 1 at the extremes; rounding would leave them near 1e-19 off.
    low = 0.0 if successes == 0 else max(0.0, centre - spread)
    high = 1.0 if successes == trials else min(1.0, centre + spread)
    return (low, high)
