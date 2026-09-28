"""Estimate a repair cost inside a range, with an interval.

Install `mimir-decisions[local]`, then run `make example NAME=usecases/estimate_cost`.
"""

from mimir import Decider, EstimateResult, Mimir


def estimate(decider: Decider, description: str) -> EstimateResult:
    """Estimate one repair in dollars; the interval covers the honest range."""
    return decider.estimate(
        description, "What will the repair cost?", 0.0, 5000.0, unit="USD", risk=0.01
    )


def main() -> None:
    model = Mimir.from_pretrained("Mythologic/MIMIR-1")
    result = estimate(model, "Replace a cracked phone screen, parts and labour.")
    print(result.status, round(result.answer, 2), result.interval)


if __name__ == "__main__":
    main()
