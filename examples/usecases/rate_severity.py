"""Rate the severity of an incident on an ordered scale.

Install `mimir-decisions[local]`, then run `make example NAME=usecases/rate_severity`.
"""

from mimir import Decider, Mimir, RateResult

LEVELS = ["low", "medium", "high", "critical"]


def rate(decider: Decider, report: str) -> RateResult:
    """Rate one incident report; levels run lowest first."""
    return decider.rate(report, "How severe is this incident?", levels=LEVELS, risk=0.01)


def main() -> None:
    model = Mimir.from_pretrained("Mythologic/MIMIR-1")
    result = rate(model, "Checkout has been down for all EU customers since 09:14.")
    print(result.status, result.answer)


if __name__ == "__main__":
    main()
