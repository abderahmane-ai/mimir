"""Rank vendors against a requirement, best first.

Install `mimir-decisions[local]`, then run `make example NAME=usecases/rank_vendors`.
"""

from mimir import Decider, Mimir, RankResult

VENDORS = {
    "acme": "Acme: SOC 2, EU data residency, 99.99% SLA",
    "globex": "Globex: no certifications, US-only hosting, 99.9% SLA",
    "initech": "Initech: SOC 2, EU data residency, 99.95% SLA",
}


def rank(decider: Decider) -> RankResult:
    """Rank the vendors for a compliant EU deployment."""
    return decider.rank(
        "EU data residency with SOC 2 and the strongest SLA.",
        "Which vendor fits best?",
        candidates=VENDORS,
        risk=0.01,
    )


def main() -> None:
    model = Mimir.from_pretrained("Mythologic/MIMIR-1")
    result = rank(model)
    print(result.status, list(result.answer))


if __name__ == "__main__":
    main()
