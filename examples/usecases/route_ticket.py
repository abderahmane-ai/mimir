"""Route a support ticket to the team that owns it.

Install `mimir-decisions[local]`, then run `make example NAME=usecases/route_ticket`.
"""

from mimir import ChoiceResult, Decider, Mimir, Status

TEAMS = {
    "billing": "Billing: invoices, payments, refunds",
    "technical": "Technical: bugs, outages, system errors",
    "sales": "Sales: pricing, new contracts",
    "other": "Other: everything else",
}


def route(decider: Decider, ticket: str) -> ChoiceResult:
    """Route one ticket; a deferred ticket goes to a person, never to a guessed team."""
    return decider.choose(
        ticket, "Which department should handle this request?", options=TEAMS, risk=0.01
    )


def main() -> None:
    model = Mimir.from_pretrained("Mythologic/MIMIR-1")
    result = route(
        model,
        "Hi, we were billed twice for March. Please refund the duplicate today or we will "
        "cancel our plan.",
    )
    if result.status is not Status.DECIDED:
        print("deferred: a person reviews the ticket")
    else:
        print(result.answer, round(result.confidence or 0.0, 3))


if __name__ == "__main__":
    main()
