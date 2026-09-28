"""Route a support ticket to the team that owns it.

Install `mimirai[local]`, then run `make example NAME=usecases/route_ticket`.
"""

from mimir import ChoiceResult, Decider, Mimir, Status

TEAMS = {
    "billing": "Billing: payments, refunds and invoices",
    "security": "Security: account access, passwords and fraud",
    "shipping": "Shipping: deliveries, tracking and returns",
}


def route(decider: Decider, ticket: str) -> ChoiceResult:
    """Route one ticket; a deferred ticket goes to a person, never to a guessed team."""
    return decider.choose(ticket, "Which team should handle this ticket?", options=TEAMS, risk=0.01)


def main() -> None:
    model = Mimir.from_pretrained("Mythologic/MIMIR-1")
    result = route(model, "I was charged twice for order 4412.")
    if result.status is not Status.DECIDED:
        print("deferred: a person reviews the ticket")
    else:
        print(result.answer, round(result.confidence or 0.0, 3))


if __name__ == "__main__":
    main()
