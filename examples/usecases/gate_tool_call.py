"""Gate a refund tool call against a written rule.

Install `mimirai[local]`, then run `make example NAME=usecases/gate_tool_call`.
"""

from mimir import CheckOutcome, Decider, Mimir, Permission

RULES = ["Refunds above 500 dollars need a manager's approval."]


def gate(decider: Decider, amount: float) -> CheckOutcome:
    """Decide whether a refund call may run: allow, deny, or escalate to a person."""
    check = decider.tool_call_check(RULES, tools=["issue_refund"])
    return check("issue_refund", {"order": "4412", "amount": amount})


def main() -> None:
    model = Mimir.from_pretrained("Mythologic/MIMIR-1")
    for amount in (120.0, 900.0):
        outcome = gate(model, amount)
        if outcome.permission is Permission.ESCALATE:
            print(amount, "escalate:", outcome.reason)
        else:
            print(amount, outcome.permission.value)


if __name__ == "__main__":
    main()
