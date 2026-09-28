"""A smolagents `ToolCallingAgent` that routes support tickets with a MIMIR decision tool.

Install `mimir-decisions[local,smolagents]`, set `HF_TOKEN` for the Inference Providers model, then
run `make example NAME=smolagents/agent`.
"""

from typing import Final

from smolagents import InferenceClientModel, Model, ToolCallingAgent

from mimir import Choice, Decider, Mimir
from mimir.integrations.smolagents import as_smolagents_tool

TEAMS: Final = {
    "billing": "Billing: payments, refunds and invoices",
    "security": "Security: account access, passwords and fraud",
    "shipping": "Shipping: deliveries, tracking and returns",
}
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


def build_agent(decider: Decider, model: Model) -> ToolCallingAgent:
    """The support agent, answering with `decider`."""
    route_ticket = decider.tool(
        "route_ticket",
        Choice("Which team should handle this ticket?", TEAMS),
        description="Route a customer support ticket to the team that owns it.",
    )
    return ToolCallingAgent(
        tools=[as_smolagents_tool(route_ticket)], model=model, instructions=INSTRUCTIONS
    )


def main() -> None:
    agent = build_agent(Mimir.from_pretrained("Mythologic/MIMIR-1"), InferenceClientModel())
    print(agent.run("I was charged twice for order 4412."))


if __name__ == "__main__":
    main()
