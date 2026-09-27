"""An OpenAI Agents SDK agent that routes support tickets with a MIMIR decision tool.

Install `mimirai[local,openai-agents]`, set `OPENAI_API_KEY`, then run
`make example NAME=openai_agents/agent`.
"""

import asyncio
from typing import Final

from agents import Agent, Model, Runner

from mimir import Choice, Decider, Mimir
from mimir.integrations.openai_agents import as_function_tool

TEAMS: Final = {
    "billing": "Billing: payments, refunds and invoices",
    "security": "Security: account access, passwords and fraud",
    "shipping": "Shipping: deliveries, tracking and returns",
}
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


def build_agent(decider: Decider, model: str | Model | None = None) -> Agent[None]:
    """The support agent, answering with `decider`; `model` defaults to the SDK's."""
    route_ticket = decider.tool(
        "route_ticket",
        Choice("Which team should handle this ticket?", TEAMS),
        description="Route a customer support ticket to the team that owns it.",
    )
    return Agent(
        name="support",
        instructions=INSTRUCTIONS,
        tools=[as_function_tool(route_ticket)],
        model=model,
    )


async def main() -> None:
    agent = build_agent(Mimir.from_pretrained("mythologic/mimir-1"))
    result = await Runner.run(agent, "I was charged twice for order 4412.")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
