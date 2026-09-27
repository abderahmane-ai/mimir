"""A PydanticAI agent that routes support tickets with MIMIR decision tools.

The tools return typed results, which PydanticAI keeps in the tool return parts. Install
`mimirai[local,pydantic-ai]` and `pydantic-ai-slim[openai]`, set `OPENAI_API_KEY`, then run
`make example NAME=pydantic_ai/agent`.
"""

import asyncio
from typing import Final

from pydantic_ai import Agent
from pydantic_ai.models import Model

from mimir import Choice, Decider, Mimir
from mimir.integrations.pydantic_ai import as_toolset

MODEL: Final = "openai:gpt-5.5"
TEAMS: Final = {
    "billing": "Billing: payments, refunds and invoices",
    "security": "Security: account access, passwords and fraud",
    "shipping": "Shipping: deliveries, tracking and returns",
}
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


def build_agent(decider: Decider, model: str | Model = MODEL) -> Agent[None, str]:
    """The support agent, answering with `decider`."""
    route_ticket = decider.tool(
        "route_ticket",
        Choice("Which team should handle this ticket?", TEAMS),
        description="Route a customer support ticket to the team that owns it.",
    )
    return Agent(model, instructions=INSTRUCTIONS, toolsets=[as_toolset([route_ticket])])


async def main() -> None:
    agent = build_agent(Mimir.from_pretrained("mythologic/mimir-1"))
    result = await agent.run("I was charged twice for order 4412.")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
