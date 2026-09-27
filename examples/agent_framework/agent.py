"""A Microsoft Agent Framework agent that routes support tickets with a MIMIR decision tool.

Install `mimirai[local,agent-framework]` and `agent-framework-openai`, set `OPENAI_API_KEY`,
then run `make example NAME=agent_framework/agent`.
"""

import asyncio
from typing import Any, Final

from agent_framework import Agent, BaseChatClient

from mimir import Choice, Decider, Mimir
from mimir.integrations.agent_framework import as_function_tool

MODEL: Final = "gpt-5.5"
TEAMS: Final = {
    "billing": "Billing: payments, refunds and invoices",
    "security": "Security: account access, passwords and fraud",
    "shipping": "Shipping: deliveries, tracking and returns",
}
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


def build_agent(decider: Decider, client: BaseChatClient[Any]) -> Agent:
    """The support agent, answering with `decider`."""
    route_ticket = decider.tool(
        "route_ticket",
        Choice("Which team should handle this ticket?", TEAMS),
        description="Route a customer support ticket to the team that owns it.",
    )
    return Agent(client=client, instructions=INSTRUCTIONS, tools=[as_function_tool(route_ticket)])


async def main() -> None:
    from agent_framework.openai import OpenAIChatClient

    agent = build_agent(Mimir.from_pretrained("vathosai/mimir-1"), OpenAIChatClient(model=MODEL))
    print((await agent.run("I was charged twice for order 4412.")).text)


if __name__ == "__main__":
    asyncio.run(main())
