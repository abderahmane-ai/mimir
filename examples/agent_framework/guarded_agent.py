"""A Microsoft Agent Framework agent whose refunds MIMIR checks against the refund rules
first.

A certified yes runs the refund; otherwise the refund does not run and the model reads the
check's reason. Install `mimirai[local,agent-framework]` and `agent-framework-openai`, set
`OPENAI_API_KEY`, then run `make example NAME=agent_framework/guarded_agent`.
"""

import asyncio
from typing import Any, Final

from agent_framework import Agent, BaseChatClient, FunctionTool

from mimir import Decider, Mimir
from mimir.integrations.agent_framework import ToolCallCheckMiddleware

MODEL: Final = "gpt-5.5"
RULES: Final = (
    "A refund is at most the amount the customer paid for the order.",
    "Refunds above 500 dollars need a manager's approval.",
)
INSTRUCTIONS: Final = "Issue the refunds customers ask for with issue_refund."


def issue_refund(order: str, amount: float) -> str:
    """Refund `amount` dollars on `order`."""
    return f"Refunded {amount:.2f} dollars on order {order}."


def build_agent(
    decider: Decider, client: BaseChatClient[Any], refund: FunctionTool | None = None
) -> Agent:
    """The refunds agent, its refund tool checked by `decider` against `RULES`."""
    tool = refund or FunctionTool(
        name="issue_refund", description="Refund the customer.", func=issue_refund
    )
    check = decider.tool_call_check(RULES, tools=[tool.name])
    return Agent(
        client=client,
        instructions=INSTRUCTIONS,
        tools=[tool],
        middleware=[ToolCallCheckMiddleware(check)],
    )


async def main() -> None:
    from agent_framework.openai import OpenAIChatClient

    agent = build_agent(Mimir.from_pretrained("vathosai/mimir-1"), OpenAIChatClient(model=MODEL))
    print((await agent.run("Refund 900 dollars on order 4412.")).text)


if __name__ == "__main__":
    asyncio.run(main())
