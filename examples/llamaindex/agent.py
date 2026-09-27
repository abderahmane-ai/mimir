"""A LlamaIndex `FunctionAgent` that routes support tickets with a MIMIR decision tool.

The typed result is each tool output's `raw_output`. Install `mimirai[local,llamaindex]` and
`llama-index-llms-openai`, set `OPENAI_API_KEY`, then run `make example NAME=llamaindex/agent`.
"""

import asyncio
from typing import Final

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM

from mimir import Choice, Decider, Mimir
from mimir.integrations.llamaindex import as_llamaindex_tool

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


def build_agent(decider: Decider, llm: FunctionCallingLLM) -> FunctionAgent:
    """The support agent, answering with `decider`."""
    route_ticket = decider.tool(
        "route_ticket",
        Choice("Which team should handle this ticket?", TEAMS),
        description="Route a customer support ticket to the team that owns it.",
    )
    return FunctionAgent(
        tools=[as_llamaindex_tool(route_ticket)], llm=llm, system_prompt=INSTRUCTIONS
    )


async def main() -> None:
    from llama_index.llms.openai import OpenAI

    agent = build_agent(Mimir.from_pretrained("vathosai/mimir-1"), OpenAI(model=MODEL))
    print(await agent.run(user_msg="I was charged twice for order 4412."))


if __name__ == "__main__":
    asyncio.run(main())
