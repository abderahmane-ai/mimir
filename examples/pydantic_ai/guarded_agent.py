"""A PydanticAI agent whose refunds MIMIR checks against the refund rules first.

A certified yes runs the refund, a certified no fails it with the reason, and anything else
ends the run with deferred tool requests until a person at the console approves or denies
them. Install `mimirai[local,pydantic-ai]` and `pydantic-ai-slim[openai]`, set
`OPENAI_API_KEY`, then run `make example NAME=pydantic_ai/guarded_agent`.
"""

import asyncio
from typing import Any, Final

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.toolsets import FunctionToolset

from mimir import Decider, Mimir
from mimir.integrations.pydantic_ai import guard

MODEL: Final = "openai:gpt-5.5"
RULES: Final = (
    "A refund is at most the amount the customer paid for the order.",
    "Refunds above 500 dollars need a manager's approval.",
)
INSTRUCTIONS: Final = "Issue the refunds customers ask for with issue_refund."
Output = str | DeferredToolRequests


def issue_refund(order: str, amount: float) -> str:
    """Refund `amount` dollars on `order`."""
    return f"Refunded {amount:.2f} dollars on order {order}."


def build_agent(
    decider: Decider,
    refunds: FunctionToolset[Any] | None = None,
    model: str | Model = MODEL,
) -> Agent[None, Output]:
    """The refunds agent, its `refunds` toolset checked by `decider` against `RULES`."""
    toolset = FunctionToolset([issue_refund]) if refunds is None else refunds
    check = decider.tool_call_check(RULES)
    return Agent(
        model,
        instructions=INSTRUCTIONS,
        toolsets=[guard(toolset, check)],
        output_type=[str, DeferredToolRequests],
    )


async def run_with_approvals(agent: Agent[None, Output], request: str) -> AgentRunResult[Output]:
    """Run `agent`, asking at the console about each call the check escalates."""
    result = await agent.run(request)
    while isinstance(result.output, DeferredToolRequests):
        approvals: dict[str, bool | ToolApproved | ToolDenied] = {}
        for pending in result.output.approvals:
            reason = result.output.metadata.get(pending.tool_call_id, {}).get("reason", "")
            answer = input(f"{reason} Approve {pending.tool_name} {pending.args}? [y/N] ")
            approved = answer.strip().lower() == "y"
            approvals[pending.tool_call_id] = (
                True if approved else ToolDenied("A person denied it.")
            )
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals=approvals),
        )
    return result


async def main() -> None:
    agent = build_agent(Mimir.from_pretrained("vathosai/mimir-1"))
    result = await run_with_approvals(agent, "Refund 900 dollars on order 4412.")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
