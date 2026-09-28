"""An OpenAI Agents SDK agent whose refunds MIMIR checks against the refund rules first.

A certified yes runs the refund, a certified no rejects it with the reason, and anything else
pauses the run until a person at the console approves or rejects it. Install
`mimir-decisions[local,openai-agents]`, set `OPENAI_API_KEY`, then run
`make example NAME=openai_agents/guarded_agent`.
"""

import asyncio
from typing import Final

from agents import Agent, FunctionTool, Model, Runner, RunResult, function_tool

from mimir import Decider, Mimir
from mimir.integrations.openai_agents import guard

RULES: Final = (
    "A refund is at most the amount the customer paid for the order.",
    "Refunds above 500 dollars need a manager's approval.",
)
INSTRUCTIONS: Final = "Issue the refunds customers ask for with issue_refund."


@function_tool
def issue_refund(order: str, amount: float) -> str:
    """Refund `amount` dollars on `order`."""
    return f"Refunded {amount:.2f} dollars on order {order}."


def build_agent(
    decider: Decider, refund: FunctionTool = issue_refund, model: str | Model | None = None
) -> Agent[None]:
    """The refunds agent, its `refund` tool checked by `decider` against `RULES`."""
    check = decider.tool_call_check(RULES, tools=[refund.name])
    return Agent(
        name="refunds", instructions=INSTRUCTIONS, tools=[guard(refund, check)], model=model
    )


async def run_with_approvals(agent: Agent[None], request: str) -> RunResult:
    """Run `agent`, asking at the console about each call the check escalates."""
    result = await Runner.run(agent, request)
    while result.interruptions:
        state = result.to_state()
        for pending in result.interruptions:
            answer = input(f"Approve {pending.name} {pending.arguments}? [y/N] ")
            if answer.strip().lower() == "y":
                state.approve(pending)
            else:
                state.reject(pending)
        result = await Runner.run(agent, state)
    return result


async def main() -> None:
    agent = build_agent(Mimir.from_pretrained("Mythologic/MIMIR-1"))
    result = await run_with_approvals(agent, "Refund 900 dollars on order 4412.")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
