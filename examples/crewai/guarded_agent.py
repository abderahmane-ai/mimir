"""A CrewAI crew whose refunds MIMIR checks against the refund rules first.

A confident yes runs the refund, a confident no blocks it, and anything else asks a person at
the console. The check is a global before-tool-call hook, registered while the crew runs.
Install `mimir-decisions[local,crewai]`, set `OPENAI_API_KEY`, then run
`make example NAME=crewai/guarded_agent`.
"""

from typing import Final

from crewai import LLM, Agent, Crew, Task
from crewai.hooks import (
    ToolCallHookContext,
    register_before_tool_call_hook,
    unregister_before_tool_call_hook,
)
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool, tool

from mimir import CheckOutcome, Decider, Mimir
from mimir.integrations.crewai import tool_call_hook

MODEL: Final = "openai/gpt-5.5"
RULES: Final = (
    "A refund is at most the amount the customer paid for the order.",
    "Refunds above 500 dollars need a manager's approval.",
)


@tool("issue_refund")
def issue_refund(order: str, amount: float) -> str:
    """Refund `amount` dollars on `order`."""
    return f"Refunded {amount:.2f} dollars on order {order}."


def ask_at_the_console(context: ToolCallHookContext, outcome: CheckOutcome) -> bool:
    """Whether the person at the console approves the escalated call."""
    answer = context.request_human_input(
        prompt=f"{outcome.reason} Approve {context.tool_name} {context.tool_input}?",
        default_message="Type 'y' to approve:",
    )
    return answer.strip().lower() == "y"


def build_crew(refund: BaseTool = issue_refund, llm: BaseLLM | None = None) -> Crew:
    """A one-agent crew issuing the refund given as the `request` input."""
    agent = Agent(
        role="Refunds agent",
        goal="Issue the refunds customers ask for.",
        backstory="You issue refunds with issue_refund.",
        llm=LLM(model=MODEL) if llm is None else llm,
        tools=[refund],
    )
    task = Task(description="{request}", expected_output="What was refunded.", agent=agent)
    return Crew(agents=[agent], tasks=[task])


def run_checked(crew: Crew, decider: Decider, request: str) -> str:
    """Run `crew` with its refunds checked by `decider` against `RULES`."""
    check = decider.tool_call_check(RULES, tools=["issue_refund"])
    hook = tool_call_hook(check, approve=ask_at_the_console)
    register_before_tool_call_hook(hook)
    try:
        return str(crew.kickoff(inputs={"request": request}))
    finally:
        unregister_before_tool_call_hook(hook)


def main() -> None:
    decider = Mimir.from_pretrained("Mythologic/MIMIR-1")
    print(run_checked(build_crew(), decider, "Refund 900 dollars on order 4412."))


if __name__ == "__main__":
    main()
