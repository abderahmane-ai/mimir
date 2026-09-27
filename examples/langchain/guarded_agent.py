"""A LangChain agent whose refunds MIMIR checks against the refund rules first.

A certified yes runs the refund, a certified no returns the reason to the model, and anything
else interrupts the graph with LangChain's human-in-the-loop request until a person at the
console approves or rejects it. Install `mimirai[local,langchain]` and `langchain-openai`,
set `OPENAI_API_KEY`, then run `make example NAME=langchain/guarded_agent`.
"""

from typing import TYPE_CHECKING, Any, Final

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from mimir import Decider, Mimir
from mimir.integrations.langchain import ToolCallCheckMiddleware

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState, InputAgentState, OutputAgentState
    from langgraph.graph.state import CompiledStateGraph

    Agent = CompiledStateGraph[AgentState[Any], None, InputAgentState, OutputAgentState[Any]]

MODEL: Final = "openai:gpt-5.5"
RULES: Final = (
    "A refund is at most the amount the customer paid for the order.",
    "Refunds above 500 dollars need a manager's approval.",
)
INSTRUCTIONS: Final = "Issue the refunds customers ask for with issue_refund."


@tool
def issue_refund(order: str, amount: float) -> str:
    """Refund `amount` dollars on `order`."""
    return f"Refunded {amount:.2f} dollars on order {order}."


def build_agent(
    decider: Decider, refund: BaseTool = issue_refund, model: str | BaseChatModel = MODEL
) -> "Agent":
    """The refunds agent, its `refund` tool checked by `decider` against `RULES`."""
    check = decider.tool_call_check(RULES, tools=[refund.name])
    return create_agent(
        model,
        tools=[refund],
        system_prompt=INSTRUCTIONS,
        middleware=[ToolCallCheckMiddleware(check)],
        checkpointer=InMemorySaver(),
    )


def run_with_approvals(agent: "Agent", request: str, thread: str) -> dict[str, Any]:
    """Run `agent` on `thread`, asking at the console about each call the check escalates."""
    config: RunnableConfig = {"configurable": {"thread_id": thread}}
    state: dict[str, Any] = agent.invoke(
        {"messages": [{"role": "user", "content": request}]}, config
    )
    while "__interrupt__" in state:
        resume = {}
        for pending in state["__interrupt__"]:
            [action] = pending.value["action_requests"]
            answer = input(
                f"{action['description']} Approve {action['name']} {action['args']}? [y/N] "
            )
            decision = "approve" if answer.strip().lower() == "y" else "reject"
            resume[pending.id] = {"decisions": [{"type": decision}]}
        state = agent.invoke(Command(resume=resume), config)
    return state


def main() -> None:
    agent = build_agent(Mimir.from_pretrained("mythologic/mimir-1"))
    state = run_with_approvals(agent, "Refund 900 dollars on order 4412.", thread="ticket-4412")
    print(state["messages"][-1].content)


if __name__ == "__main__":
    main()
