"""A LangChain agent that routes support tickets with a MIMIR decision tool.

The tool's typed result is the tool message's `artifact`. Install
`mimirai[local,langchain]` and `langchain-openai`, set `OPENAI_API_KEY`, then run
`make example NAME=langchain/agent`.
"""

from typing import TYPE_CHECKING, Any, Final

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from mimir import Choice, Decider, Mimir
from mimir.integrations.langchain import as_structured_tool

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState, InputAgentState, OutputAgentState
    from langgraph.graph.state import CompiledStateGraph

    Agent = CompiledStateGraph[AgentState[Any], None, InputAgentState, OutputAgentState[Any]]

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


def build_agent(decider: Decider, model: str | BaseChatModel = MODEL) -> "Agent":
    """The support agent, answering with `decider`."""
    route_ticket = decider.tool(
        "route_ticket",
        Choice("Which team should handle this ticket?", TEAMS),
        description="Route a customer support ticket to the team that owns it.",
    )
    return create_agent(model, tools=[as_structured_tool(route_ticket)], system_prompt=INSTRUCTIONS)


def main() -> None:
    agent = build_agent(Mimir.from_pretrained("Mythologic/MIMIR-1"))
    state = agent.invoke({"messages": [{"role": "user", "content": "I was charged twice."}]})
    print(state["messages"][-1].content)


if __name__ == "__main__":
    main()
