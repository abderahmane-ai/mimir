import asyncio
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Final

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState, InputAgentState, OutputAgentState
    from langgraph.graph.state import CompiledStateGraph

from mimir.core.decisions import Choice
from mimir.core.results import ChoiceResult, Status
from mimir.integrations.langchain import ToolCallCheckMiddleware, as_structured_tool
from tests.conftest import RecordingDecider, YesNoDecider

TEXT: Final = "my card was charged twice"
RULE: Final = "Refunds above 500 dollars need a manager."
THREAD: Final[RunnableConfig] = {"configurable": {"thread_id": "ticket-4412"}}
if TYPE_CHECKING:
    Agent = CompiledStateGraph[AgentState[Any], None, InputAgentState, OutputAgentState[Any]]


class ScriptedChatModel(GenericFakeChatModel):
    """A fake chat model that accepts tools and answers with its scripted messages."""

    def bind_tools(self, *_: object, **__: object) -> "ScriptedChatModel":
        return self


def _script(*messages: AIMessage) -> ScriptedChatModel:
    replies: Iterator[AIMessage | str] = iter(messages)
    return ScriptedChatModel(messages=replies)


def _call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}])


def _tool_messages(state: dict[str, Any]) -> list[ToolMessage]:
    return [message for message in state["messages"] if isinstance(message, ToolMessage)]


def test_the_agent_loop_calls_the_decision_tool_and_keeps_the_typed_result() -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    model = _script(_call("route_ticket", {"context": TEXT}), AIMessage(content="billing"))
    agent = create_agent(model, tools=[as_structured_tool(route)])
    state = agent.invoke({"messages": [HumanMessage("Route this ticket.")]})
    [message] = _tool_messages(state)
    assert isinstance(message.artifact, ChoiceResult)
    assert (message.artifact.answer, message.artifact.status) == ("billing", Status.DECIDED)
    assert ChoiceResult.model_validate_json(str(message.content)) == message.artifact
    assert state["messages"][-1].content == "billing"
    assert decider.calls[0].requests[0][0].passages[0].text == TEXT


def test_an_async_agent_runs_the_tool_through_its_coroutine() -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    model = _script(_call("route_ticket", {"context": [TEXT]}), AIMessage(content="billing"))
    agent = create_agent(model, tools=[as_structured_tool(route)])
    state = asyncio.run(agent.ainvoke({"messages": [HumanMessage("Route this ticket.")]}))
    [message] = _tool_messages(state)
    assert isinstance(message.artifact, ChoiceResult)
    assert message.status == "success"
    assert decider.calls[0].requests[0][0].passages[0].text == TEXT


def test_the_structured_tool_carries_the_decision_tool_contract() -> None:
    route = RecordingDecider().tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    converted = as_structured_tool(route)
    assert (converted.name, converted.description) == ("route_ticket", route.agent_description)
    assert converted.args_schema == route.input_schema
    assert converted.response_format == "content_and_artifact"


@pytest.mark.parametrize("args", [{}, {"context": 3}, {"context": "x", "extra": 1}])
def test_invalid_arguments_become_an_error_message(args: dict[str, Any]) -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    model = _script(_call("route_ticket", args), AIMessage(content="done"))
    agent = create_agent(model, tools=[as_structured_tool(route)])
    [message] = _tool_messages(agent.invoke({"messages": [HumanMessage("Route.")]}))
    assert message.status == "error"
    assert "validation error" in str(message.content)
    assert decider.calls == []


def _refund_tool(refunds: list[float]) -> BaseTool:
    @tool
    def issue_refund(amount: float) -> str:
        """Refund the customer."""
        refunds.append(amount)
        return f"refunded {amount}"

    return issue_refund


def _checked_agent(decider: YesNoDecider, refunds: list[float], *replies: AIMessage) -> "Agent":
    check = decider.tool_call_check(RULE, tools=["issue_refund"])
    model = _script(_call("issue_refund", {"amount": 900.0}), *replies)
    return create_agent(
        model,
        tools=[_refund_tool(refunds)],
        middleware=[ToolCallCheckMiddleware(check)],
        checkpointer=InMemorySaver(),
    )


def test_an_allowed_call_runs() -> None:
    refunds: list[float] = []
    decider = YesNoDecider(answer=True)
    agent = _checked_agent(decider, refunds, AIMessage(content="done"))
    state = agent.invoke({"messages": [HumanMessage("Refund.")]}, THREAD)
    assert refunds == [900.0]
    assert [message.content for message in _tool_messages(state)] == ["refunded 900.0"]
    [(context, _)] = decider.calls[0].requests
    assert [(field.key, field.value) for field in context.fields] == [
        ("tool", "issue_refund"),
        ("arguments.amount", 900.0),
    ]


def test_a_denied_call_becomes_an_error_message_with_the_reason() -> None:
    refunds: list[float] = []
    agent = _checked_agent(YesNoDecider(answer=False), refunds, AIMessage(content="done"))
    state = asyncio.run(agent.ainvoke({"messages": [HumanMessage("Refund.")]}, THREAD))
    [message] = _tool_messages(state)
    assert refunds == []
    assert (message.status, message.tool_call_id) == ("error", "call-1")
    assert message.content == (
        "The call to issue_refund is not allowed under the rules (confidence 0.90)."
    )


def test_an_escalated_call_interrupts_with_the_human_in_the_loop_request() -> None:
    refunds: list[float] = []
    decider = YesNoDecider(answer=True, status=Status.DEFERRED)
    agent = _checked_agent(decider, refunds, AIMessage(content="done"))
    paused = agent.invoke({"messages": [HumanMessage("Refund.")]}, THREAD)
    [pending] = paused["__interrupt__"]
    [action] = pending.value["action_requests"]
    assert (action["name"], action["args"]) == ("issue_refund", {"amount": 900.0})
    assert "needs a person's approval" in action["description"]
    assert pending.value["review_configs"] == [
        {"action_name": "issue_refund", "allowed_decisions": ["approve", "reject"]}
    ]
    assert refunds == []
    resumed = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), THREAD)
    assert refunds == [900.0]
    assert resumed["messages"][-1].content == "done"


def test_a_rejected_escalation_tells_the_model_why() -> None:
    refunds: list[float] = []
    decider = YesNoDecider(answer=True, status=Status.DEFERRED)
    agent = _checked_agent(decider, refunds, AIMessage(content="done"))
    agent.invoke({"messages": [HumanMessage("Refund.")]}, THREAD)
    decision = {"type": "reject", "message": "Ask the manager first."}
    state = agent.invoke(Command(resume={"decisions": [decision]}), THREAD)
    [message] = _tool_messages(state)
    assert (message.content, message.status, refunds) == ("Ask the manager first.", "error", [])


@pytest.mark.parametrize(
    "response", [{"decisions": []}, {"decisions": [{"type": "edit"}]}, "approve"]
)
def test_a_malformed_review_response_raises_naming_the_tool(response: object) -> None:
    decider = YesNoDecider(answer=True, status=Status.DEFERRED)
    agent = _checked_agent(decider, [], AIMessage(content="done"))
    agent.invoke({"messages": [HumanMessage("Refund.")]}, THREAD)
    with pytest.raises(ValueError, match="issue_refund"):
        agent.invoke(Command(resume=response), THREAD)
