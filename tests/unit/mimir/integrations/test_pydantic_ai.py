import asyncio
from collections.abc import Callable
from typing import Any, Final

import pytest
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from mimir.core.decisions import Choice
from mimir.core.results import ChoiceResult, Status
from mimir.integrations.pydantic_ai import as_tool, as_toolset, guard
from tests.conftest import RecordingDecider, YesNoDecider

TEXT: Final = "my card was charged twice"
RULE: Final = "Refunds above 500 dollars need a manager."
Script = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


def _script(name: str, args: dict[str, Any]) -> Script:
    """Call `name` with `args` once, then answer with the last tool part's content."""

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        parts = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ]
        returned = [part for part in parts if isinstance(part, ToolReturnPart | RetryPromptPart)]
        if not returned:
            return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id="call-1")])
        return ModelResponse(parts=[TextPart(str(returned[-1].content))])

    return respond


def _returns(messages: list[ModelMessage]) -> list[ToolReturnPart | RetryPromptPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart | RetryPromptPart)
    ]


def test_the_agent_loop_calls_the_decision_tool_and_keeps_the_typed_result() -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    model = FunctionModel(_script("route_ticket", {"context": TEXT}))
    agent = Agent(model, toolsets=[as_toolset([route])])
    result = asyncio.run(agent.run("Route this ticket."))
    [returned] = _returns(result.all_messages())
    assert isinstance(returned, ToolReturnPart)
    assert isinstance(returned.content, ChoiceResult)
    assert (returned.content.answer, returned.content.status) == ("billing", Status.DECIDED)
    assert decider.calls[0].requests[0][0].passages[0].text == TEXT


def test_the_tool_carries_the_decision_tool_contract() -> None:
    route = RecordingDecider().tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    definition = as_tool(route).tool_def
    assert (definition.name, definition.description) == ("route_ticket", route.agent_description)
    assert definition.parameters_json_schema == route.input_schema


@pytest.mark.parametrize("args", [{}, {"context": 3}, {"context": "x", "extra": 1}])
def test_invalid_arguments_ask_the_model_to_retry(args: dict[str, Any]) -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    agent = Agent(FunctionModel(_script("route_ticket", args)), toolsets=[as_toolset([route])])
    result = asyncio.run(agent.run("Route."))
    [returned] = _returns(result.all_messages())
    assert isinstance(returned, RetryPromptPart)
    assert "validation error" in str(returned.content)
    assert decider.calls == []


def _refunds_toolset(refunds: list[float]) -> FunctionToolset[Any]:
    def issue_refund(amount: float) -> str:
        """Refund the customer."""
        refunds.append(amount)
        return f"refunded {amount}"

    return FunctionToolset([issue_refund])


def _checked_agent(
    decider: YesNoDecider, refunds: list[float]
) -> Agent[None, str | DeferredToolRequests]:
    check = decider.tool_call_check(RULE, tools=["issue_refund"])
    return Agent(
        FunctionModel(_script("issue_refund", {"amount": 900.0})),
        toolsets=[guard(_refunds_toolset(refunds), check)],
        output_type=[str, DeferredToolRequests],
    )


def test_an_allowed_call_runs() -> None:
    refunds: list[float] = []
    decider = YesNoDecider(answer=True)
    result = asyncio.run(_checked_agent(decider, refunds).run("Refund."))
    assert (refunds, result.output) == ([900.0], "refunded 900.0")
    [(context, _)] = decider.calls[0].requests
    assert [(field.key, field.value) for field in context.fields] == [
        ("tool", "issue_refund"),
        ("arguments.amount", 900.0),
    ]


def test_a_denied_call_fails_with_the_reason() -> None:
    refunds: list[float] = []
    result = asyncio.run(_checked_agent(YesNoDecider(answer=False), refunds).run("Refund."))
    assert refunds == []
    assert (
        result.output
        == "The call to issue_refund is not allowed under the rules (confidence 0.90)."
    )


def test_an_escalated_call_waits_for_approval_then_runs() -> None:
    refunds: list[float] = []
    agent = _checked_agent(YesNoDecider(answer=True, status=Status.DEFERRED), refunds)
    paused = asyncio.run(agent.run("Refund."))
    assert isinstance(paused.output, DeferredToolRequests)
    [pending] = paused.output.approvals
    assert (pending.tool_name, pending.args_as_dict()) == ("issue_refund", {"amount": 900.0})
    assert "needs a person's approval" in paused.output.metadata[pending.tool_call_id]["reason"]
    assert refunds == []
    approvals = DeferredToolResults(approvals={pending.tool_call_id: True})
    resumed = asyncio.run(
        agent.run(message_history=paused.all_messages(), deferred_tool_results=approvals)
    )
    assert (refunds, resumed.output) == ([900.0], "refunded 900.0")


def test_a_denied_approval_tells_the_model_why() -> None:
    refunds: list[float] = []
    agent = _checked_agent(YesNoDecider(answer=True, status=Status.DEFERRED), refunds)
    paused = asyncio.run(agent.run("Refund."))
    assert isinstance(paused.output, DeferredToolRequests)
    [pending] = paused.output.approvals
    denial = DeferredToolResults(approvals={pending.tool_call_id: ToolDenied("Ask the manager.")})
    resumed = asyncio.run(
        agent.run(message_history=paused.all_messages(), deferred_tool_results=denial)
    )
    assert (refunds, resumed.output) == ([], "Ask the manager.")
