import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Final

import pytest
from agent_framework import (
    Agent,
    AgentResponse,
    BaseChatClient,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    FunctionTool,
    Message,
)

from mimir.core.decisions import Choice
from mimir.core.results import ChoiceResult, Status
from mimir.integrations.agent_framework import ToolCallCheckMiddleware, as_function_tool
from tests.conftest import RecordingDecider, YesNoDecider

TEXT: Final = "my card was charged twice"
RULE: Final = "Refunds above 500 dollars need a manager."


class ScriptedChatClient(FunctionInvocationLayer[Any], BaseChatClient[Any]):
    """A non-streaming chat client: one scripted call, then the last tool result as text."""

    def __init__(self, name: str, arguments: Mapping[str, Any]) -> None:
        super().__init__()
        self.name = name
        self.arguments = dict(arguments)

    async def _respond(self, messages: Sequence[Message]) -> ChatResponse:
        results = [
            content
            for message in messages
            for content in message.contents
            if content.type == "function_result"
        ]
        if not results:
            call = Content.from_function_call(
                call_id="call-1", name=self.name, arguments=self.arguments
            )
            return ChatResponse(messages=[Message(role="assistant", contents=[call])])
        return ChatResponse(
            messages=[Message(role="assistant", contents=[str(results[-1].result)])]
        )

    def _inner_get_response(
        self, *, messages: Sequence[Message], stream: bool, **_: object
    ) -> Awaitable[ChatResponse]:
        assert not stream, "the scripted client does not stream"
        return self._respond(messages)


def _run(agent: Agent) -> AgentResponse:
    async def main() -> AgentResponse:
        return await agent.run("Handle the ticket.")

    return asyncio.run(main())


def _results(response: AgentResponse) -> list[Content]:
    return [
        content
        for message in response.messages
        for content in message.contents
        if content.type == "function_result"
    ]


def test_the_agent_loop_calls_the_decision_tool_and_reads_its_result() -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    client = ScriptedChatClient("route_ticket", {"context": TEXT})
    response = _run(Agent(client=client, tools=[as_function_tool(route)]))
    result = ChoiceResult.model_validate_json(response.text)
    assert (result.answer, result.status) == ("billing", Status.DECIDED)
    assert decider.calls[0].requests[0][0].passages[0].text == TEXT


def test_the_function_tool_carries_the_decision_tool_contract() -> None:
    route = RecordingDecider().tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    converted = as_function_tool(route)
    assert (converted.name, converted.description) == ("route_ticket", route.agent_description)
    assert converted.parameters()["required"] == ["context"]


@pytest.mark.parametrize("arguments", [{}, {"context": 3}])
def test_invalid_arguments_go_back_to_the_model(arguments: dict[str, Any]) -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    client = ScriptedChatClient("route_ticket", arguments)
    response = _run(Agent(client=client, tools=[as_function_tool(route)]))
    [result] = _results(response)
    assert result.result == "Error: Argument parsing failed."
    assert "for ToolArguments\ncontext" in str(result.exception)
    assert decider.calls == []


def _refund_tool(refunds: list[float]) -> FunctionTool:
    def issue_refund(amount: float) -> str:
        """Refund the customer."""
        refunds.append(amount)
        return f"refunded {amount}"

    return FunctionTool(name="issue_refund", description="Refund the customer.", func=issue_refund)


def _checked(decider: YesNoDecider, refunds: list[float]) -> AgentResponse:
    check = decider.tool_call_check(RULE, tools=["issue_refund"])
    client = ScriptedChatClient("issue_refund", {"amount": 900.0})
    agent = Agent(
        client=client, tools=[_refund_tool(refunds)], middleware=[ToolCallCheckMiddleware(check)]
    )
    return _run(agent)


def test_an_allowed_call_runs() -> None:
    refunds: list[float] = []
    decider = YesNoDecider(answer=True)
    response = _checked(decider, refunds)
    assert (refunds, response.text) == ([900.0], "refunded 900.0")
    [(context, _)] = decider.calls[0].requests
    assert [(field.key, field.value) for field in context.fields] == [
        ("tool", "issue_refund"),
        ("arguments.amount", 900.0),
    ]


@pytest.mark.parametrize(
    ("answer", "status", "reason"),
    [
        (False, Status.DECIDED, "is not allowed under the rules"),
        (True, Status.DEFERRED, "needs a person's approval"),
    ],
)
def test_a_denied_or_escalated_call_is_not_run(answer: bool, status: Status, reason: str) -> None:
    refunds: list[float] = []
    response = _checked(YesNoDecider(answer=answer, status=status), refunds)
    assert refunds == []
    assert reason in response.text
