import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Final

import pytest
from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from pydantic import PrivateAttr

from mimir.core.decisions import Choice
from mimir.core.results import ChoiceResult, Status
from mimir.integrations.google_adk import as_adk_tool, tool_call_callback
from tests.conftest import RecordingDecider, YesNoDecider

TEXT: Final = "my card was charged twice"
RULE: Final = "Refunds above 500 dollars need a manager."
USER: Final = "customer"
CONFIRMATION: Final = "adk_request_confirmation"


class ScriptedLlm(BaseLlm):
    """A non-streaming model answering each turn with the next scripted part."""

    replies: list[types.Part]
    _turns: int = PrivateAttr(default=0)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        assert not stream, f"the scripted model does not stream, asked for {llm_request.model}"
        self._turns += 1
        reply = self.replies[self._turns - 1]
        yield LlmResponse(content=types.Content(role="model", parts=[reply]))


def _call(name: str, args: dict[str, Any]) -> types.Part:
    return types.Part(function_call=types.FunctionCall(id="call-1", name=name, args=args))


def _responses(events: list[Event], name: str) -> list[dict[str, Any]]:
    return [
        response.response or {}
        for event in events
        for response in event.get_function_responses()
        if response.name == name
    ]


class Session:
    """One ADK session on an in-memory runner."""

    def __init__(self, agent: LlmAgent) -> None:
        self.runner = InMemoryRunner(agent=agent, app_name="support")
        session = asyncio.run(
            self.runner.session_service.create_session(app_name="support", user_id=USER)
        )
        self.session_id = session.id

    def send(self, *parts: types.Part) -> list[Event]:
        async def main() -> list[Event]:
            message = types.Content(role="user", parts=list(parts))
            return [
                event
                async for event in self.runner.run_async(
                    user_id=USER, session_id=self.session_id, new_message=message
                )
            ]

        return asyncio.run(main())


def test_the_agent_loop_calls_the_decision_tool_and_reads_its_result() -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    llm = ScriptedLlm(
        model="scripted",
        replies=[_call("route_ticket", {"context": TEXT}), types.Part(text="billing")],
    )
    agent = LlmAgent(name="support", model=llm, tools=[as_adk_tool(route)])
    events = Session(agent).send(types.Part(text="Route this ticket."))
    [response] = _responses(events, "route_ticket")
    result = ChoiceResult.model_validate(response)
    assert (result.answer, result.status) == ("billing", Status.DECIDED)
    assert events[-1].content is not None
    assert events[-1].content.parts is not None
    assert events[-1].content.parts[0].text == "billing"
    assert decider.calls[0].requests[0][0].passages[0].text == TEXT


def test_the_declaration_carries_the_argument_schema() -> None:
    route = RecordingDecider().tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    declaration = as_adk_tool(route)._get_declaration()
    assert (declaration.name, declaration.description) == ("route_ticket", route.agent_description)
    assert declaration.parameters_json_schema == route.input_schema


@pytest.mark.parametrize("args", [{}, {"context": 3}, {"context": "x", "extra": 1}])
def test_invalid_arguments_give_an_error_response(args: dict[str, Any]) -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    llm = ScriptedLlm(
        model="scripted", replies=[_call("route_ticket", args), types.Part(text="done")]
    )
    agent = LlmAgent(name="support", model=llm, tools=[as_adk_tool(route)])
    [response] = _responses(Session(agent).send(types.Part(text="Route.")), "route_ticket")
    assert str(response["error"]).startswith("Invalid arguments for route_ticket: ")
    assert decider.calls == []


def _checked(decider: YesNoDecider, refunds: list[float]) -> Session:
    def issue_refund(amount: float) -> str:
        """Refund the customer."""
        refunds.append(amount)
        return f"refunded {amount}"

    llm = ScriptedLlm(
        model="scripted",
        replies=[_call("issue_refund", {"amount": 900.0}), types.Part(text="done")],
    )
    check = decider.tool_call_check(RULE, tools=["issue_refund"])
    agent = LlmAgent(
        name="refunds",
        model=llm,
        tools=[FunctionTool(issue_refund)],
        before_tool_callback=tool_call_callback(check),
    )
    return Session(agent)


def test_an_allowed_call_runs() -> None:
    refunds: list[float] = []
    decider = YesNoDecider(answer=True)
    events = _checked(decider, refunds).send(types.Part(text="Refund."))
    assert refunds == [900.0]
    assert _responses(events, "issue_refund") == [{"result": "refunded 900.0"}]
    [(context, _)] = decider.calls[0].requests
    assert [(field.key, field.value) for field in context.fields] == [
        ("tool", "issue_refund"),
        ("arguments.amount", 900.0),
    ]


def test_a_denied_call_is_skipped_with_the_reason() -> None:
    refunds: list[float] = []
    events = _checked(YesNoDecider(answer=False), refunds).send(types.Part(text="Refund."))
    assert refunds == []
    assert _responses(events, "issue_refund") == [
        {"error": "The call to issue_refund is not allowed under the rules (confidence 0.90)."}
    ]


def _confirmation_call(events: list[Event]) -> types.FunctionCall:
    [call] = [
        call for event in events for call in event.get_function_calls() if call.name == CONFIRMATION
    ]
    return call


@pytest.mark.parametrize("confirmed", [True, False])
def test_an_escalated_call_waits_for_confirmation(confirmed: bool) -> None:
    refunds: list[float] = []
    session = _checked(YesNoDecider(answer=True, status=Status.DEFERRED), refunds)
    paused = session.send(types.Part(text="Refund."))
    request = _confirmation_call(paused)
    assert request.args is not None
    assert "needs a person's approval" in request.args["toolConfirmation"]["hint"]
    assert refunds == []
    reply = types.FunctionResponse(
        id=request.id, name=CONFIRMATION, response={"confirmed": confirmed}
    )
    resumed = session.send(types.Part(function_response=reply))
    [response] = _responses(resumed, "issue_refund")
    if confirmed:
        assert (refunds, response) == ([900.0], {"result": "refunded 900.0"})
    else:
        assert (refunds, response) == ([], {"error": "A person rejected the call to issue_refund."})
