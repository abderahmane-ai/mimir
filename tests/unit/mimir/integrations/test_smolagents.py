import ast
from typing import Any, Final

import pytest
from smolagents import CodeAgent, Tool, ToolCallingAgent
from smolagents.memory import ActionStep
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
)

from mimir.core.decisions import Choice
from mimir.core.results import ChoiceResult, Status
from mimir.integrations.smolagents import CONTEXT_INPUT, as_smolagents_tool
from tests.conftest import RecordingDecider

TEXT: Final = "my card was charged twice"


class ScriptedModel(Model):
    """A model answering each turn with the next scripted message."""

    def __init__(self, replies: list[ChatMessage]) -> None:
        super().__init__(model_id="scripted")
        self.replies = replies
        self.prompts: list[list[ChatMessage]] = []

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: object,
    ) -> ChatMessage:
        self.prompts.append(messages)
        return self.replies[len(self.prompts) - 1]


def _call(name: str, arguments: dict[str, Any]) -> ChatMessage:
    function = ChatMessageToolCallFunction(name=name, arguments=arguments)
    call = ChatMessageToolCall(function=function, id=f"call-{name}", type="function")
    return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[call])


def _observations(agent: ToolCallingAgent | CodeAgent) -> list[str]:
    return [
        step.observations
        for step in agent.memory.steps
        if isinstance(step, ActionStep) and step.observations is not None
    ]


def test_a_tool_calling_agent_calls_the_decision_tool_and_reads_its_result() -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    model = ScriptedModel(
        [_call("route_ticket", {"context": TEXT}), _call("final_answer", {"answer": "done"})]
    )
    agent = ToolCallingAgent(tools=[as_smolagents_tool(route)], model=model)
    assert agent.run("Route this ticket.") == "done"
    observed = ast.literal_eval(_observations(agent)[0])
    result = ChoiceResult.model_validate(observed)
    assert (result.answer, result.status) == ("billing", Status.DECIDED)
    assert decider.calls[0].requests[0][0].passages[0].text == TEXT


def test_a_code_agent_reads_the_result_as_a_dict() -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    code = (
        f"result = route_ticket(context={TEXT!r})\n"
        "final_answer(result['status'] + ':' + result['answer'])"
    )
    model = ScriptedModel(
        [
            ChatMessage(
                role=MessageRole.ASSISTANT, content=f"Thought: route it.\n<code>\n{code}\n</code>"
            )
        ]
    )
    agent = CodeAgent(tools=[as_smolagents_tool(route)], model=model)
    assert agent.run("Route this ticket.") == "decided:billing"


def test_the_tool_carries_the_decision_tool_contract() -> None:
    route = RecordingDecider().tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    converted = as_smolagents_tool(route)
    assert (converted.name, converted.description) == ("route_ticket", route.agent_description)
    assert converted.inputs == {"context": CONTEXT_INPUT}
    assert (converted.output_type, converted.output_schema) == ("object", route.output_schema)


@pytest.mark.parametrize(
    ("context", "error"),
    [
        (3, "Argument context has type 'integer'"),
        ({"passages": 3}, "validation error"),
    ],
)
def test_invalid_arguments_are_a_tool_error_the_model_reads(context: object, error: str) -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    model = ScriptedModel(
        [_call("route_ticket", {"context": context}), _call("final_answer", {"answer": "done"})]
    )
    agent = ToolCallingAgent(tools=[as_smolagents_tool(route)], model=model)
    assert agent.run("Route.") == "done"
    [failed] = [
        step
        for step in agent.memory.steps
        if isinstance(step, ActionStep) and step.error is not None
    ]
    assert error in str(failed.error)
    assert error in str(model.prompts[1])
    assert decider.calls == []


@pytest.mark.parametrize("name", ["route-ticket", "class"])
def test_names_smolagents_cannot_call_are_refused(name: str) -> None:
    route = RecordingDecider().tool(name, Choice("q", ["a", "b"]), "Route.")
    with pytest.raises(ValueError, match="Python identifiers"):
        as_smolagents_tool(route)
