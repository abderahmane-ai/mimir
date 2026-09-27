import asyncio
from collections.abc import Sequence
from typing import Any, Final

import pytest
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ToolCallBlock
from llama_index.core.llms import MockFunctionCallingLLM

from mimir.core.decisions import Choice
from mimir.core.results import ChoiceResult, Status
from mimir.integrations.llamaindex import as_llamaindex_tool
from tests.conftest import RecordingDecider

TEXT: Final = "my card was charged twice"


def _script(name: str, arguments: dict[str, Any]) -> MockFunctionCallingLLM:
    """Call `name` with `arguments` once, then answer with the tool's content."""

    def respond(messages: Sequence[ChatMessage], **_: object) -> ChatMessage:
        returned = [message for message in messages if message.role == MessageRole.TOOL]
        if not returned:
            call = ToolCallBlock(tool_call_id="call-1", tool_name=name, tool_kwargs=arguments)
            return ChatMessage(role=MessageRole.ASSISTANT, blocks=[call])
        text = TextBlock(text=str(returned[-1].content))
        return ChatMessage(role=MessageRole.ASSISTANT, blocks=[text])

    return MockFunctionCallingLLM(response_generator=respond)


def _run(agent: FunctionAgent) -> tuple[str, list[ToolCallResult]]:
    async def main() -> tuple[str, list[ToolCallResult]]:
        handler = agent.run(user_msg="Route this ticket.")
        results = [
            event async for event in handler.stream_events() if isinstance(event, ToolCallResult)
        ]
        return str(await handler), results

    return asyncio.run(main())


def test_the_agent_loop_calls_the_decision_tool_and_keeps_the_typed_result() -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    agent = FunctionAgent(
        tools=[as_llamaindex_tool(route)], llm=_script("route_ticket", {"context": TEXT})
    )
    answer, [call] = _run(agent)
    result = call.tool_output.raw_output
    assert isinstance(result, ChoiceResult)
    assert (result.answer, result.status) == ("billing", Status.DECIDED)
    assert ChoiceResult.model_validate_json(answer) == result
    assert decider.calls[0].requests[0][0].passages[0].text == TEXT


def test_the_tool_carries_the_decision_tool_contract() -> None:
    route = RecordingDecider().tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    metadata = as_llamaindex_tool(route).metadata
    assert (metadata.name, metadata.description) == ("route_ticket", route.agent_description)
    parameters = metadata.get_parameters_dict()
    assert parameters["required"] == ["context"]
    assert parameters["properties"] == route.input_schema["properties"]


@pytest.mark.parametrize("arguments", [{}, {"context": 3}, {"context": "x", "extra": 1}])
def test_invalid_arguments_give_an_error_output(arguments: dict[str, Any]) -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    agent = FunctionAgent(tools=[as_llamaindex_tool(route)], llm=_script("route_ticket", arguments))
    answer, [call] = _run(agent)
    assert call.tool_output.is_error
    assert answer.startswith("Invalid arguments for route_ticket: ")
    assert decider.calls == []


def test_positional_input_is_an_error_output() -> None:
    route = RecordingDecider().tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    output = as_llamaindex_tool(route).call(TEXT)
    assert output.is_error
    assert "keyword arguments only" in str(output.content)
