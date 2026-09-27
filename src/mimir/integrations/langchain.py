"""Decision tools as LangChain tools, and tool-call checks as agent middleware.

`as_structured_tool` gives a `StructuredTool` for `create_agent` and LangGraph's `ToolNode`:
the model reads the result as JSON, and the typed result is the tool message's `artifact`.
Invalid arguments become an error tool message.

`ToolCallCheckMiddleware` decides each call a check applies to. A denied call becomes an error
tool message carrying the check's reason. An escalated call interrupts the graph with the
request of LangChain's `HumanInTheLoopMiddleware`, so the same reviewers and UIs handle it; the
run resumes with `Command(resume={"decisions": [{"type": "approve"}]})`, or a `reject`
decision with an optional `message`. Interrupts need the agent to have a checkpointer.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    HITLRequest,
    ReviewConfig,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool, ToolException
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command, interrupt

from mimir.core.checks import CheckOutcome, Permission, ToolCallCheck
from mimir.core.results import DecisionResult
from mimir.core.tools import ARGUMENT_ERRORS, DecisionTool

ToolReply = ToolMessage | Command[Any]


def as_structured_tool(tool: DecisionTool) -> StructuredTool:
    """The LangChain tool of `tool`, with its argument schema."""

    def run(**arguments: object) -> tuple[str, DecisionResult]:
        try:
            result = tool.call_with(arguments)
        except ARGUMENT_ERRORS as error:
            raise ToolException(str(error)) from error
        return result.model_dump_json(), result

    async def arun(**arguments: object) -> tuple[str, DecisionResult]:
        try:
            result = await tool.acall_with(arguments)
        except ARGUMENT_ERRORS as error:
            raise ToolException(str(error)) from error
        return result.model_dump_json(), result

    return StructuredTool.from_function(
        func=run,
        coroutine=arun,
        name=tool.name,
        description=tool.agent_description,
        args_schema=tool.input_schema,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )


class ToolCallCheckMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    """Agent middleware applying a `ToolCallCheck` before each tool call it covers."""

    def __init__(self, check: ToolCallCheck) -> None:
        super().__init__()
        self.check = check

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolReply]
    ) -> ToolReply:
        call = request.tool_call
        refusal = _refusal(self.check(call["name"], call["args"]), request)
        return handler(request) if refusal is None else refusal

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolReply]],
    ) -> ToolReply:
        call = request.tool_call
        refusal = _refusal(await self.check.acall(call["name"], call["args"]), request)
        return await handler(request) if refusal is None else refusal


def _refusal(outcome: CheckOutcome, request: ToolCallRequest) -> ToolMessage | None:
    """None when the call may run; otherwise the tool message the model reads instead."""
    call = request.tool_call
    if outcome.permission is Permission.ALLOW:
        return None
    if outcome.permission is Permission.DENY:
        return _error_message(outcome.reason, request)
    review = HITLRequest(
        action_requests=[
            ActionRequest(name=call["name"], args=call["args"], description=outcome.reason)
        ],
        review_configs=[
            ReviewConfig(action_name=call["name"], allowed_decisions=["approve", "reject"])
        ],
    )
    response = interrupt(review)
    decisions = response.get("decisions") if isinstance(response, dict) else None
    if not isinstance(decisions, list) or len(decisions) != 1 or not isinstance(decisions[0], dict):
        message = f"resume the {call['name']} review with one decision, got {response!r}"
        raise ValueError(message)
    decision = decisions[0]
    if decision.get("type") == "approve":
        return None
    if decision.get("type") == "reject":
        rejected = f"A person rejected the call to {call['name']}."
        return _error_message(str(decision.get("message") or rejected), request)
    message = f"a {call['name']} review decision is 'approve' or 'reject', got {decision!r}"
    raise ValueError(message)


def _error_message(content: str, request: ToolCallRequest) -> ToolMessage:
    call = request.tool_call
    return ToolMessage(
        content=content, tool_call_id=call["id"] or "", name=call["name"], status="error"
    )
