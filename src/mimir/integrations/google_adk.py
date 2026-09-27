"""Decision tools as Google ADK tools, and tool-call checks as a before-tool callback.

`as_adk_tool` gives a tool whose declaration carries the decision tool's argument schema and
whose response is the result as a JSON object. Invalid arguments give an `error` response the
model reads.

`tool_call_callback` gives a `before_tool_callback` for an `LlmAgent`. A denied call is
skipped with an `error` response carrying the check's reason. An escalated call asks for
confirmation through ADK's own flow, as tools built with `require_confirmation` do: the run
emits an `adk_request_confirmation` call and resumes when the reply confirms or rejects it.

ADK 2.10 marks JSON Schema declarations and tool confirmation experimental and warns on use.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from mimir.core.checks import Permission, ToolCallCheck
from mimir.core.tools import ARGUMENT_ERRORS, DecisionTool

ToolCallCallback = Callable[
    [BaseTool, dict[str, Any], ToolContext], Awaitable[dict[str, Any] | None]
]


class AdkDecisionTool(BaseTool):
    """An ADK tool answering with a `DecisionTool`, named and described after it."""

    def __init__(self, tool: DecisionTool) -> None:
        super().__init__(name=tool.name, description=tool.agent_description)
        self._tool = tool

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._tool.input_schema,
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> dict[str, Any]:
        try:
            result = await self._tool.acall_with(args)
        except ARGUMENT_ERRORS as error:
            return {"error": f"Invalid arguments for {self.name}: {error}"}
        return result.model_dump(mode="json")


def as_adk_tool(tool: DecisionTool) -> AdkDecisionTool:
    """The ADK tool of `tool`."""
    return AdkDecisionTool(tool)


def tool_call_callback(check: ToolCallCheck) -> ToolCallCallback:
    """A `before_tool_callback` applying `check` to every tool call."""

    async def callback(
        tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        confirmation = tool_context.tool_confirmation
        if confirmation is not None:
            if confirmation.confirmed:
                return None
            return {"error": f"A person rejected the call to {tool.name}."}
        outcome = await check.acall(tool.name, args)
        if outcome.permission is Permission.ALLOW:
            return None
        if outcome.permission is Permission.DENY:
            return {"error": outcome.reason}
        tool_context.request_confirmation(hint=outcome.reason)
        tool_context.actions.skip_summarization = True
        return {"error": f"{outcome.reason} Waiting for confirmation."}

    return callback
