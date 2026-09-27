"""Decision tools as a PydanticAI toolset, and tool-call checks as a toolset wrapper.

`as_toolset` gives a `FunctionToolset` whose tools take the decision tools' argument schemas
and return their typed results. Invalid arguments ask the model to retry.

`guard` wraps any toolset so a `ToolCallCheck` decides each call. A denied call fails with the
check's reason, which the model reads. An escalated call requires approval: the run ends with
`DeferredToolRequests` (declare it in the agent's `output_type`) whose `metadata` holds the
reason, and resumes with `DeferredToolResults` approving or denying the call.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import ApprovalRequired, ModelRetry, RunContext, Tool, ToolFailed
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from mimir.core.checks import Permission, ToolCallCheck
from mimir.core.results import DecisionResult
from mimir.core.tools import ARGUMENT_ERRORS, DecisionTool


def as_tool(tool: DecisionTool) -> Tool[Any]:
    """The PydanticAI tool of `tool`, with its argument schema and typed result."""

    async def decide(**arguments: object) -> DecisionResult:
        try:
            return await tool.acall_with(arguments)
        except ARGUMENT_ERRORS as error:
            raise ModelRetry(str(error)) from error

    return Tool.from_schema(
        decide, name=tool.name, description=tool.agent_description, json_schema=tool.input_schema
    )


def as_toolset(tools: Iterable[DecisionTool]) -> FunctionToolset[Any]:
    """A toolset of the PydanticAI tools of `tools`, in order."""
    return FunctionToolset([as_tool(tool) for tool in tools])


@dataclass
class CheckedToolset(WrapperToolset[Any]):
    """A toolset whose calls a `ToolCallCheck` allows, denies or sends for approval."""

    check: ToolCallCheck

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]
    ) -> object:
        if not ctx.tool_call_approved:
            outcome = await self.check.acall(name, tool_args)
            if outcome.permission is Permission.DENY:
                raise ToolFailed(outcome.reason)
            if outcome.permission is Permission.ESCALATE:
                raise ApprovalRequired(metadata={"reason": outcome.reason})
        return await super().call_tool(name, tool_args, ctx, tool)


def guard(toolset: AbstractToolset[Any], check: ToolCallCheck) -> CheckedToolset:
    """`toolset`, with every call decided by `check` first."""
    return CheckedToolset(wrapped=toolset, check=check)
