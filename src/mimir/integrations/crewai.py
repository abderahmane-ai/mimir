"""Decision tools as CrewAI tools, and tool-call checks as a before-tool-call hook.

`as_crewai_tool` gives a `BaseTool` taking the decision tool's arguments and returning its
typed result, declared as the tool's `result_schema`.

`tool_call_hook` gives a hook for `register_before_tool_call_hook`. CrewAI hooks can only let
a call run or block it, and a blocked call reads to the model as CrewAI's own blocked
message. A denied call is blocked. An escalated call goes to `approve`, which returns whether
a person allowed it (for a console, `context.request_human_input`); without `approve` it is
blocked.
"""

from collections.abc import Callable

from crewai.hooks import ToolCallHookContext
from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr

from mimir.core.checks import CheckOutcome, Permission, ToolCallCheck
from mimir.core.results import RESULT_FOR_SPEC, DecisionResult
from mimir.core.tools import DecisionTool, ToolArguments

Approver = Callable[[ToolCallHookContext, CheckOutcome], bool]
ToolCallHook = Callable[[ToolCallHookContext], bool | None]


class CrewDecisionTool(BaseTool):
    """A CrewAI tool answering with a `DecisionTool`, named and described after it."""

    args_schema: type[BaseModel] = ToolArguments
    _decision_tool: DecisionTool = PrivateAttr()

    def __init__(self, *, decision_tool: DecisionTool) -> None:
        super().__init__(
            name=decision_tool.name,
            description=decision_tool.agent_description,
            result_schema=RESULT_FOR_SPEC[decision_tool.spec.type],
        )
        self._decision_tool = decision_tool

    def _run(self, context: object) -> DecisionResult:
        return self._decision_tool.call_with({"context": context})

    async def _arun(self, context: object) -> DecisionResult:
        return await self._decision_tool.acall_with({"context": context})


def as_crewai_tool(tool: DecisionTool) -> CrewDecisionTool:
    """The CrewAI tool of `tool`."""
    return CrewDecisionTool(decision_tool=tool)


def tool_call_hook(check: ToolCallCheck, *, approve: Approver | None = None) -> ToolCallHook:
    """A before-tool-call hook applying `check`; escalations go to `approve`."""

    def hook(context: ToolCallHookContext) -> bool | None:
        outcome = check(context.tool_name, context.tool_input)
        if outcome.permission is Permission.ALLOW:
            return None
        if outcome.permission is Permission.ESCALATE and approve is not None:
            return None if approve(context, outcome) else False
        return False

    return hook
