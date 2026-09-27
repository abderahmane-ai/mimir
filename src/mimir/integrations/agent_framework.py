"""Decision tools as Microsoft Agent Framework function tools, and tool-call checks as
function middleware.

`as_function_tool` gives a `FunctionTool` validated by the decision tool's argument model and
returning its result as JSON.

`ToolCallCheckMiddleware` decides each call a check applies to before it runs. A denied or an
escalated call is not run, and the model reads the check's reason instead of a result. The
framework's approval requests from middleware go through its private security layer, so an
escalation does not pause the run; give tools that must pause `approval_mode="always_require"`.
"""

from collections.abc import Awaitable, Callable

from agent_framework import FunctionInvocationContext, FunctionMiddleware, FunctionTool

from mimir.core.checks import Permission, ToolCallCheck
from mimir.core.context import ContextInput
from mimir.core.results import DecisionResult
from mimir.core.tools import DecisionTool, ToolArguments


def as_function_tool(tool: DecisionTool) -> FunctionTool:
    """The function tool of `tool`."""

    async def decide(context: ContextInput) -> DecisionResult:
        return await tool.acall(context)

    return FunctionTool(
        name=tool.name,
        description=tool.agent_description,
        func=decide,
        input_model=ToolArguments,
        result_parser=_json,
    )


def _json(result: DecisionResult) -> str:
    return result.model_dump_json()


class ToolCallCheckMiddleware(FunctionMiddleware):
    """Function middleware applying a `ToolCallCheck` before each call it covers."""

    def __init__(self, check: ToolCallCheck) -> None:
        self.check = check

    async def process(
        self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
    ) -> None:
        outcome = await self.check.acall(context.function.name, dict(context.arguments))
        if outcome.permission is Permission.ALLOW:
            await call_next()
            return
        context.result = outcome.reason
