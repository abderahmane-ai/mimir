"""Decision tools as OpenAI Agents SDK function tools, and tool-call checks as approvals.

`as_function_tool` gives a `FunctionTool` taking the decision tool's arguments and returning
its result as JSON; invalid arguments go back to the model as a message. `guard` returns a
copy of any function tool that a `ToolCallCheck` gates: an escalated call pauses the run for
a person (`RunState.approve` or `RunState.reject`), and a denied call is rejected with the
check's reason, which the model reads.
"""

import copy
import json
from typing import Any

from agents import (
    FunctionTool,
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
)
from agents.tool_context import ToolContext

from mimir.core.checks import Permission, ToolCallCheck
from mimir.core.tools import ARGUMENT_ERRORS, DecisionTool


def as_function_tool(tool: DecisionTool) -> FunctionTool:
    """The function tool of `tool`, with its argument schema. The SDK takes only strict output
    schemas and a result's probabilities are an open mapping, so none is declared."""

    async def invoke(context: ToolContext[Any], arguments: str) -> str:
        try:
            result = await tool.acall_with(json.loads(arguments))
        except (json.JSONDecodeError, *ARGUMENT_ERRORS) as error:
            return f"Invalid arguments for {tool.name}: {error}"
        return result.model_dump_json()

    return FunctionTool(
        name=tool.name,
        description=tool.agent_description,
        params_json_schema=tool.input_schema,
        on_invoke_tool=invoke,
        strict_json_schema=False,
    )


def guard(tool: FunctionTool, check: ToolCallCheck) -> FunctionTool:
    """A copy of `tool` whose calls `check` allows, denies or escalates for approval. The
    check replaces the tool's own `needs_approval`; its guardrails run before the check's."""

    async def needs_approval(
        context: RunContextWrapper[Any], arguments: dict[str, Any], call_id: str
    ) -> bool:
        outcome = await check.acall(tool.name, arguments)
        return outcome.permission is Permission.ESCALATE

    async def reject_denied(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        outcome = await check.acall(tool.name, json.loads(data.context.tool_arguments))
        if outcome.permission is Permission.DENY:
            return ToolGuardrailFunctionOutput.reject_content(outcome.reason, outcome)
        return ToolGuardrailFunctionOutput.allow(outcome)

    guardrail: ToolInputGuardrail[Any] = ToolInputGuardrail(
        guardrail_function=reject_denied, name="tool_call_check"
    )
    guarded = copy.copy(tool)
    guarded.needs_approval = needs_approval
    guarded.tool_input_guardrails = [*(tool.tool_input_guardrails or []), guardrail]
    return guarded
