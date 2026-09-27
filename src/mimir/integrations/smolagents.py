"""Decision tools as smolagents tools, for `ToolCallingAgent` and `CodeAgent`.

The tool takes one `context` input and returns the result as a JSON object described by its
`output_schema`, so a `CodeAgent` reads `result["status"]` and `result["answer"]`. smolagents
reports invalid arguments to the model as a tool error. Its callbacks run after each step,
not before a tool call, so tool-call checks are not adapted.
"""

import keyword
from typing import Any, ClassVar

from smolagents import Tool

from mimir.core.tools import DecisionTool

CONTEXT_INPUT: dict[str, str | list[str]] = {
    "type": ["string", "array", "object"],
    "description": (
        "A string (one passage), a list of strings (passages), a Context object, or "
        "{'state': <JSON>} for a structured state."
    ),
}


class SmolDecisionTool(Tool):
    """A smolagents tool answering with a `DecisionTool`, named and described after it."""

    output_type = "object"
    skip_forward_signature_validation: ClassVar[bool] = True

    def __init__(self, tool: DecisionTool) -> None:
        if not tool.name.isidentifier() or keyword.iskeyword(tool.name):
            message = f"smolagents tool names are Python identifiers; {tool.name!r} is not"
            raise ValueError(message)
        self.name = tool.name
        self.description = tool.agent_description
        self.inputs = {"context": dict(CONTEXT_INPUT)}
        self.output_schema = tool.output_schema
        self._tool = tool
        super().__init__()

    def forward(self, context: object) -> dict[str, Any]:
        return self._tool.call_with({"context": context}).model_dump(mode="json")


def as_smolagents_tool(tool: DecisionTool) -> SmolDecisionTool:
    """The smolagents tool of `tool`."""
    return SmolDecisionTool(tool)
