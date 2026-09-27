"""Named decision tools for agents.

A `DecisionTool` fixes the question and options, so the caller supplies only the context.
Framework adapters and the MCP server expose these tools.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, JsonValue

from mimir.core.context import ContextInput, ContextLike
from mimir.core.decisions import DecisionSpec
from mimir.core.results import RESULT_FOR_SPEC, DecisionResult

if TYPE_CHECKING:
    from mimir.core.decider import Decider

# The names OpenAI function tools and MCP tools both accept.
TOOL_NAME: Final = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ToolArguments(BaseModel):
    """Arguments of a decision tool call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    context: ContextInput


@dataclass(frozen=True, slots=True)
class DecisionTool:
    """A decision spec bound to a name. `tool(context)` returns its result."""

    name: str
    spec: DecisionSpec
    description: str
    decider: "Decider" = field(repr=False)
    risk: float
    alpha: float | None

    def __post_init__(self) -> None:
        if not TOOL_NAME.match(self.name):
            message = f"tool name {self.name!r} must match {TOOL_NAME.pattern}"
            raise ValueError(message)
        if not self.description.strip():
            message = f"tool {self.name!r} has a blank description"
            raise ValueError(message)

    def __call__(self, context: ContextLike) -> DecisionResult:
        return self.decider.decide(context, self.spec, risk=self.risk, alpha=self.alpha)

    async def acall(self, context: ContextLike) -> DecisionResult:
        return await self.decider.adecide(context, self.spec, risk=self.risk, alpha=self.alpha)

    @property
    def input_schema(self) -> dict[str, JsonValue]:
        """JSON Schema of the tool's arguments."""
        return ToolArguments.model_json_schema()

    @property
    def output_schema(self) -> dict[str, JsonValue]:
        """JSON Schema of the tool's result."""
        return RESULT_FOR_SPEC[self.spec.type].model_json_schema(mode="serialization")
