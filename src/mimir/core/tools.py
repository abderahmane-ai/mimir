"""Named decision tools for agents.

A `DecisionTool` fixes the question and options, so the caller supplies only the context.
Framework adapters and the MCP server expose these tools. `ToolDefinitions` declares them, as
a tools file does.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, JsonValue, StringConstraints, model_validator
from pydantic import Field as PydanticField

from mimir.core.context import ContextInput, ContextLike
from mimir.core.decisions import DecisionSpec
from mimir.core.results import RESULT_FOR_SPEC, DecisionResult
from mimir.core.wire import DEFAULT_RISK

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
        if not TOOL_NAME.fullmatch(self.name):
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


class ToolDefinition(BaseModel):
    """One declared tool: the arguments of `Decider.tool`."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: Annotated[str, StringConstraints(pattern=TOOL_NAME.pattern)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    decision: DecisionSpec
    risk: float = DEFAULT_RISK
    alpha: float | None = PydanticField(default=None, gt=0, lt=1)


class ToolDefinitions(BaseModel):
    """The tools a server exposes, in the order it lists them. Names are unique."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tools: tuple[ToolDefinition, ...] = PydanticField(min_length=1)

    @model_validator(mode="after")
    def _check_unique_names(self) -> Self:
        counts = Counter(tool.name for tool in self.tools)
        repeated = sorted(name for name, count in counts.items() if count > 1)
        if repeated:
            message = f"tool names {repeated} are declared more than once"
            raise ValueError(message)
        return self

    def bind(self, decider: "Decider") -> tuple[DecisionTool, ...]:
        """Return one `DecisionTool` per definition, answered by `decider`."""
        return tuple(
            decider.tool(
                item.name, item.decision, item.description, risk=item.risk, alpha=item.alpha
            )
            for item in self.tools
        )
