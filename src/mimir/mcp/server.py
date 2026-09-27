"""Decision tools as MCP tools, on the SDK's `MCPServer`.

Configured `DecisionTool`s come first, then, with `with_generic_tools`, `mimir_choose`,
`mimir_verify`, `mimir_rank` and `mimir_rate`, which take the question and options as
arguments. Every tool is read-only, idempotent and closed-world, and returns its result as
structured content. A deferred decision is a normal result; invalid arguments and engine
failures are tool errors.
"""

from collections.abc import Awaitable, Callable, Sequence
from importlib import metadata
from typing import Final

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.tools import Tool
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from mimir.core.context import ContextInput
from mimir.core.decider import Decider
from mimir.core.decisions import Choice, Rank, Rate, Verify
from mimir.core.errors import MimirError
from mimir.core.results import (
    RESULT_FOR_SPEC,
    ChoiceResult,
    DecisionResult,
    RankResult,
    RateResult,
    VerifyResult,
)
from mimir.core.tools import DecisionTool, ToolArguments
from mimir.core.wire import DEFAULT_RISK

SERVER_NAME: Final = "mimir"
GUIDANCE: Final = (
    "Act on `answer` only when `status` is `decided` or `abstained`; when it is `deferred`, "
    "escalate to a person and pass on `deferral.reason`."
)
ANNOTATIONS: Final = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)
OptionsArgument = list[str] | dict[str, str]
Call = Callable[..., Awaitable[DecisionResult]]


class ConfiguredArguments(ToolArguments, ArgModelBase):
    """`ToolArguments`, as the SDK validates them."""


class _GenericArguments(ArgModelBase):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context: ContextInput


class ChooseArguments(_GenericArguments):
    question: str = Field(description="The question to decide.")
    options: OptionsArgument = Field(
        description="Option texts, or a mapping from your option id to its text."
    )


class VerifyArguments(_GenericArguments):
    claim: str = Field(description="The statement to check against the context.")


class RankArguments(_GenericArguments):
    question: str = Field(description="What the candidates are ranked by.")
    candidates: OptionsArgument = Field(
        description="Candidate texts, or a mapping from your candidate id to its text."
    )


class RateArguments(_GenericArguments):
    question: str = Field(description="The question to rate.")
    levels: OptionsArgument = Field(
        description="Level texts from lowest to highest, or a mapping from level id to text."
    )


def _tool(
    name: str,
    description: str,
    call: Call,
    arguments: type[ArgModelBase],
    result: type[BaseModel],
    schemas: tuple[dict[str, JsonValue], dict[str, JsonValue]],
) -> Tool:
    async def answer(**values: object) -> DecisionResult:
        try:
            return await call(**values)
        except (MimirError, ValidationError) as error:
            raise ToolError(str(error)) from error

    input_schema, output_schema = schemas
    return Tool(
        fn=answer,
        name=name,
        description=f"{description}\n\n{GUIDANCE}",
        parameters=input_schema,
        fn_metadata=FuncMetadata(
            arg_model=arguments, output_model=result, output_schema=output_schema
        ),
        is_async=True,
        annotations=ANNOTATIONS,
    )


def _schemas(
    arguments: type[BaseModel], result: type[BaseModel]
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    return arguments.model_json_schema(), result.model_json_schema(mode="serialization")


def configured_tool(tool: DecisionTool) -> Tool:
    """The MCP tool of a `DecisionTool`, with the same input and output schemas."""

    async def call(context: ContextInput) -> DecisionResult:
        return await tool.acall(context)

    result = RESULT_FOR_SPEC[tool.spec.type]
    schemas = (tool.input_schema, tool.output_schema)
    return _tool(tool.name, tool.description, call, ConfiguredArguments, result, schemas)


def generic_tools(decider: Decider, risk: float) -> list[Tool]:
    """`mimir_choose`, `mimir_verify`, `mimir_rank` and `mimir_rate`, decided at `risk`."""

    async def choose(
        context: ContextInput, question: str, options: OptionsArgument
    ) -> ChoiceResult:
        return await decider.adecide(context, Choice(question, options), risk=risk)

    async def verify(context: ContextInput, claim: str) -> VerifyResult:
        return await decider.adecide(context, Verify(claim), risk=risk)

    async def rank(context: ContextInput, question: str, candidates: OptionsArgument) -> RankResult:
        return await decider.adecide(context, Rank(question, candidates), risk=risk)

    async def rate(context: ContextInput, question: str, levels: OptionsArgument) -> RateResult:
        return await decider.adecide(context, Rate(question, levels), risk=risk)

    return [
        _tool(
            "mimir_choose",
            "Choose the option the context supports, or none of them.",
            choose,
            ChooseArguments,
            ChoiceResult,
            _schemas(ChooseArguments, ChoiceResult),
        ),
        _tool(
            "mimir_verify",
            "Check whether the context supports or contradicts a claim.",
            verify,
            VerifyArguments,
            VerifyResult,
            _schemas(VerifyArguments, VerifyResult),
        ),
        _tool(
            "mimir_rank",
            "Rank candidates from best to worst for a question, given the context.",
            rank,
            RankArguments,
            RankResult,
            _schemas(RankArguments, RankResult),
        ),
        _tool(
            "mimir_rate",
            "Rate the context on an ordered scale of levels.",
            rate,
            RateArguments,
            RateResult,
            _schemas(RateArguments, RateResult),
        ),
    ]


def create_server(
    decider: Decider,
    tools: Sequence[DecisionTool],
    *,
    with_generic_tools: bool = False,
    risk: float = DEFAULT_RISK,
) -> MCPServer:
    """An MCP server with `tools`, then the generic tools if asked. `risk` is the generic
    tools' risk level.

    Raises:
        ValueError: neither tools nor generic tools, or two tools with one name.
    """
    if not tools and not with_generic_tools:
        message = "an MCP server needs configured tools (--tools FILE) or --generic-tools"
        raise ValueError(message)
    listed = [configured_tool(tool) for tool in tools]
    if with_generic_tools:
        listed.extend(generic_tools(decider, risk))
    names = [tool.name for tool in listed]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        message = f"tool names {repeated} are listed more than once"
        raise ValueError(message)
    return MCPServer(
        SERVER_NAME,
        title="MIMIR",
        description="Typed, calibrated and certified decisions over a context.",
        version=metadata.version("mimirai"),
        tools=listed,
    )
