"""Decision tools as LlamaIndex tools, for `FunctionAgent`, `ReActAgent` and `AgentWorkflow`.

The model reads the result as JSON and the typed result is the tool output's `raw_output`.
Invalid arguments give an error output the model reads. LlamaIndex has no hook that gates a
tool call before it runs, so tool-call checks are not adapted.
"""

from llama_index.core.tools import AsyncBaseTool, ToolMetadata, ToolOutput

from mimir.core.tools import ARGUMENT_ERRORS, DecisionTool, ToolArguments


class LlamaDecisionTool(AsyncBaseTool):
    """A LlamaIndex tool answering with a `DecisionTool`, named and described after it."""

    def __init__(self, tool: DecisionTool) -> None:
        self._tool = tool
        self._metadata = ToolMetadata(
            description=tool.agent_description, name=tool.name, fn_schema=ToolArguments
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, *args: object, **kwargs: object) -> ToolOutput:
        if args:
            return self._invalid(kwargs, f"takes keyword arguments only, got {args!r}")
        try:
            result = self._tool.call_with(kwargs)
        except ARGUMENT_ERRORS as error:
            return self._invalid(kwargs, str(error))
        return self._output(kwargs, result.model_dump_json(), result)

    async def acall(self, *args: object, **kwargs: object) -> ToolOutput:
        if args:
            return self._invalid(kwargs, f"takes keyword arguments only, got {args!r}")
        try:
            result = await self._tool.acall_with(kwargs)
        except ARGUMENT_ERRORS as error:
            return self._invalid(kwargs, str(error))
        return self._output(kwargs, result.model_dump_json(), result)

    def _output(self, arguments: dict[str, object], content: str, result: object) -> ToolOutput:
        return ToolOutput(
            tool_name=self._tool.name, content=content, raw_input=arguments, raw_output=result
        )

    def _invalid(self, arguments: dict[str, object], reason: str) -> ToolOutput:
        return ToolOutput(
            tool_name=self._tool.name,
            content=f"Invalid arguments for {self._tool.name}: {reason}",
            raw_input=arguments,
            is_error=True,
        )


def as_llamaindex_tool(tool: DecisionTool) -> LlamaDecisionTool:
    """The LlamaIndex tool of `tool`."""
    return LlamaDecisionTool(tool)
