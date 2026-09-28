"""A LlamaIndex `FunctionAgent` using MIMIR's MCP server over stdio.

The server runs `examples/tools.yaml`, so the agent sees `route_ticket` and passes only the
ticket. Install `llama-index-tools-mcp`, `llama-index-llms-openai` and `uv`, set
`OPENAI_API_KEY`, then run `make example NAME=llamaindex/mcp_agent`.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

MODEL: Final = "gpt-5.5"
TOOLS: Final = Path(__file__).parents[1] / "tools.yaml"
SERVER: Final = (
    "uvx",
    ("--from", "mimir-decisions[local,mcp]", "mimir-decisions", "mcp", "--tools", str(TOOLS)),
)
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


async def mimir_tools(command: str, arguments: Sequence[str]) -> list[FunctionTool]:
    """The tools of MIMIR's MCP server, started as `command` with `arguments`."""
    client = BasicMCPClient(command, args=list(arguments), timeout=90)
    return await McpToolSpec(client=client).to_tool_list_async()


def build_agent(tools: Sequence[FunctionTool], llm: FunctionCallingLLM) -> FunctionAgent:
    """The support agent, with the server's `tools`."""
    return FunctionAgent(tools=list(tools), llm=llm, system_prompt=INSTRUCTIONS)


async def main() -> None:
    from llama_index.llms.openai import OpenAI

    agent = build_agent(await mimir_tools(*SERVER), OpenAI(model=MODEL))
    print(await agent.run(user_msg="I was charged twice for order 4412."))


if __name__ == "__main__":
    asyncio.run(main())
