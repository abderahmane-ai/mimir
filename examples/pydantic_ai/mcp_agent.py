"""A PydanticAI agent using MIMIR's MCP server over stdio, through `MCPToolset`.

The server runs `examples/tools.yaml`, so the agent sees `route_ticket` and passes only the
ticket. Install `pydantic-ai-slim[mcp,openai]` and `uv`, set `OPENAI_API_KEY`, then run
`make example NAME=pydantic_ai/mcp_agent`.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models import Model

MODEL: Final = "openai:gpt-5.5"
TOOLS: Final = Path(__file__).parents[1] / "tools.yaml"
SERVER: Final = ("uvx", ("--from", "mimirai[local,mcp]", "mimirai", "mcp", "--tools", str(TOOLS)))
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


def mimir_server(command: str, arguments: Sequence[str]) -> MCPToolset[Any]:
    """MIMIR's MCP server, started as `command` with `arguments`."""
    return MCPToolset(StdioTransport(command=command, args=list(arguments)))


def build_agent(server: MCPToolset[Any], model: str | Model = MODEL) -> Agent[None, str]:
    """The support agent, with the tools of `server`."""
    return Agent(model, instructions=INSTRUCTIONS, toolsets=[server])


async def main() -> None:
    agent = build_agent(mimir_server(*SERVER))
    async with agent:
        result = await agent.run("I was charged twice for order 4412.")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
