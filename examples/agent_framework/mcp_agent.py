"""A Microsoft Agent Framework agent using MIMIR's MCP server over stdio, through
`MCPStdioTool`.

The server runs `examples/tools.yaml`, so the agent sees `route_ticket` and passes only the
ticket. Agent Framework's MCP client needs the MCP SDK 1, so the server runs in its own
environment through `uvx`. Install `agent-framework`, `mcp<2` and `uv`, set
`OPENAI_API_KEY`, then run `make example NAME=agent_framework/mcp_agent`.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from agent_framework import Agent, BaseChatClient, MCPStdioTool

MODEL: Final = "gpt-5.5"
TOOLS: Final = Path(__file__).parents[1] / "tools.yaml"
SERVER: Final = ("uvx", ("--from", "mimir-decisions[local,mcp]", "mimir-decisions", "mcp", "--tools", str(TOOLS)))
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


def mimir_server(command: str, arguments: Sequence[str]) -> MCPStdioTool:
    """MIMIR's MCP server, started as `command` with `arguments`."""
    return MCPStdioTool(name="mimir", command=command, args=list(arguments))


def build_agent(server: MCPStdioTool, client: BaseChatClient[Any]) -> Agent:
    """The support agent, with the tools of `server`."""
    return Agent(client=client, instructions=INSTRUCTIONS, tools=[server])


async def main() -> None:
    from agent_framework.openai import OpenAIChatClient

    async with mimir_server(*SERVER) as server:
        agent = build_agent(server, OpenAIChatClient(model=MODEL))
        print((await agent.run("I was charged twice for order 4412.")).text)


if __name__ == "__main__":
    asyncio.run(main())
