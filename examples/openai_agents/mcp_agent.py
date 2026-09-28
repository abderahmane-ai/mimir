"""An OpenAI Agents SDK agent using MIMIR's MCP server over stdio.

The server runs `examples/tools.yaml`, so the agent sees `route_ticket` and passes only the
ticket. Install `openai-agents` and `uv`, set `OPENAI_API_KEY`, then run
`make example NAME=openai_agents/mcp_agent`.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from agents import Agent, Model, Runner
from agents.mcp import MCPServer, MCPServerStdio

TOOLS: Final = Path(__file__).parents[1] / "tools.yaml"
SERVER: Final = ("uvx", ("--from", "mimir-decisions[local,mcp]", "mimir-decisions", "mcp", "--tools", str(TOOLS)))
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


def mimir_server(command: str, arguments: Sequence[str]) -> MCPServerStdio:
    """MIMIR's MCP server, started as `command` with `arguments`."""
    return MCPServerStdio(
        params={"command": command, "args": list(arguments)},
        name="mimir",
        client_session_timeout_seconds=90,
    )


def build_agent(server: MCPServer, model: str | Model | None = None) -> Agent[None]:
    """The support agent, with the tools of `server`; `model` defaults to the SDK's."""
    return Agent(name="support", instructions=INSTRUCTIONS, mcp_servers=[server], model=model)


async def main() -> None:
    async with mimir_server(*SERVER) as server:
        result = await Runner.run(build_agent(server), "I was charged twice for order 4412.")
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
