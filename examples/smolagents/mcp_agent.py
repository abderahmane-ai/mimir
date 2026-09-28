"""A smolagents `ToolCallingAgent` using MIMIR's MCP server over stdio, through `MCPClient`.

The server runs `examples/tools.yaml`, so the agent sees `route_ticket` and passes only the
ticket. The smolagents MCP client needs the MCP SDK 1, so the server runs in its own
environment through `uvx`. Install `smolagents[mcp]`, `mcp<2` and `uv`, set `HF_TOKEN`, then
run `make example NAME=smolagents/mcp_agent`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from mcp import StdioServerParameters
from smolagents import InferenceClientModel, MCPClient, Model, Tool, ToolCallingAgent

TOOLS: Final = Path(__file__).parents[1] / "tools.yaml"
SERVER: Final = (
    "uvx",
    ("--from", "mimir-decisions[local,mcp]", "mimir-decisions", "mcp", "--tools", str(TOOLS)),
)
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


def mimir_server(command: str, arguments: Sequence[str]) -> MCPClient:
    """MIMIR's MCP server, started as `command` with `arguments`."""
    parameters = StdioServerParameters(command=command, args=list(arguments))
    return MCPClient(parameters, structured_output=True)


def build_agent(tools: Sequence[Tool], model: Model) -> ToolCallingAgent:
    """The support agent, with the server's `tools`."""
    return ToolCallingAgent(tools=list(tools), model=model, instructions=INSTRUCTIONS)


def main() -> None:
    with mimir_server(*SERVER) as tools:
        agent = build_agent(tools, InferenceClientModel())
        print(agent.run("I was charged twice for order 4412."))


if __name__ == "__main__":
    main()
