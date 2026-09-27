"""A LangChain agent using MIMIR's MCP server over stdio, through `langchain.mcp`.

The server runs `examples/tools.yaml`, so the agent sees `route_ticket` and passes only the
ticket. Install `langchain[mcp]`, `langchain-openai` and `uv`, set `OPENAI_API_KEY`, then run
`make example NAME=langchain/mcp_agent`.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from fastmcp.client.transports import StdioTransport
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState, InputAgentState, OutputAgentState
    from langgraph.graph.state import CompiledStateGraph

    Agent = CompiledStateGraph[AgentState[Any], None, InputAgentState, OutputAgentState[Any]]

MODEL: Final = "openai:gpt-5.5"
TOOLS: Final = Path(__file__).parents[1] / "tools.yaml"
SERVER: Final = ("uvx", ("--from", "mimirai[local,mcp]", "mimirai", "mcp", "--tools", str(TOOLS)))
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


def mimir_server(command: str, arguments: Sequence[str]) -> MCPAdapter:
    """MIMIR's MCP server, started as `command` with `arguments`."""
    return MCPAdapter(StdioTransport(command=command, args=list(arguments)))


def build_agent(tools: Sequence[BaseTool], model: str | BaseChatModel = MODEL) -> "Agent":
    """The support agent, with the server's `tools`."""
    return create_agent(model, tools=list(tools), system_prompt=INSTRUCTIONS)


async def main() -> None:
    async with mimir_server(*SERVER) as server:
        agent = build_agent(await server.list_tools())
        state = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "I was charged twice."}]}
        )
        print(state["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
