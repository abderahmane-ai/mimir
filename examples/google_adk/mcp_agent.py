"""A Google ADK agent using MIMIR's MCP server over stdio, through `McpToolset`.

The server runs `examples/tools.yaml`, so the agent sees `route_ticket` and passes only the
ticket. Install `google-adk[mcp]` and `uv`, set `GOOGLE_API_KEY`, then run
`make example NAME=google_adk/mcp_agent`.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.genai import types
from mcp import StdioServerParameters

MODEL: Final = "gemini-3.5-flash"
TOOLS: Final = Path(__file__).parents[1] / "tools.yaml"
SERVER: Final = ("uvx", ("--from", "mimir-decisions[local,mcp]", "mimir-decisions", "mcp", "--tools", str(TOOLS)))
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


def mimir_server(command: str, arguments: Sequence[str]) -> McpToolset:
    """MIMIR's MCP server, started as `command` with `arguments`."""
    parameters = StdioServerParameters(command=command, args=list(arguments))
    return McpToolset(connection_params=StdioConnectionParams(server_params=parameters, timeout=90))


def build_agent(server: McpToolset, model: str | BaseLlm = MODEL) -> LlmAgent:
    """The support agent, with the tools of `server`."""
    return LlmAgent(name="support", model=model, instruction=INSTRUCTIONS, tools=[server])


async def ask(agent: LlmAgent, text: str) -> str:
    """The agent's final reply to `text`, in a new in-memory session."""
    runner = InMemoryRunner(agent=agent, app_name="support")
    session = await runner.session_service.create_session(app_name="support", user_id="customer")
    message = types.Content(role="user", parts=[types.Part(text=text)])
    reply = ""
    async for event in runner.run_async(
        user_id="customer", session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            reply = "".join(part.text or "" for part in event.content.parts)
    return reply


async def main() -> None:
    server = mimir_server(*SERVER)
    try:
        print(await ask(build_agent(server), "I was charged twice for order 4412."))
    finally:
        await server.close()


if __name__ == "__main__":
    asyncio.run(main())
