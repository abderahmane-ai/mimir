"""The Google ADK MCP example against a real `mimir mcp` process on the test release."""

import asyncio
import json
from pathlib import Path

import pytest
from google.genai import types

from examples.google_adk import mcp_agent
from mimir.core.results import ChoiceResult
from tests.conftest import MIMIR_SERVER, mcp_server_arguments
from tests.unit.mimir.integrations.test_google_adk import ScriptedLlm, Session, _call, _responses

pytestmark = pytest.mark.integration


def test_the_agent_routes_a_ticket_through_the_server(hub_release: Path) -> None:
    llm = ScriptedLlm(
        model="scripted",
        replies=[
            _call("route_ticket", {"context": "my card was charged twice"}),
            types.Part(text="done"),
        ],
    )
    server = mcp_agent.mimir_server(MIMIR_SERVER, mcp_server_arguments(hub_release))
    try:
        events = Session(mcp_agent.build_agent(server, llm)).send(types.Part(text="Route."))
    finally:
        asyncio.run(server.close())
    [response] = _responses(events, "route_ticket")
    structured = response.get("structuredContent") or json.loads(response["content"][0]["text"])
    result = ChoiceResult.model_validate(structured)
    assert set(result.probabilities) == {"billing", "security", "shipping"}
