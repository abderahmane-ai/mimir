"""The Agent Framework MCP example, on the MCP SDK 1, against a real `mimir mcp` process."""

import asyncio
import json
from pathlib import Path

import pytest
from agent_framework import AgentResponse

from examples.agent_framework import mcp_agent
from mimir.core.results import ChoiceResult
from tests.conftest import MIMIR_SERVER, mcp_server_arguments
from tests.unit.mimir.integrations.test_agent_framework import ScriptedChatClient

pytestmark = pytest.mark.integration


def test_the_agent_routes_a_ticket_through_the_server(hub_release: Path) -> None:
    client = ScriptedChatClient("route_ticket", {"context": "my card was charged twice"})

    async def main() -> AgentResponse:
        async with mcp_agent.mimir_server(
            MIMIR_SERVER, mcp_server_arguments(hub_release)
        ) as server:
            return await mcp_agent.build_agent(server, client).run("Route.")

    result = ChoiceResult.model_validate(json.loads(asyncio.run(main()).text))
    assert set(result.probabilities) == {"billing", "security", "shipping"}
