"""The LlamaIndex MCP example against a real `mimir mcp` process on the test release."""

import asyncio
import json
from pathlib import Path

import pytest

from examples.llamaindex import mcp_agent
from mimir.core.results import ChoiceResult
from tests.conftest import MIMIR_SERVER, mcp_server_arguments
from tests.unit.mimir.integrations.test_llamaindex import _run, _script

pytestmark = pytest.mark.integration


def test_the_agent_routes_a_ticket_through_the_server(release: Path) -> None:
    tools = asyncio.run(mcp_agent.mimir_tools(MIMIR_SERVER, mcp_server_arguments(release)))
    assert [tool.metadata.name for tool in tools] == ["route_ticket"]
    llm = _script("route_ticket", {"context": "my card was charged twice"})
    _, [call] = _run(mcp_agent.build_agent(tools, llm))
    assert not call.tool_output.is_error
    [block] = call.tool_output.raw_output.content
    result = ChoiceResult.model_validate(json.loads(block.text))
    assert set(result.probabilities) == {"billing", "security", "shipping"}
