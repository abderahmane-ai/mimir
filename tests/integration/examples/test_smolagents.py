"""The smolagents MCP example, on the MCP SDK 1, against a real `mimir mcp` process."""

import ast
from pathlib import Path

import pytest

from examples.smolagents import mcp_agent
from mimir.core.results import ChoiceResult
from tests.conftest import MIMIR_SERVER, mcp_server_arguments
from tests.unit.mimir.integrations.test_smolagents import ScriptedModel, _call, _observations

pytestmark = pytest.mark.integration


def test_the_agent_routes_a_ticket_through_the_server(release: Path) -> None:
    model = ScriptedModel(
        [
            _call("route_ticket", {"context": "my card was charged twice"}),
            _call("final_answer", {"answer": "done"}),
        ]
    )
    with mcp_agent.mimir_server(MIMIR_SERVER, mcp_server_arguments(release)) as tools:
        assert [tool.name for tool in tools] == ["route_ticket"]
        agent = mcp_agent.build_agent(tools, model)
        assert agent.run("Route.") == "done"
    result = ChoiceResult.model_validate(ast.literal_eval(_observations(agent)[0]))
    assert set(result.probabilities) == {"billing", "security", "shipping"}
