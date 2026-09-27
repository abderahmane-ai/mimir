"""The PydanticAI MCP example against a real `mimir mcp` process on the test release."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from examples.pydantic_ai import mcp_agent
from mimir.core.results import ChoiceResult
from tests.conftest import MIMIR_SERVER, mcp_server_arguments
from tests.unit.mimir.integrations.test_pydantic_ai import _returns, _script

pytestmark = pytest.mark.integration


def test_the_agent_routes_a_ticket_through_the_server(release: Path) -> None:
    server = mcp_agent.mimir_server(MIMIR_SERVER, mcp_server_arguments(release))
    model = FunctionModel(_script("route_ticket", {"context": "my card was charged twice"}))
    agent = mcp_agent.build_agent(server, model)

    async def main() -> list[ToolReturnPart]:
        async with agent:
            result = await agent.run("Route.")
        return [
            part for part in _returns(result.all_messages()) if isinstance(part, ToolReturnPart)
        ]

    [returned] = asyncio.run(main())
    content = returned.content
    result = ChoiceResult.model_validate(
        content if isinstance(content, dict) else json.loads(str(content))
    )
    assert set(result.probabilities) == {"billing", "security", "shipping"}
