"""The OpenAI Agents SDK MCP example against a real `mimir mcp` process on the test release."""

import asyncio
import json
from pathlib import Path
from typing import Final

import pytest
from agents import RunConfig, Runner, RunResult
from agents.testing import ScriptedModel, assistant_message
from agents.testing import function_call as scripted_call

from examples.openai_agents import mcp_agent
from mimir.core.results import ChoiceResult
from tests.conftest import MIMIR_SERVER, mcp_server_arguments

pytestmark = pytest.mark.integration

TEXT: Final = "my card was charged twice"


def test_the_agent_routes_a_ticket_through_the_server(release: Path) -> None:
    model = ScriptedModel(
        [
            [scripted_call("route_ticket", {"context": TEXT}, call_id="call-1")],
            [assistant_message("done")],
        ]
    )

    async def main() -> RunResult:
        async with mcp_agent.mimir_server(MIMIR_SERVER, mcp_server_arguments(release)) as server:
            agent = mcp_agent.build_agent(server, model)
            return await Runner.run(agent, "Route.", run_config=RunConfig(tracing_disabled=True))

    assert asyncio.run(main()).final_output == "done"
    items = model.calls[1].input
    assert isinstance(items, list)
    [output] = [item["output"] for item in items if item.get("type") == "function_call_output"]
    assert isinstance(output, list)
    [content] = output
    result = ChoiceResult.model_validate(json.loads(content["text"]))
    assert set(result.probabilities) == {"billing", "security", "shipping"}
