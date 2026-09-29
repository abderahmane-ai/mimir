"""The CrewAI MCP example, on the MCP SDK 1, against a real `mimir mcp --http` process."""

import json
import socket
import subprocess
import time
from pathlib import Path

import pytest

from examples.crewai import mcp_agent
from mimir.core.results import ChoiceResult
from tests.conftest import MIMIR_SERVER, mcp_server_arguments
from tests.unit.mimir.integrations.test_crewai import FINAL, ScriptedLLM, _action

pytestmark = pytest.mark.integration

START_TIMEOUT_S = 60.0


@pytest.fixture(autouse=True)
def _no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def _wait_listening(port: int) -> None:
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    pytest.fail(f"mimir mcp --http did not listen on port {port} in {START_TIMEOUT_S} s")


def test_the_crew_routes_a_ticket_through_the_server(hub_release: Path) -> None:
    port = _free_port()
    arguments = [*mcp_server_arguments(hub_release), "--http", "--port", str(port)]
    process = subprocess.Popen([MIMIR_SERVER, *arguments])
    try:
        _wait_listening(port)
        tool = f"127_0_0_1_{port}_mcp_route_ticket"
        llm = ScriptedLLM(
            model="scripted", replies=[_action(tool, {"context": "charged twice"}), FINAL]
        )
        server = mcp_agent.mimir_server(f"http://127.0.0.1:{port}/mcp")
        crew = mcp_agent.build_crew(server, llm)
        assert str(crew.kickoff(inputs={"ticket": "charged twice"})) == "done"
    finally:
        process.terminate()
        process.wait(timeout=20)
    result = ChoiceResult.model_validate(json.loads(llm.observation(1)))
    assert set(result.probabilities) == {"billing", "security", "shipping"}
