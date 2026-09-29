"""`mimir serve` and `mimir mcp` as real processes on the test release."""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import httpx2
import pytest
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from mimir.client.http import MimirClient
from mimir.core.decisions import Choice

pytestmark = pytest.mark.integration

MIMIR: Final = Path(sys.executable).parent / "mimir"
TEXT: Final = "my card was charged twice"
TOOLS: Final = """\
tools:
  - name: route_ticket
    description: Route a support ticket.
    decision: {type: choice, question: which team, options: [billing, security]}
"""
START_TIMEOUT_S: Final = 60.0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def _tools(tmp_path: Path) -> Path:
    path = tmp_path / "tools.yaml"
    path.write_text(TOOLS, encoding="utf-8")
    return path


@contextmanager
def _process(*arguments: str, keys: str = "") -> Iterator[subprocess.Popen[bytes]]:
    environment = {**os.environ, "MIMIR_API_KEYS": keys}
    process = subprocess.Popen([str(MIMIR), *arguments], env=environment)
    try:
        yield process
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _wait_ready(url: str, headers: dict[str, str] | None = None) -> None:
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/readyz", headers=headers).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    pytest.fail(f"{url} was not ready within {START_TIMEOUT_S} s")


async def _call_over_http(url: str, name: str, arguments: dict[str, str]) -> object:
    """Call a tool with the secret key, retrying while the model loads."""
    for _ in range(300):
        try:
            async with (
                httpx2.AsyncClient(headers={"Authorization": "Bearer secret"}) as http,
                Client(streamable_http_client(url, http_client=http)) as client,
            ):
                found = await client.call_tool(name, arguments)
        except (httpx2.TransportError, TimeoutError, ExceptionGroup):
            await asyncio.sleep(1.0)
            continue
        if not found.is_error:
            return found.structured_content
        await asyncio.sleep(0.1)
    pytest.fail(f"{url} never answered {name}")


def _model(release: Path) -> tuple[str, ...]:
    return ("--model", str(release), "--allow-unsigned", "--device", "cpu")


def test_serve_answers_the_http_api_and_mcp_on_one_port(hub_release: Path, tmp_path: Path) -> None:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    arguments = ("serve", "--port", str(port), "--tools", str(_tools(tmp_path)), "--mcp")
    with _process(*arguments, *_model(hub_release), keys="secret") as process:
        _wait_ready(url)
        with MimirClient(url, api_key="secret") as client:
            result = client.decide(TEXT, Choice("which team", ["billing", "security"]))
            assert result.type == "choice"
            assert result.actionable is True
            assert client.info().model == str(hub_release.resolve())
        tool = httpx.post(
            f"{url}/v1/tools/route_ticket",
            json={"context": TEXT},
            headers={"Authorization": "Bearer secret"},
        )
        assert tool.json()["answer"] == result.answer
        content = asyncio.run(_call_over_http(f"{url}/mcp", "route_ticket", {"context": TEXT}))
        assert isinstance(content, dict)
        assert content["answer"] == result.answer
    assert process.returncode == 0


def test_mcp_serves_stdio(hub_release: Path, tmp_path: Path) -> None:
    parameters = StdioServerParameters(
        command=str(MIMIR),
        args=["mcp", "--tools", str(_tools(tmp_path)), "--generic-tools", *_model(hub_release)],
    )

    async def main() -> tuple[list[str], object]:
        async with Client(parameters) as client:
            names = [tool.name for tool in (await client.list_tools()).tools]
            for _ in range(300):
                try:
                    result = await client.call_tool("route_ticket", {"context": TEXT})
                except (httpx.TransportError, TimeoutError):
                    await asyncio.sleep(1.0)
                    continue
                if not result.is_error:
                    return names, result.structured_content
                await asyncio.sleep(0.1)
            pytest.fail("the stdio server never loaded its model")

    names, content = asyncio.run(main())
    assert names == ["route_ticket", "mimir_choose", "mimir_verify", "mimir_rank", "mimir_rate"]
    assert isinstance(content, dict)
    assert content["type"] == "choice"


def test_mcp_serves_http_behind_a_key(hub_release: Path) -> None:
    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    arguments = ("mcp", "--http", "--port", str(port), "--generic-tools", *_model(hub_release))
    with _process(*arguments, keys="secret") as process:
        deadline = time.monotonic() + START_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                refused = httpx.post(url, json={})
                break
            except httpx.TransportError:
                time.sleep(0.2)
        else:
            pytest.fail(f"{url} did not start")
        assert refused.status_code == 401
        verify_arguments = {"context": TEXT, "claim": "the card was charged twice"}
        content = asyncio.run(_call_over_http(url, "mimir_verify", verify_arguments))
        assert isinstance(content, dict)
        assert content["type"] == "verify"
    assert process.returncode == 0
