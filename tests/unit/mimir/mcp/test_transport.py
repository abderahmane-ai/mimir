import asyncio
from typing import Final

import httpx
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from starlette.applications import Starlette
from starlette.types import ASGIApp

from mimir.core.decisions import Choice
from mimir.mcp.server import create_server
from mimir.mcp.transport import MCP_PATH, streamable_http_app
from mimir.runtime.engine import Mimir
from mimir.server.auth import BearerKeys

TEXT = "my card was charged twice"
URL: Final = f"http://127.0.0.1:8000{MCP_PATH}"
INITIALIZE: Final = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}
ACCEPT: Final = {"Accept": "application/json, text/event-stream"}


def _app(engine: Mimir) -> tuple[Starlette, ASGIApp]:
    tool = engine.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    inner = streamable_http_app(
        create_server(engine, [tool]), host="127.0.0.1", max_body_bytes=4096
    )
    return inner, BearerKeys(inner, frozenset({"secret"}))


@pytest.mark.parametrize(("mode", "version"), [("auto", "2026-07-28"), ("legacy", "2025-11-25")])
def test_clients_call_tools_over_http(engine: Mimir, mode: str, version: str) -> None:
    inner, app = _app(engine)

    async def main() -> tuple[str, object]:
        transport = httpx2.ASGITransport(app=app)
        async with (
            inner.router.lifespan_context(inner),
            httpx2.AsyncClient(
                transport=transport, headers={"Authorization": "Bearer secret"}
            ) as http,
            Client(streamable_http_client(URL, http_client=http), mode=mode) as client,
        ):
            result = await client.call_tool("route_ticket", {"context": TEXT})
            return client.protocol_version, result.structured_content

    protocol, content = asyncio.run(main())
    assert protocol == version
    assert isinstance(content, dict)
    assert content["type"] == "choice"


def test_responses_are_json_and_need_a_key(engine: Mimir) -> None:
    inner, app = _app(engine)

    async def main() -> tuple[httpx.Response, httpx.Response]:
        async with (
            inner.router.lifespan_context(inner),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client,
        ):
            refused = await client.post(URL, json=INITIALIZE, headers=ACCEPT)
            accepted = await client.post(
                URL, json=INITIALIZE, headers={**ACCEPT, "Authorization": "Bearer secret"}
            )
            return refused, accepted

    refused, accepted = asyncio.run(main())
    assert refused.status_code == 401
    assert accepted.status_code == 200
    assert accepted.headers["content-type"].startswith("application/json")
    assert accepted.json()["result"]["protocolVersion"] == "2025-11-25"
    assert "mcp-session-id" not in accepted.headers


def test_a_loopback_server_refuses_foreign_hosts_and_large_bodies(engine: Mimir) -> None:
    inner, _ = _app(engine)

    async def main() -> tuple[httpx.Response, httpx.Response]:
        async with (
            inner.router.lifespan_context(inner),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=inner)) as client,
        ):
            foreign = await client.post(
                URL, json=INITIALIZE, headers={**ACCEPT, "Host": "evil.example"}
            )
            large = await client.post(
                URL, json={**INITIALIZE, "padding": "x" * 5000}, headers=ACCEPT
            )
            return foreign, large

    foreign, large = asyncio.run(main())
    assert foreign.status_code == 421
    assert large.status_code == 413
