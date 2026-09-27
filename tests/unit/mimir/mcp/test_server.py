import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import httpx
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from pydantic import JsonValue

from mimir.client.http import MimirClient
from mimir.core.decider import Decider
from mimir.core.decisions import Choice, Estimate, Rate, YesNo
from mimir.core.tools import DecisionTool
from mimir.mcp.server import GUIDANCE, create_server
from mimir.runtime.engine import Mimir
from mimir.server.app import create_app
from mimir.server.metrics import Metrics
from mimir.server.model import ServedModel
from tests.conftest import load_engine

T = TypeVar("T")
TEXT = "my card was charged twice"
TEAM = Choice("which team", ["billing", "security", "shipping"])


def _tools(decider: Decider) -> list[DecisionTool]:
    return [
        decider.tool("route_ticket", TEAM, "Route a support ticket."),
        decider.tool("estimate_price", Estimate("price", 0, 100), "Estimate the price."),
    ]


def _connected(server: MCPServer, work: Callable[[Client], Awaitable[T]], mode: str = "auto") -> T:
    async def main() -> T:
        async with Client(server, mode=mode) as client:
            return await work(client)

    return asyncio.run(main())


def _without_latency(body: object) -> dict[str, object]:
    assert isinstance(body, dict)
    return {key: value for key, value in body.items() if key != "latency_ms"}


def test_tools_are_listed_in_order_with_the_contract_schemas(engine: Mimir) -> None:
    tools = _tools(engine)
    server = create_server(engine, tools, with_generic_tools=True)
    listed = _connected(server, lambda client: client.list_tools()).tools
    assert [tool.name for tool in listed] == [
        "route_ticket",
        "estimate_price",
        "mimir_choose",
        "mimir_verify",
        "mimir_rank",
        "mimir_rate",
    ]
    for decision_tool, mcp_tool in zip(tools, listed, strict=False):
        assert mcp_tool.input_schema == decision_tool.input_schema
        assert mcp_tool.output_schema == decision_tool.output_schema
        assert mcp_tool.description == f"{decision_tool.description}\n\n{GUIDANCE}"
    for tool in listed:
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.idempotent_hint is True
        assert tool.annotations.open_world_hint is False
        assert tool.annotations.destructive_hint is False


@pytest.mark.parametrize(("mode", "version"), [("auto", "2026-07-28"), ("legacy", "2025-11-25")])
def test_a_configured_tool_answers_as_the_engine(engine: Mimir, mode: str, version: str) -> None:
    server = create_server(engine, _tools(engine))

    async def work(client: Client) -> tuple[str, object]:
        result = await client.call_tool("route_ticket", {"context": TEXT})
        assert result.is_error is False
        return client.protocol_version, result.structured_content

    protocol, content = _connected(server, work, mode)
    assert protocol == version
    expected = engine.decide(TEXT, TEAM).model_dump(mode="json")
    assert _without_latency(content) == _without_latency(expected)


def test_a_deferral_is_a_normal_result(engine: Mimir) -> None:
    server = create_server(engine, _tools(engine))
    result = _connected(
        server, lambda client: client.call_tool("estimate_price", {"context": TEXT})
    )
    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["status"] == "deferred"
    assert result.structured_content["deferral"]["reason"] == "no_certified_threshold"


def test_generic_tools_build_their_specs_from_the_arguments(engine: Mimir) -> None:
    server = create_server(engine, [], with_generic_tools=True)
    levels = ["low", "medium", "high"]

    async def work(client: Client) -> list[object]:
        calls: list[tuple[str, dict[str, JsonValue]]] = [
            ("mimir_choose", {"question": "which team", "options": ["billing", "security"]}),
            ("mimir_verify", {"claim": "the card was charged twice"}),
            ("mimir_rank", {"question": "best", "candidates": {"r": "refund", "o": "order"}}),
            ("mimir_rate", {"question": "rate", "levels": list[JsonValue](levels)}),
        ]
        results = [
            await client.call_tool(name, {"context": TEXT, **arguments})
            for name, arguments in calls
        ]
        assert all(result.is_error is False for result in results)
        return [result.structured_content for result in results]

    choose, verify, rank, rate = _connected(server, work)
    assert _without_latency(rate) == _without_latency(
        engine.decide(TEXT, Rate("rate", levels)).model_dump(mode="json")
    )
    assert isinstance(choose, dict)
    assert isinstance(verify, dict)
    assert isinstance(rank, dict)
    assert (choose["type"], verify["type"], rank["type"]) == ("choice", "verify", "rank")
    assert sorted(rank["answer"]) == ["o", "r"]


def test_invalid_arguments_are_tool_errors_naming_the_field(engine: Mimir) -> None:
    server = create_server(engine, _tools(engine), with_generic_tools=True)

    async def work(client: Client) -> list[str]:
        results = [
            await client.call_tool("route_ticket", {"context": TEXT, "question": "q"}),
            await client.call_tool("route_ticket", {}),
            await client.call_tool(
                "mimir_choose", {"context": TEXT, "question": "q", "options": ["only"]}
            ),
        ]
        assert all(result.is_error for result in results)
        return [str(getattr(result.content[0], "text", "")) for result in results]

    extra, missing, spec = _connected(server, work)
    assert "question\n  Extra inputs are not permitted" in extra
    assert "context\n  Field required" in missing
    assert "options needs at least 2 entries; got 1: ['only']" in spec


def test_a_loading_model_is_a_tool_error(release: Path) -> None:
    served = ServedModel(lambda: load_engine(release))
    server = create_server(served, _tools(served))
    result = _connected(server, lambda client: client.call_tool("route_ticket", {"context": TEXT}))
    assert result.is_error is True
    assert "the model is still loading" in str(getattr(result.content[0], "text", ""))


def test_remote_tools_forward_to_a_mimir_server(release: Path, engine: Mimir) -> None:
    served = ServedModel(lambda: load_engine(release))
    app = create_app(served, metrics=Metrics())

    async def main() -> object:
        async with app.router.lifespan_context(app):
            while served.state == "loading":
                await asyncio.sleep(0.01)
            remote = MimirClient("http://mimir", async_transport=httpx.ASGITransport(app=app))
            server = create_server(remote, _tools(remote))
            async with Client(server) as client:
                result = await client.call_tool("route_ticket", {"context": TEXT})
            await remote.aclose()
            return result.structured_content

    content = asyncio.run(main())
    expected = engine.decide(TEXT, TEAM).model_dump(mode="json")
    assert _without_latency(content) == _without_latency(expected)


def test_a_server_needs_tools_with_distinct_names(engine: Mimir) -> None:
    with pytest.raises(ValueError, match="needs configured tools"):
        create_server(engine, [])
    clash = engine.tool("mimir_choose", YesNo("q"), "Clashes with a generic tool.")
    with pytest.raises(ValueError, match=r"\['mimir_choose'\] are listed more than once"):
        create_server(engine, [clash], with_generic_tools=True)
