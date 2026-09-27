"""The LangChain MCP example against a real `mimir mcp` process on the test release."""

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from examples.langchain import mcp_agent
from mimir.core.results import ChoiceResult
from tests.conftest import MIMIR_SERVER, mcp_server_arguments

pytestmark = pytest.mark.integration

TEXT: Final = "my card was charged twice"


class ScriptedChatModel(GenericFakeChatModel):
    """A fake chat model that accepts tools and answers with its scripted messages."""

    def bind_tools(self, *_: object, **__: object) -> "ScriptedChatModel":
        return self


def test_the_agent_routes_a_ticket_through_the_server(release: Path) -> None:
    call = {"name": "route_ticket", "args": {"context": TEXT}, "id": "call-1"}
    replies: Iterator[AIMessage | str] = iter(
        [AIMessage(content="", tool_calls=[call]), AIMessage(content="done")]
    )
    model = ScriptedChatModel(messages=replies)

    async def main() -> dict[str, Any]:
        async with mcp_agent.mimir_server(MIMIR_SERVER, mcp_server_arguments(release)) as server:
            tools = await server.list_tools()
            assert [tool.name for tool in tools] == ["route_ticket"]
            agent = mcp_agent.build_agent(tools, model)
            state: dict[str, Any] = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Route."}]}
            )
            return state

    state = asyncio.run(main())
    [message] = [item for item in state["messages"] if isinstance(item, ToolMessage)]
    assert message.status == "success"
    assert isinstance(message.content, list)
    [content] = message.content
    assert isinstance(content, dict)
    result = ChoiceResult.model_validate(json.loads(content["text"]))
    assert set(result.probabilities) == {"billing", "security", "shipping"}
    assert state["messages"][-1].content == "done"
