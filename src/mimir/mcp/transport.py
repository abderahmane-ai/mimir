"""The MCP server's Streamable HTTP app at `/mcp`: stateless, with JSON responses."""

from typing import Final

from mcp.server.mcpserver import MCPServer
from starlette.applications import Starlette

MCP_PATH: Final = "/mcp"


def streamable_http_app(server: MCPServer, *, host: str, max_body_bytes: int) -> Starlette:
    """The HTTP app of `server`. On a loopback `host`, foreign `Host` and `Origin` headers are
    refused."""
    return server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        json_response=True,
        stateless_http=True,
        max_request_body_size=max_body_bytes,
        host=host,
    )
