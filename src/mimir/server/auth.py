"""Bearer API keys for the HTTP server and the MCP HTTP transport.

Keys are read from `$MIMIR_API_KEYS`, comma-separated. With keys set, every path except
`/healthz` and `/readyz` requires `Authorization: Bearer <key>`.
"""

import hmac
import ipaddress
import os
from collections.abc import Mapping
from typing import Final

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from mimir.core.wire import ErrorBody, ErrorDetail

API_KEYS_ENVIRONMENT: Final = "MIMIR_API_KEYS"
OPEN_PATHS: Final = frozenset({"/healthz", "/readyz"})
LOOPBACK_NAMES: Final = frozenset({"localhost"})
UNAUTHORIZED: Final = 401


def api_keys_from_environment(environ: Mapping[str, str] = os.environ) -> frozenset[str]:
    """Return the keys in `$MIMIR_API_KEYS`, with surrounding spaces and empty entries dropped."""
    raw = environ.get(API_KEYS_ENVIRONMENT, "")
    return frozenset(key.strip() for key in raw.split(",") if key.strip())


def is_loopback(host: str) -> bool:
    """Whether `host` names only this machine: `localhost` or a loopback IP address."""
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def check_exposure(host: str, keys: frozenset[str], *, allow_no_auth: bool) -> None:
    """Raise `ValueError` for a server without keys on a non-loopback address."""
    if keys or allow_no_auth or is_loopback(host):
        return
    message = (
        f"--host {host} accepts connections from other machines, and {API_KEYS_ENVIRONMENT} "
        "holds no key: set it, or pass --allow-no-auth to serve without authentication"
    )
    raise ValueError(message)


def is_authorized(authorization: str | None, keys: frozenset[str]) -> bool:
    """Whether an `Authorization` header carries one of `keys` as a bearer token."""
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    presented = token.strip().encode()
    return any(hmac.compare_digest(presented, key.encode()) for key in keys)


class BearerKeys:
    """ASGI middleware answering 401 to requests without a valid key. Without keys, it passes
    every request."""

    def __init__(self, app: ASGIApp, keys: frozenset[str]) -> None:
        self.app = app
        self.keys = keys

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.keys or scope["path"] in OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        if is_authorized(Headers(scope=scope).get("authorization"), self.keys):
            await self.app(scope, receive, send)
            return
        body = ErrorBody(
            error=ErrorDetail(type="unauthorized", message="missing or invalid bearer API key")
        )
        response = JSONResponse(
            body.model_dump(mode="json"),
            status_code=UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)
