"""The request body size limit. Larger bodies are refused with 413 before the app reads them."""

from typing import Final

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mimir.core.wire import MAX_BODY_BYTES, ErrorBody, ErrorDetail

BAD_REQUEST: Final = 400
TOO_LARGE: Final = 413


def _refusal(status: int, kind: str, message: str) -> JSONResponse:
    body = ErrorBody(error=ErrorDetail(type=kind, message=message))
    return JSONResponse(body.model_dump(mode="json"), status_code=status)


class BodyLimit:
    """ASGI middleware refusing HTTP request bodies over `max_bytes` with 413."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        if max_bytes < 1:
            message = f"max_bytes must be at least 1; got {max_bytes}"
            raise ValueError(message)
        self.app = app
        self.max_bytes = max_bytes

    def _too_large(self, size: str) -> JSONResponse:
        message = f"request body is {size} bytes; the limit is {self.max_bytes}"
        return _refusal(TOO_LARGE, "too_large", message)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            if not declared.isdigit():
                refusal = _refusal(BAD_REQUEST, "invalid_request", f"Content-Length {declared!r}")
                await refusal(scope, receive, send)
            elif int(declared) > self.max_bytes:
                await self._too_large(declared)(scope, receive, send)
            else:
                await self.app(scope, receive, send)
            return
        chunks: list[bytes] = []
        received = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return
            chunk: bytes = message.get("body", b"")
            received += len(chunk)
            if received > self.max_bytes:
                await self._too_large(f"at least {received}")(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        whole: Message = {"type": "http.request", "body": b"".join(chunks), "more_body": False}
        is_replayed = False

        async def replay() -> Message:
            nonlocal is_replayed
            if is_replayed:
                return await receive()
            is_replayed = True
            return whole

        await self.app(scope, replay, send)
