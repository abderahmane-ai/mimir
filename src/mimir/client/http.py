"""`MimirClient`, a `Decider` backed by a MIMIR HTTP server.

Requests are retried on connection errors, timeouts and HTTP 429, 502, 503, 504 and 529, with
exponential backoff that honours `Retry-After`. Other error statuses raise the matching
`ServerResponseError` subclass, which keeps the status and body.
"""

import asyncio
import os
import time
from collections.abc import Sequence
from http import HTTPStatus
from typing import Final, Self

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from mimir.core.context import Context
from mimir.core.decider import Decider
from mimir.core.decisions import DecisionSpec
from mimir.core.errors import (
    AuthenticationError,
    ClientError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    ServerConnectionError,
    ServerError,
    ServerResponseError,
    ServerTimeoutError,
)
from mimir.core.results import DecisionResult
from mimir.core.wire import (
    BatchItem,
    BatchRequest,
    BatchResponse,
    DecideRequest,
    ErrorBody,
    ModelInfo,
    UncertifiedRequest,
)

API_KEY_ENVIRONMENT: Final = "MIMIR_API_KEY"
RETRY_STATUSES: Final = frozenset({429, 502, 503, 504, 529})
ERROR_TYPES: Final[dict[int, type[ServerResponseError]]] = {
    HTTPStatus.BAD_REQUEST: InvalidRequestError,
    HTTPStatus.UNPROCESSABLE_ENTITY: InvalidRequestError,
    HTTPStatus.UNAUTHORIZED: AuthenticationError,
    HTTPStatus.FORBIDDEN: AuthenticationError,
    HTTPStatus.NOT_FOUND: NotFoundError,
    HTTPStatus.TOO_MANY_REQUESTS: RateLimitError,
}
MAX_BACKOFF_S: Final = 30.0
BODY_EXCERPT_CHARS: Final = 500
RESULT: Final = TypeAdapter[DecisionResult](DecisionResult)


def response_error(response: httpx.Response) -> ServerResponseError:
    """Map an error response to its exception, using the server's message when present."""
    body = response.text
    try:
        message = ErrorBody.model_validate_json(body).error.message
    except ValidationError:
        message = body[:BODY_EXCERPT_CHARS] or response.reason_phrase
    status = response.status_code
    kind = ERROR_TYPES.get(status)
    if kind is None:
        kind = ServerError if status >= HTTPStatus.INTERNAL_SERVER_ERROR else ServerResponseError
    return kind(status, body, message)


def retry_delay_s(response: httpx.Response | None, attempt: int, backoff_s: float) -> float:
    """Return the wait before retry `attempt` (0-based): `Retry-After` in seconds if the server
    sent one, else `backoff_s * 2**attempt`, capped at `MAX_BACKOFF_S`."""
    header: str | None = None if response is None else response.headers.get("retry-after")
    if header is not None:
        try:
            seconds = float(header)
        except ValueError:
            seconds = None
        if seconds is not None:
            return min(max(seconds, 0.0), MAX_BACKOFF_S)
    backoff: float = backoff_s * 2**attempt
    return min(backoff, MAX_BACKOFF_S)


def _chunks(items: Sequence[BatchItem], size: int | None) -> list[Sequence[BatchItem]]:
    step = len(items) if size is None else size
    return [items[start : start + step] for start in range(0, len(items), step)]


class MimirClient(Decider):
    """Client for a MIMIR server, with the same interface as the local engine.

    Args:
        base_url: Server URL, e.g. `https://mimir.internal`.
        api_key: Bearer token; defaults to `$MIMIR_API_KEY`.
        timeout_s: Timeout for each HTTP request.
        max_retries: Retries after the first attempt for retryable failures.
        backoff_s: Base delay of the exponential backoff.
        transport: Custom httpx transport, e.g. for testing.
        async_transport: Custom httpx async transport, e.g. for testing.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if max_retries < 0:
            message = f"max_retries must be at least 0; got {max_retries}"
            raise ValueError(message)
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENVIRONMENT)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._client = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout_s, transport=transport
        )
        self._async_client = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout_s, transport=async_transport
        )

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._async_client.aclose()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def _settle(
        self, request: str, attempt: int, outcome: httpx.Response | httpx.TransportError
    ) -> httpx.Response | None:
        """Return a successful response, None to retry, or raise the final error."""
        is_last = attempt == self._max_retries
        if isinstance(outcome, httpx.TransportError):
            if not is_last:
                return None
            message = f"{request}: {outcome}"
            if isinstance(outcome, httpx.TimeoutException):
                raise ServerTimeoutError(message) from outcome
            raise ServerConnectionError(message) from outcome
        if outcome.is_success:
            return outcome
        if outcome.status_code in RETRY_STATUSES and not is_last:
            return None
        raise response_error(outcome)

    def _send(self, method: str, path: str, body: BaseModel | None) -> httpx.Response:
        content = None if body is None else body.model_dump_json()
        headers = {} if body is None else {"Content-Type": "application/json"}
        for attempt in range(self._max_retries + 1):
            outcome: httpx.Response | httpx.TransportError
            try:
                outcome = self._client.request(method, path, content=content, headers=headers)
            except httpx.TransportError as error:
                outcome = error
            settled = self._settle(f"{method} {path}", attempt, outcome)
            if settled is not None:
                return settled
            response = outcome if isinstance(outcome, httpx.Response) else None
            time.sleep(retry_delay_s(response, attempt, self._backoff_s))
        message = f"{method} {path}: max_retries is negative"
        raise ClientError(message)

    async def _asend(self, method: str, path: str, body: BaseModel | None) -> httpx.Response:
        content = None if body is None else body.model_dump_json()
        headers = {} if body is None else {"Content-Type": "application/json"}
        for attempt in range(self._max_retries + 1):
            outcome: httpx.Response | httpx.TransportError
            try:
                outcome = await self._async_client.request(
                    method, path, content=content, headers=headers
                )
            except httpx.TransportError as error:
                outcome = error
            settled = self._settle(f"{method} {path}", attempt, outcome)
            if settled is not None:
                return settled
            response = outcome if isinstance(outcome, httpx.Response) else None
            await asyncio.sleep(retry_delay_s(response, attempt, self._backoff_s))
        message = f"{method} {path}: max_retries is negative"
        raise ClientError(message)

    def info(self) -> ModelInfo:
        return ModelInfo.model_validate_json(self._send("GET", "/v1/models", None).content)

    def _run(
        self,
        requests: Sequence[tuple[Context, DecisionSpec]],
        *,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        if risk is None:
            return [
                RESULT.validate_json(
                    self._send(
                        "POST",
                        "/v1/decide/uncertified",
                        UncertifiedRequest(context=context, decision=spec),
                    ).content
                )
                for context, spec in requests
            ]
        if len(requests) == 1:
            context, spec = requests[0]
            body = DecideRequest(context=context, decision=spec, risk=risk, alpha=alpha)
            return [RESULT.validate_json(self._send("POST", "/v1/decide", body).content)]
        items = [BatchItem(context=context, decision=spec) for context, spec in requests]
        results: list[DecisionResult] = []
        for chunk in _chunks(items, batch_size):
            body_batch = BatchRequest(items=list(chunk), risk=risk, alpha=alpha)
            response = self._send("POST", "/v1/decide/batch", body_batch)
            results.extend(BatchResponse.model_validate_json(response.content).results)
        return results

    async def _arun(
        self,
        requests: Sequence[tuple[Context, DecisionSpec]],
        *,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        if risk is None:
            responses = await asyncio.gather(
                *(
                    self._asend(
                        "POST",
                        "/v1/decide/uncertified",
                        UncertifiedRequest(context=context, decision=spec),
                    )
                    for context, spec in requests
                )
            )
            return [RESULT.validate_json(response.content) for response in responses]
        if len(requests) == 1:
            context, spec = requests[0]
            body = DecideRequest(context=context, decision=spec, risk=risk, alpha=alpha)
            response = await self._asend("POST", "/v1/decide", body)
            return [RESULT.validate_json(response.content)]
        items = [BatchItem(context=context, decision=spec) for context, spec in requests]
        results: list[DecisionResult] = []
        for chunk in _chunks(items, batch_size):
            body_batch = BatchRequest(items=list(chunk), risk=risk, alpha=alpha)
            batch_response = await self._asend("POST", "/v1/decide/batch", body_batch)
            results.extend(BatchResponse.model_validate_json(batch_response.content).results)
        return results
