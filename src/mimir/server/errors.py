"""HTTP error responses. Every error is an `ErrorBody`.

| Status | `error.type` |
|---|---|
| 401 | `unauthorized` |
| 404 | `not_found` |
| 409 | `no_policy` |
| 413 | `too_large` |
| 422 | `invalid_request`, `input_limit`, `risk_level`, `invalid_context` |
| 500 | `internal` |
| 503 | `not_ready` |
"""

import logging
from collections.abc import Mapping
from typing import Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException

from mimir.core.errors import ContextError, InputLimitError, PolicyError, RiskLevelError
from mimir.core.wire import ErrorBody, ErrorDetail
from mimir.server.model import NotReadyError

NOT_READY: Final = 503
RETRY_AFTER_S: Final = 5
MAPPED: Final[tuple[tuple[type[Exception], int, str], ...]] = (
    (NotReadyError, NOT_READY, "not_ready"),
    (InputLimitError, 422, "input_limit"),
    (RiskLevelError, 422, "risk_level"),
    (ContextError, 422, "invalid_context"),
    (PolicyError, 409, "no_policy"),
    (ValidationError, 422, "invalid_request"),
)
HTTP_TYPES: Final = {404: "not_found", 405: "method_not_allowed"}

logger = logging.getLogger(__name__)


def error_response(
    status: int, kind: str, message: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    body = ErrorBody(error=ErrorDetail(type=kind, message=message))
    return JSONResponse(body.model_dump(mode="json"), status_code=status, headers=headers)


def _validation_message(error: RequestValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()
    )


async def _on_validation(_: Request, error: Exception) -> JSONResponse:
    if not isinstance(error, RequestValidationError):
        raise error
    return error_response(422, "invalid_request", _validation_message(error))


async def _on_http(_: Request, error: Exception) -> JSONResponse:
    if not isinstance(error, HTTPException):
        raise error
    kind = HTTP_TYPES.get(error.status_code, "http_error")
    return error_response(error.status_code, kind, str(error.detail), error.headers)


async def _on_mapped(_: Request, error: Exception) -> JSONResponse:
    for kind, status, name in MAPPED:
        if isinstance(error, kind):
            headers = {"Retry-After": str(RETRY_AFTER_S)} if status == NOT_READY else None
            return error_response(status, name, str(error), headers)
    raise error


async def _on_defect(request: Request, error: Exception) -> JSONResponse:
    logger.error(
        "request failed method=%s path=%s", request.method, request.url.path, exc_info=error
    )
    return error_response(500, "internal", "internal server error")


def install_error_handlers(app: FastAPI) -> None:
    """Answer every error with an `ErrorBody`."""
    app.add_exception_handler(RequestValidationError, _on_validation)
    app.add_exception_handler(HTTPException, _on_http)
    for kind, _, _ in MAPPED:
        app.add_exception_handler(kind, _on_mapped)
    app.add_exception_handler(Exception, _on_defect)
