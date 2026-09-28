"""The HTTP API of `mimir serve`.

| Route | Does |
|---|---|
| `POST /v1/decide` | one certified decision |
| `POST /v1/decide/uncertified` | the model's raw answer |
| `POST /v1/decide/batch` | up to `max_batch_items` certified decisions, in order |
| `POST /v1/tools/{name}` | a configured `DecisionTool`, given only its context |
| `POST /v1/systemone` | Jev's request and response format |
| `GET /v1/models` | the loaded model, runtime and certified risk levels |
| `GET /healthz`, `GET /readyz` | liveness and readiness |
| `GET /metrics` | Prometheus metrics |
| `/mcp` | the MCP endpoint, when an MCP app is given |
"""

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from importlib import metadata
from typing import Final, Never

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, JsonValue
from starlette.applications import Starlette

from mimir.compat.systemone.v1 import SystemOneRequest, SystemOneResponse, respond, translate
from mimir.core.errors import InputLimitError
from mimir.core.results import DecisionResult
from mimir.core.tools import DecisionTool, ToolArguments
from mimir.core.wire import (
    MAX_BATCH_ITEMS,
    MAX_BODY_BYTES,
    BatchRequest,
    BatchResponse,
    DecideRequest,
    ErrorBody,
    ModelInfo,
    UncertifiedRequest,
)
from mimir.server.auth import BearerKeys
from mimir.server.errors import install_error_handlers
from mimir.server.limits import BodyLimit
from mimir.server.metrics import CONTENT_TYPE, Metrics, RequestMetrics
from mimir.server.model import LoadState, ServedModel

NOT_READY: Final = 503
ERROR_RESPONSES: Final[dict[int | str, dict[str, object]]] = {
    status: {"model": ErrorBody} for status in (401, 404, 409, 413, 422, 503)
}


class Probe(BaseModel):
    """Body of `GET /healthz` and `GET /readyz`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: LoadState
    error: str | None


def _probe(served: ServedModel, is_healthy: bool) -> JSONResponse:
    body = Probe(status=served.state, error=served.error)
    return JSONResponse(body.model_dump(mode="json"), status_code=200 if is_healthy else NOT_READY)


def create_app(
    served: ServedModel,
    *,
    metrics: Metrics,
    tools: Sequence[DecisionTool] = (),
    api_keys: frozenset[str] = frozenset(),
    max_body_bytes: int = MAX_BODY_BYTES,
    max_batch_items: int = MAX_BATCH_ITEMS,
    mcp: Starlette | None = None,
) -> FastAPI:
    """Build the server. `tools` are bound to `served`. The lifespan runs `served` and `mcp`."""
    by_name = {tool.name: tool for tool in tools}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(served.running())
            if mcp is not None:
                await stack.enter_async_context(mcp.router.lifespan_context(mcp))
            yield

    app = FastAPI(
        title="MIMIR",
        summary="Typed, calibrated and certified decisions.",
        version=metadata.version("mimir-decisions"),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        responses=ERROR_RESPONSES,
    )
    install_error_handlers(app)

    @app.post("/v1/decide")
    async def decide(request: DecideRequest) -> DecisionResult:
        return await served.adecide(
            request.context, request.decision, risk=request.risk, alpha=request.alpha
        )

    @app.post("/v1/decide/uncertified")
    async def decide_uncertified(request: UncertifiedRequest) -> DecisionResult:
        return await served.adecide_uncertified(request.context, request.decision)

    @app.post("/v1/decide/batch")
    async def decide_batch(request: BatchRequest) -> BatchResponse:
        if len(request.items) > max_batch_items:
            raise InputLimitError(
                limit="batch items", value=len(request.items), maximum=max_batch_items
            )
        items = [(item.context, item.decision) for item in request.items]
        results = await served.adecide_many(items, risk=request.risk, alpha=request.alpha)
        return BatchResponse(results=results)

    @app.post("/v1/tools/{name}")
    async def call_tool(name: str, arguments: ToolArguments) -> DecisionResult:
        tool = by_name.get(name)
        if tool is None:
            detail = f"no tool named {name!r}; this server has {sorted(by_name)}"
            raise HTTPException(status_code=404, detail=detail)
        return await tool.acall(arguments.context)

    @app.post("/v1/systemone")
    async def systemone(request: SystemOneRequest) -> SystemOneResponse:
        context, translated = translate(request)
        specs = [item.spec for item in translated.values()]
        results = await served.adecide_many([(context, spec) for spec in specs])
        engine = served.engine
        tokens = await asyncio.to_thread(
            lambda: sum(engine.count_tokens(context, spec) for spec in specs)
        )
        answered = dict(zip(translated, results, strict=True))
        return respond(translated, answered, served.info().model, tokens)

    @app.get("/v1/models")
    async def models() -> ModelInfo:
        return served.info()

    @app.get("/healthz", response_model=Probe, responses={NOT_READY: {"model": Probe}})
    async def healthz() -> JSONResponse:
        return _probe(served, is_healthy=served.state != "failed")

    @app.get("/readyz", response_model=Probe, responses={NOT_READY: {"model": Probe}})
    async def readyz() -> JSONResponse:
        return _probe(served, is_healthy=served.state == "ready")

    @app.get("/metrics", include_in_schema=False)
    async def prometheus() -> Response:
        return Response(metrics.exposition(), media_type=CONTENT_TYPE)

    if mcp is not None:
        app.router.routes.extend(mcp.routes)
    app.add_middleware(BodyLimit, max_bytes=max_body_bytes)
    app.add_middleware(BearerKeys, keys=api_keys)
    app.add_middleware(RequestMetrics, metrics=metrics)
    return app


def _no_model() -> Never:
    message = "the OpenAPI document is generated without a model"
    raise RuntimeError(message)


def openapi_document() -> dict[str, JsonValue]:
    """The server's OpenAPI 3.1 document, generated without loading a model."""
    app = create_app(ServedModel(_no_model), metrics=Metrics())
    document: dict[str, JsonValue] = app.openapi()
    return document
