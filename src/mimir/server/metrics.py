"""Prometheus metrics of one server: requests, latency, batch sizes and decision statuses.

Requests are labelled by route template, e.g. `/v1/tools/{name}`, or `unmatched`.
"""

import time
from collections.abc import Sequence
from typing import Final

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mimir.core.results import DecisionResult

BATCH_BUCKETS: Final = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
CONTENT_TYPE: Final = CONTENT_TYPE_LATEST
UNMATCHED: Final = "unmatched"


class Metrics:
    """The server's counters and histograms. Implements `mimir.server.batcher.BatchObserver`."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "mimir_requests",
            "HTTP requests by route template and status code.",
            ["route", "code"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "mimir_request_seconds",
            "HTTP request latency by route template.",
            ["route"],
            registry=self.registry,
        )
        self.batch_requests = Histogram(
            "mimir_batch_requests",
            "Requests per engine batch.",
            buckets=BATCH_BUCKETS,
            registry=self.registry,
        )
        self.decisions = Counter(
            "mimir_decisions",
            "Decisions by spec type and status.",
            ["type", "status"],
            registry=self.registry,
        )

    def observe_request(self, route: str, code: int, seconds: float) -> None:
        self.requests.labels(route=route, code=str(code)).inc()
        self.latency.labels(route=route).observe(seconds)

    def observe_batch(self, results: Sequence[DecisionResult]) -> None:
        self.batch_requests.observe(len(results))
        for result in results:
            self.decisions.labels(type=result.type, status=result.status.value).inc()

    def exposition(self) -> bytes:
        """The registry in the Prometheus text format, served as `CONTENT_TYPE`."""
        return generate_latest(self.registry)


class RequestMetrics:
    """ASGI middleware recording every HTTP request's route, status code and latency."""

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        code = 500

        async def tracked(message: Message) -> None:
            nonlocal code
            if message["type"] == "http.response.start":
                code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, tracked)
        finally:
            route = scope.get("route")
            template = route.path if isinstance(route, Route | Mount) else UNMATCHED
            self.metrics.observe_request(template, code, time.perf_counter() - started)
