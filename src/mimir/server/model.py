"""`ServedModel`, the `Decider` behind `mimir serve` and `mimir mcp`.

The model loads in the background once `running()` is entered. Until it is loaded, sync
decisions raise `NotReadyError`, and async ones first wait up to `load_wait_s` for the load to
end. Async decisions go through a `Batcher`, sync ones straight to the engine.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Final, Literal

from mimir.core.context import Context
from mimir.core.decider import Decider
from mimir.core.decisions import DecisionSpec
from mimir.core.errors import MimirError
from mimir.core.results import DecisionResult
from mimir.core.wire import Mode, ModelInfo
from mimir.server.batcher import BATCH_WAIT_S, Batcher, BatchObserver

if TYPE_CHECKING:
    from mimir.runtime.engine import Mimir

LoadState = Literal["loading", "ready", "failed"]
# On CPU, requests sharing a graph run carried 1.26-1.40 times their tokens in padding and
# cut throughput from 1.66 to 1.30 requests/s (fp32, Apple M4, 2026-09-27), so each runs alone.
GRAPH_BATCH_SIZE: Final[Mapping[str, int]] = {"cpu": 1}

logger = logging.getLogger(__name__)


class NotReadyError(MimirError):
    """The served model is still loading, or failed to load."""


class ServedModel(Decider):
    """A model served while `running()` is entered.

    Args:
        load: Returns the loaded engine, e.g. a `Mimir.from_pretrained` call.
        batch_tokens: Encoder tokens that fill a batch; None uses the release's token budget.
        wait_s: The longest a request waits for a batch to fill.
        observer: Receives every batch's results.
        load_wait_s: The longest an async decision waits for a load still in progress.
    """

    def __init__(
        self,
        load: "Callable[[], Mimir]",
        *,
        batch_tokens: int | None = None,
        wait_s: float = BATCH_WAIT_S,
        observer: BatchObserver | None = None,
        load_wait_s: float = 0.0,
    ) -> None:
        self._load = load
        self._batch_tokens = batch_tokens
        self._wait_s = wait_s
        self._observer = observer
        self._load_wait_s = load_wait_s
        self._state: LoadState = "loading"
        self._error: str | None = None
        self._engine: Mimir | None = None
        self._batcher: Batcher | None = None
        self._settled = asyncio.Event()

    @property
    def state(self) -> LoadState:
        return self._state

    @property
    def error(self) -> str | None:
        """Why the load failed, or None."""
        return self._error

    @property
    def engine(self) -> "Mimir":
        """The loaded engine. Raises `NotReadyError` before it is loaded."""
        if self._engine is None:
            raise NotReadyError(self._not_ready())
        return self._engine

    def _not_ready(self) -> str:
        if self._state == "failed":
            return f"the model failed to load: {self._error}"
        return "the model is still loading; retry in a few seconds"

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        """Load the model in the background and serve it until exit."""
        tasks: list[asyncio.Task[None]] = []
        tasks.append(asyncio.create_task(self._start(tasks)))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task

    async def _start(self, tasks: list[asyncio.Task[None]]) -> None:
        started = time.perf_counter()
        try:
            engine = await asyncio.to_thread(self._load)
        except Exception as error:
            self._state, self._error = "failed", f"{type(error).__name__}: {error}"
            self._settled.set()
            logger.exception("model load failed error=%r", self._error)
            return
        info = engine.info()
        budget = self._batch_tokens or engine.snapshot.config.layout.crossing_tokens
        graph_batch_size = GRAPH_BATCH_SIZE.get(info.device)
        batcher = Batcher(
            engine,
            token_budget=budget,
            wait_s=self._wait_s,
            graph_batch_size=graph_batch_size,
            observer=self._observer,
        )
        tasks.append(asyncio.create_task(batcher.run()))
        self._engine, self._batcher, self._state = engine, batcher, "ready"
        self._settled.set()
        logger.info(
            "model ready model=%s revision=%s variant=%s device=%s certification=%s "
            "batch_tokens=%d graph_batch_size=%s seconds=%.1f",
            info.model,
            info.revision,
            info.variant,
            info.device,
            info.certification,
            budget,
            graph_batch_size,
            time.perf_counter() - started,
        )

    def info(self) -> ModelInfo:
        return self.engine.info()

    def _run(
        self,
        requests: Sequence[tuple[Context, DecisionSpec]],
        *,
        mode: Mode,
        min_confidence: float | None,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        engine = self.engine
        if risk is None:
            return engine.decide_uncertified_many(requests, batch_size=batch_size)
        return engine.decide_many(
            requests,
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
            batch_size=batch_size,
        )

    async def _arun(
        self,
        requests: Sequence[tuple[Context, DecisionSpec]],
        *,
        mode: Mode,
        min_confidence: float | None,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        """Async requests share batches across callers; `batch_size` is not applied."""
        del batch_size
        if self._batcher is None and self._load_wait_s > 0:
            with suppress(TimeoutError):
                await asyncio.wait_for(self._settled.wait(), self._load_wait_s)
        if self._batcher is None:
            raise NotReadyError(self._not_ready())
        return await self._batcher.submit(
            requests, mode=mode, min_confidence=min_confidence, risk=risk, alpha=alpha
        )
