"""`Batcher`, which gathers concurrent requests into engine calls.

Requests are grouped by mode, floor, risk and alpha. A group runs as one engine call when it
holds `token_budget` encoder tokens or its oldest request has waited `wait_s`. One engine
call runs at a time, and `graph_batch_size` caps the requests in each model run within it.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol

from mimir.core.context import Context
from mimir.core.decisions import DecisionSpec
from mimir.core.results import DecisionResult
from mimir.core.wire import Mode

if TYPE_CHECKING:
    from mimir.runtime.engine import Mimir

BATCH_WAIT_S: Final = 0.005

Requests = Sequence[tuple[Context, DecisionSpec]]
GroupKey = tuple[Mode, float | None, float | None, float | None]


class BatchObserver(Protocol):
    def observe_batch(self, results: Sequence[DecisionResult]) -> None: ...


@dataclass(slots=True)
class _Group:
    deadline: float
    requests: list[tuple[Context, DecisionSpec]] = field(default_factory=list)
    futures: list[asyncio.Future[DecisionResult]] = field(default_factory=list)
    tokens: int = 0


class Batcher:
    """Gathers `submit` calls into engine calls. Results arrive only while `run` runs."""

    def __init__(
        self,
        engine: "Mimir",
        *,
        token_budget: int,
        wait_s: float = BATCH_WAIT_S,
        graph_batch_size: int | None = None,
        observer: BatchObserver | None = None,
    ) -> None:
        if token_budget < 1 or wait_s < 0:
            message = (
                "token_budget must be at least 1 and wait_s at least 0; "
                f"got {token_budget} and {wait_s}"
            )
            raise ValueError(message)
        self._engine = engine
        self._token_budget = token_budget
        self._wait_s = wait_s
        self._graph_batch_size = graph_batch_size
        self._observer = observer
        self._groups: dict[GroupKey, _Group] = {}
        self._wake = asyncio.Event()

    async def submit(
        self,
        requests: Requests,
        *,
        mode: Mode,
        min_confidence: float | None,
        risk: float | None,
        alpha: float | None,
    ) -> list[DecisionResult]:
        """Queue requests and return their results in order. `risk=None` skips the policy.

        Raises what the engine raises for these requests: `InputLimitError` and `ContextError`
        before queueing, and any engine failure of the batch they ran in.
        """
        tokens = await asyncio.to_thread(self._count_tokens, requests)
        loop = asyncio.get_running_loop()
        key: GroupKey = (mode, min_confidence, risk, alpha)
        group = self._groups.get(key)
        if group is None:
            group = self._groups[key] = _Group(deadline=loop.time() + self._wait_s)
        futures = [loop.create_future() for _ in requests]
        group.requests.extend(requests)
        group.futures.extend(futures)
        group.tokens += sum(tokens)
        self._wake.set()
        return list(await asyncio.gather(*futures))

    async def run(self) -> None:
        """Run engine calls until cancelled; requests still queued then are cancelled."""
        try:
            while True:
                key = await self._next_due()
                await self._decide(key, self._groups.pop(key))
        finally:
            for group in self._groups.values():
                for future in group.futures:
                    future.cancel()
            self._groups.clear()

    def _count_tokens(self, requests: Requests) -> list[int]:
        return [self._engine.count_tokens(context, spec) for context, spec in requests]

    async def _next_due(self) -> GroupKey:
        """Wait for a full group or an expired wait, and return the oldest such group."""
        loop = asyncio.get_running_loop()
        while True:
            now = loop.time()
            due = [
                key
                for key, group in self._groups.items()
                if group.tokens >= self._token_budget or group.deadline <= now
            ]
            if due:
                return min(due, key=lambda key: self._groups[key].deadline)
            self._wake.clear()
            deadlines = [group.deadline for group in self._groups.values()]
            timeout = max(min(deadlines) - now, 0.0) if deadlines else None
            try:
                await asyncio.wait_for(self._wake.wait(), timeout)
            except TimeoutError:
                continue

    def _call_engine(
        self,
        group: _Group,
        mode: Mode,
        min_confidence: float | None,
        risk: float | None,
        alpha: float | None,
    ) -> list[DecisionResult]:
        size = self._graph_batch_size
        if risk is None:
            return self._engine.decide_uncertified_many(group.requests, batch_size=size)
        return self._engine.decide_many(
            group.requests,
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
            batch_size=size,
        )

    async def _decide(self, key: GroupKey, group: _Group) -> None:
        mode, min_confidence, risk, alpha = key
        try:
            results = await asyncio.to_thread(
                self._call_engine, group, mode, min_confidence, risk, alpha
            )
        except asyncio.CancelledError:
            for future in group.futures:
                future.cancel()
            raise
        except Exception as error:
            for future in group.futures:
                if not future.done():
                    future.set_exception(error)
            return
        if self._observer is not None:
            self._observer.observe_batch(results)
        for future, result in zip(group.futures, results, strict=True):
            if not future.done():
                future.set_result(result)
