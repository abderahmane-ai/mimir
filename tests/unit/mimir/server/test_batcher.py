import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from typing import TypeVar

import pytest

from mimir.core.context import Context
from mimir.core.decisions import Choice, DecisionSpec, Rank, YesNo
from mimir.core.errors import InputLimitError, RiskLevelError
from mimir.core.results import DecisionResult
from mimir.runtime.engine import Mimir
from mimir.server.batcher import Batcher
from tests.conftest import BatchRecorder, count_graph_runs

T = TypeVar("T")
TEXT = Context.coerce("my card was charged twice")
TEAM = Choice("which team", ["billing", "security", "shipping"])
REQUESTS: list[tuple[Context, DecisionSpec]] = [
    (TEXT, TEAM),
    (Context.coerce("the invoice is late"), YesNo("is it late")),
    (TEXT, Rank("best", ["refund", "order"])),
]


def _values(results: Sequence[DecisionResult]) -> list[dict[str, object]]:
    return [result.model_dump(exclude={"latency_ms"}) for result in results]


def _running(batcher: Batcher, work: Callable[[], Awaitable[T]]) -> T:
    async def main() -> T:
        worker = asyncio.create_task(batcher.run())
        try:
            return await work()
        finally:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    return asyncio.run(main())


def test_concurrent_requests_share_one_engine_call(engine: Mimir) -> None:
    recorder = BatchRecorder()
    batcher = Batcher(engine, token_budget=100_000, wait_s=0.05, observer=recorder)

    async def work() -> list[list[DecisionResult]]:
        return list(
            await asyncio.gather(
                *(batcher.submit([request], risk=0.01, alpha=None) for request in REQUESTS)
            )
        )

    found = _running(batcher, work)
    assert [len(batch) for batch in recorder.batches] == [3]
    assert _values([results[0] for results in found]) == _values(
        engine.decide_many(REQUESTS, risk=0.01)
    )


def test_groups_split_by_risk_and_alpha_and_uncertified(engine: Mimir) -> None:
    recorder = BatchRecorder()
    batcher = Batcher(engine, token_budget=100_000, wait_s=0.02, observer=recorder)

    async def work() -> list[list[DecisionResult]]:
        return list(
            await asyncio.gather(
                batcher.submit(REQUESTS[:1], risk=0.01, alpha=None),
                batcher.submit(REQUESTS[:1], risk=0.01, alpha=0.3),
                batcher.submit(REQUESTS[:1], risk=None, alpha=None),
                batcher.submit(REQUESTS[1:], risk=0.01, alpha=None),
            )
        )

    certified, other_alpha, uncertified, rest = _running(batcher, work)
    assert sorted(len(batch) for batch in recorder.batches) == [1, 1, 3]
    assert uncertified[0].certificate is None
    assert certified[0].certificate is not None
    assert _values(uncertified) == _values(engine.decide_uncertified_many(REQUESTS[:1]))
    assert _values(other_alpha) == _values(engine.decide_many(REQUESTS[:1], alpha=0.3))
    assert [result.type for result in rest] == ["yes_no", "rank"]


def test_a_full_group_runs_without_waiting(engine: Mimir) -> None:
    batcher = Batcher(engine, token_budget=1, wait_s=30.0)
    started = time.perf_counter()
    found = _running(batcher, lambda: batcher.submit(REQUESTS, risk=0.01, alpha=None))
    assert time.perf_counter() - started < 10
    assert [result.type for result in found] == ["choice", "yes_no", "rank"]


def test_an_engine_failure_reaches_every_request_of_its_batch(engine: Mimir) -> None:
    recorder = BatchRecorder()
    batcher = Batcher(engine, token_budget=100_000, wait_s=0.02, observer=recorder)

    async def work() -> list[list[DecisionResult] | BaseException]:
        return list(
            await asyncio.gather(
                *(batcher.submit([request], risk=0.3, alpha=None) for request in REQUESTS),
                return_exceptions=True,
            )
        )

    outcomes = _running(batcher, work)
    assert all(isinstance(outcome, RiskLevelError) for outcome in outcomes)
    assert "risk 0.3 is not certified" in str(outcomes[0])
    assert recorder.batches == []


def test_requests_over_the_release_limits_fail_before_queueing(engine: Mimir) -> None:
    recorder = BatchRecorder()
    batcher = Batcher(engine, token_budget=100_000, wait_s=0.0, observer=recorder)
    options = [f"option {index}" for index in range(9)]
    too_many = [(TEXT, Choice("which", options))]
    with pytest.raises(InputLimitError, match="options is 9; the release is tested up to 8"):
        _running(batcher, lambda: batcher.submit(too_many, risk=0.01, alpha=None))
    assert recorder.batches == []


def test_stopping_cancels_queued_requests(engine: Mimir) -> None:
    batcher = Batcher(engine, token_budget=100_000, wait_s=30.0)

    async def main() -> None:
        worker = asyncio.create_task(batcher.run())
        pending = asyncio.create_task(batcher.submit(REQUESTS[:1], risk=0.01, alpha=None))
        await asyncio.sleep(0.2)
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(main())


@pytest.mark.parametrize(("size", "expected_runs"), [(None, 1), (1, 3)])
def test_the_graph_batch_size_caps_requests_per_graph_run(
    engine: Mimir, monkeypatch: pytest.MonkeyPatch, size: int | None, expected_runs: int
) -> None:
    recorder = BatchRecorder()
    batcher = Batcher(
        engine, token_budget=100_000, wait_s=0.05, graph_batch_size=size, observer=recorder
    )
    runs = count_graph_runs(engine, monkeypatch)

    async def work() -> list[list[DecisionResult]]:
        return list(
            await asyncio.gather(
                *(batcher.submit([request], risk=0.01, alpha=None) for request in REQUESTS)
            )
        )

    found = _running(batcher, work)
    assert [len(batch) for batch in recorder.batches] == [3]
    assert len(runs) == expected_runs
    assert _values([results[0] for results in found]) == _values(
        engine.decide_many(REQUESTS, risk=0.01)
    )


@pytest.mark.parametrize(("budget", "wait"), [(0, 0.0), (1, -0.1)])
def test_the_budget_and_wait_are_validated(engine: Mimir, budget: int, wait: float) -> None:
    with pytest.raises(ValueError, match="token_budget must be at least 1"):
        Batcher(engine, token_budget=budget, wait_s=wait)
