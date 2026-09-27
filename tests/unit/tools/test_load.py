import asyncio
import json
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from load import Record, Report, percentile, read_records, request_bodies, run
from mimir.core.decisions import Choice, Estimate, MultiChoice, Rank, Rate
from mimir.core.wire import UncertifiedRequest
from mimir.server.app import create_app
from mimir.server.metrics import Metrics
from mimir.server.model import ServedModel
from tests.conftest import load_engine

STATE: Final[JsonValue] = {
    "passages": [{"title": "ticket", "text": "my card was charged twice"}],
    "tables": [],
    "fields": [],
}


def _record(
    identifier: str, kind: str, question: str, candidates: list[str], span: list[float] | None
) -> dict[str, JsonValue]:
    return {
        "id": identifier,
        "decision_type": kind,
        "question": question,
        "candidates": list[JsonValue](candidates),
        "state": STATE,
        "range": None if span is None else list[JsonValue](span),
        "raw_json": None,
    }


RECORDS: Final = [
    _record("e0", "categorical", "which team", ["billing", "security", "shipping"], None),
    _record("e1", "binary", "is it late", ["yes", "no"], None),
    _record("e2", "multilabel", "which", ["refund", "order"], None),
    _record("e3", "ranking", "best", ["refund", "order"], None),
    _record("e4", "ordinal", "rate", ["low", "medium", "high"], None),
    _record("e5", "continuous", "price", [], [0.0, 100.0]),
]


def _release_records(root: Path) -> Path:
    (root / "equivalence").mkdir(exist_ok=True)
    lines = [json.dumps(record) for record in RECORDS]
    (root / "equivalence" / "records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_records_become_the_specs_of_their_decision_type(tmp_path: Path) -> None:
    records = read_records(_release_records(tmp_path))
    assert [record.spec() for record in records] == [
        Choice("which team", ["billing", "security", "shipping"]),
        Choice("is it late", ["yes", "no"]),
        MultiChoice("which", ["refund", "order"]),
        Rank("best", ["refund", "order"]),
        Rate("rate", ["low", "medium", "high"]),
        Estimate("price", 0, 100),
    ]
    certified = json.loads(request_bodies(records, certified=True, seed=0)[0])
    assert set(certified) == {"context", "decision", "risk", "alpha"}
    uncertified = json.loads(request_bodies(records, certified=False, seed=0)[0])
    assert set(uncertified) == {"context", "decision"}


def test_the_order_is_seeded_and_keeps_every_record(tmp_path: Path) -> None:
    records = read_records(_release_records(tmp_path))
    in_file_order = [
        UncertifiedRequest(context=record.state, decision=record.spec()).model_dump_json().encode()
        for record in records
    ]
    first = request_bodies(records, certified=False, seed=0)
    assert first == request_bodies(records, certified=False, seed=0)
    assert first != in_file_order
    assert first != request_bodies(records, certified=False, seed=1)
    assert sorted(first) == sorted(in_file_order)


def test_a_continuous_record_without_a_range_is_refused() -> None:
    record = Record.model_validate({**RECORDS[5], "range": None})
    with pytest.raises(ValueError, match="record e5: a continuous record needs a range"):
        record.spec()


def test_percentiles_use_the_nearest_rank() -> None:
    values = [float(value) for value in range(1, 101)]
    assert (percentile(values, 0.5), percentile(values, 0.95), percentile(values, 0.99)) == (
        50.0,
        95.0,
        99.0,
    )
    assert percentile([7.0], 0.95) == 7.0


def _report(root: Path) -> Report:
    served = ServedModel(lambda: load_engine(root))
    app = create_app(served, metrics=Metrics())

    async def main() -> Report:
        async with app.router.lifespan_context(app):
            while served.state == "loading":
                await asyncio.sleep(0.01)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://mimir") as client:
                return await run(client, read_records(root), (1, 3), 4)

    return asyncio.run(main())


def test_a_run_measures_every_client_count_without_errors(release: Path) -> None:
    report = _report(_release_records(release))
    assert (report.route, report.records, report.seed) == ("/v1/decide", 6, 0)
    assert [(level.clients, level.requests, level.errors) for level in report.levels] == [
        (1, 4, 0),
        (3, 12, 0),
    ]
    for level in report.levels:
        latency = level.latency_ms
        assert 0 < latency.p50 <= latency.p95 <= latency.p99 <= latency.max
        assert level.requests_per_s == pytest.approx(level.requests / level.seconds)
    assert report.server.runtime.hardware


def test_a_server_without_a_policy_is_measured_uncertified(
    release_builder: object,
) -> None:
    assert callable(release_builder)
    report = _report(_release_records(release_builder("bare", with_policy=False)))
    assert report.route == "/v1/decide/uncertified"
