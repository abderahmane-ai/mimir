"""Load test of a running MIMIR server: requests per second and latency per client count.

Requests are the release's equivalence records in a seeded random order, sent to
`POST /v1/decide`, or to `POST /v1/decide/uncertified` when the server has no policy. The
report names the server's runtime and hardware.
"""

import argparse
import asyncio
import datetime
import math
import os
import random
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict

from mimir.core.context import Context
from mimir.core.decisions import (
    Choice,
    DecisionSpec,
    Estimate,
    ModelType,
    MultiChoice,
    Rank,
    Rate,
)
from mimir.core.wire import DecideRequest, ModelInfo, UncertifiedRequest

RECORDS_FILE: Final = "equivalence/records.jsonl"
CLIENTS: Final = (1, 8, 32)
REQUESTS_PER_CLIENT: Final = 16
WARM_UP_REQUESTS: Final = 4
SEED: Final = 0
TIMEOUT_S: Final = 300.0


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Record(BaseModel):
    """One line of `equivalence/records.jsonl`. Its other keys are not read."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    decision_type: ModelType
    question: str
    candidates: list[str]
    state: Context
    range: tuple[float, float] | None

    def spec(self) -> DecisionSpec:
        match self.decision_type:
            case "binary" | "categorical":
                return Choice(self.question, self.candidates)
            case "multilabel":
                return MultiChoice(self.question, self.candidates)
            case "ranking":
                return Rank(self.question, self.candidates)
            case "ordinal":
                return Rate(self.question, self.candidates)
            case "continuous":
                if self.range is None:
                    message = f"record {self.id}: a continuous record needs a range"
                    raise ValueError(message)
                return Estimate(self.question, *self.range)


class Latency(_Frozen):
    p50: float
    p95: float
    p99: float
    max: float


class Level(_Frozen):
    """The measurement at one client count. Latencies are in milliseconds."""

    clients: int
    requests: int
    errors: int
    seconds: float
    requests_per_s: float
    latency_ms: Latency


class Report(_Frozen):
    server: ModelInfo
    route: str
    records: int
    seed: int
    levels: tuple[Level, ...]
    mimir_decisions: str
    measured_at: str


def read_records(release: Path) -> list[Record]:
    lines = (release / RECORDS_FILE).read_text(encoding="utf-8").splitlines()
    return [Record.model_validate_json(line) for line in lines if line.strip()]


def request_bodies(records: list[Record], *, certified: bool, seed: int) -> list[bytes]:
    """One request body per record, in an order shuffled by `seed`."""
    requests = [
        DecideRequest(context=record.state, decision=record.spec())
        if certified
        else UncertifiedRequest(context=record.state, decision=record.spec())
        for record in records
    ]
    bodies = [request.model_dump_json().encode() for request in requests]
    random.Random(seed).shuffle(bodies)
    return bodies


def percentile(sorted_values: list[float], share: float) -> float:
    """The nearest-rank percentile of already sorted values."""
    return sorted_values[max(math.ceil(share * len(sorted_values)) - 1, 0)]


async def measure_level(
    client: httpx.AsyncClient, route: str, bodies: list[bytes], clients: int, per_client: int
) -> Level:
    latencies: list[float] = []
    errors = 0
    sent = 0

    async def worker() -> None:
        nonlocal errors, sent
        for _ in range(per_client):
            body = bodies[sent % len(bodies)]
            sent += 1
            started = time.perf_counter()
            response = await client.post(
                route, content=body, headers={"Content-Type": "application/json"}
            )
            latencies.append((time.perf_counter() - started) * 1000)
            errors += not response.is_success

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(clients)))
    seconds = time.perf_counter() - started
    ordered = sorted(latencies)
    return Level(
        clients=clients,
        requests=len(ordered),
        errors=errors,
        seconds=seconds,
        requests_per_s=len(ordered) / seconds,
        latency_ms=Latency(
            p50=percentile(ordered, 0.50),
            p95=percentile(ordered, 0.95),
            p99=percentile(ordered, 0.99),
            max=ordered[-1],
        ),
    )


async def run(
    client: httpx.AsyncClient,
    records: list[Record],
    clients: tuple[int, ...],
    per_client: int,
    seed: int = SEED,
) -> Report:
    """Warm the server up, then measure each client count in turn."""
    response = (await client.get("/v1/models")).raise_for_status()
    info = ModelInfo.model_validate_json(response.content)
    certified = info.certification != "none"
    route = "/v1/decide" if certified else "/v1/decide/uncertified"
    bodies = request_bodies(records, certified=certified, seed=seed)
    await measure_level(client, route, bodies, 1, WARM_UP_REQUESTS)
    levels = [await measure_level(client, route, bodies, count, per_client) for count in clients]
    return Report(
        server=info,
        route=route,
        records=len(records),
        seed=seed,
        levels=tuple(levels),
        mimir_decisions=metadata.version("mimir-decisions"),
        measured_at=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test a running MIMIR server.")
    parser.add_argument("--url", required=True, help="Server URL, e.g. http://127.0.0.1:8000.")
    parser.add_argument("--release", type=Path, required=True, help="Release directory.")
    parser.add_argument("--clients", default=",".join(map(str, CLIENTS)), help="Client counts.")
    parser.add_argument("--requests-per-client", type=int, default=REQUESTS_PER_CLIENT)
    parser.add_argument("--seed", type=int, default=SEED, help="Seed of the record order.")
    parser.add_argument("--out", type=Path, help="Also write the report to this file.")
    arguments = parser.parse_args()
    clients = tuple(int(count) for count in arguments.clients.split(","))
    key = os.environ.get("MIMIR_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    records = read_records(arguments.release)

    async def measure() -> Report:
        async with httpx.AsyncClient(
            base_url=arguments.url, headers=headers, timeout=TIMEOUT_S
        ) as client:
            return await run(
                client, records, clients, arguments.requests_per_client, arguments.seed
            )

    report = asyncio.run(measure())
    text = report.model_dump_json(indent=2)
    if arguments.out is not None:
        arguments.out.write_text(f"{text}\n", encoding="utf-8")
    sys.stdout.write(f"{text}\n")
    if any(level.errors for level in report.levels):
        sys.exit(1)


if __name__ == "__main__":
    main()
