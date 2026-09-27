import asyncio
import threading
from pathlib import Path

import pytest

from mimir.core.decisions import Choice, YesNo
from mimir.core.errors import ArtifactError
from mimir.core.results import ChoiceResult
from mimir.runtime.engine import Mimir
from mimir.server.model import NotReadyError, ServedModel
from tests.conftest import BatchRecorder, count_graph_runs, load_engine

TEXT = "my card was charged twice"
TEAM = Choice("which team", ["billing", "security", "shipping"])


async def _until_settled(served: ServedModel) -> None:
    for _ in range(500):
        if served.state != "loading":
            return
        await asyncio.sleep(0.01)
    pytest.fail("the model neither loaded nor failed")


def test_decisions_wait_for_the_load_and_then_run_in_batches(release: Path) -> None:
    gate = threading.Event()
    recorder = BatchRecorder()

    def load() -> Mimir:
        gate.wait(10)
        return load_engine(release)

    served = ServedModel(load, wait_s=0.02, observer=recorder)

    async def main() -> list[ChoiceResult]:
        async with served.running():
            assert served.state == "loading"
            with pytest.raises(NotReadyError, match="still loading; retry once GET /readyz"):
                await served.adecide(TEXT, TEAM)
            with pytest.raises(NotReadyError):
                served.info()
            gate.set()
            await _until_settled(served)
            return list(await asyncio.gather(*(served.adecide(TEXT, TEAM) for _ in range(4))))

    results = asyncio.run(main())
    assert served.state == "ready"
    assert served.error is None
    assert [len(batch) for batch in recorder.batches] == [4]
    assert {result.answer for result in results} == {results[0].answer}


def test_on_cpu_each_request_runs_the_graph_alone(
    release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = load_engine(release)
    runs = count_graph_runs(engine, monkeypatch)
    recorder = BatchRecorder()
    served = ServedModel(lambda: engine, wait_s=0.05, observer=recorder)

    async def main() -> None:
        async with served.running():
            await _until_settled(served)
            await asyncio.gather(*(served.adecide(TEXT, TEAM) for _ in range(4)))

    asyncio.run(main())
    assert engine.info().device == "cpu"
    assert [len(batch) for batch in recorder.batches] == [4]
    assert len(runs) == 4


def test_the_ready_model_answers_sync_calls_and_reports_its_engine(release: Path) -> None:
    served = ServedModel(lambda: load_engine(release))

    async def main() -> None:
        async with served.running():
            await _until_settled(served)
            engine = served.engine
            assert served.info() == engine.info()
            direct = engine.decide_uncertified(TEXT, YesNo("is it late"))
            found = await asyncio.to_thread(served.decide_uncertified, TEXT, YesNo("is it late"))
            assert found.model_dump(exclude={"latency_ms"}) == direct.model_dump(
                exclude={"latency_ms"}
            )

    asyncio.run(main())


def test_a_failed_load_is_kept_and_reported(tmp_path: Path) -> None:
    def load() -> Mimir:
        message = f"{tmp_path}: no manifest.json"
        raise ArtifactError(message)

    served = ServedModel(load)

    async def main() -> None:
        async with served.running():
            await _until_settled(served)
            with pytest.raises(NotReadyError, match=r"failed to load: ArtifactError: .*manifest"):
                await served.adecide(TEXT, TEAM)

    asyncio.run(main())
    assert served.state == "failed"
    assert served.error == f"ArtifactError: {tmp_path}: no manifest.json"
