import hashlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

from mimir.core.errors import UncertifiedRuntimeError
from mimir.runtime.session import (
    CPU,
    CUDA,
    check_single_runtime,
    installed_runtimes,
    open_session,
    options_sha256,
    resolve_device,
    runtime_version,
)
from tests.conftest import build_graph


def test_options_hash_is_the_canonical_json_of_every_option() -> None:
    canonical = json.dumps(
        {
            "provider": CUDA,
            "provider_options": {"use_tf32": "0"},
            "session_options": {
                "graph_optimization_level": "ORT_ENABLE_ALL",
                "use_deterministic_compute": True,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert options_sha256(CUDA) == hashlib.sha256(canonical.encode()).hexdigest()
    assert options_sha256(CPU) != options_sha256(CUDA)


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: [CPU])
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(UncertifiedRuntimeError, match="local-gpu"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="'tpu'"):
        resolve_device("tpu")
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: [CUDA, CPU])
    assert resolve_device("auto") == "cuda"
    assert resolve_device("cuda") == "cuda"


def test_two_runtime_builds_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    assert installed_runtimes() == ["onnxruntime"]
    check_single_runtime()
    real = metadata.distribution

    def distribution(name: str) -> metadata.Distribution:
        if name == "onnxruntime-gpu":
            return real("onnxruntime")
        return real(name)

    monkeypatch.setattr("importlib.metadata.distribution", distribution)
    with pytest.raises(UncertifiedRuntimeError, match="pip uninstall onnxruntime"):
        check_single_runtime()


def test_open_session_runs_the_graph_on_the_cpu(tmp_path: Path) -> None:
    graph = tmp_path / "model.onnx"
    build_graph(graph)
    opened = open_session(graph, "cpu")
    present = np.array([[True, True, False]])
    outputs = opened.run(
        ["utilities", "abstain"],
        {
            "candidate_present": present,
            "key_mask": np.array([[True]]),
            **{
                name: np.zeros((1, 1), dtype=np.int64)
                for name in (
                    "chunk_ids",
                    "key_chunk",
                    "key_position",
                    "candidate_ids",
                    "candidate_index",
                )
            },
            **{
                name: np.zeros((1, 1), dtype=np.bool_)
                for name in ("chunk_mask", "candidate_mask", "candidate_text_mask")
            },
            **{
                name: np.zeros(1, dtype=np.int64)
                for name in (
                    "chunk_record",
                    "question_length",
                    "decision_type",
                    "typed_kind",
                    "typed_year",
                    "typed_month",
                    "typed_day",
                    "typed_second",
                    "typed_chunk",
                    "typed_position",
                )
            },
            "typed_number": np.zeros(1, dtype=np.float32),
        },
    )
    np.testing.assert_array_equal(outputs[0], [[2.0, 2.0, 0.0]])
    np.testing.assert_array_equal(outputs[1], [0.5])


def test_runtime_version_is_the_pinned_one() -> None:
    assert runtime_version() == "1.30.0"


def test_importing_the_session_module_turns_off_runtime_telemetry() -> None:
    script = (
        "import onnxruntime\n"
        "calls = []\n"
        "onnxruntime.disable_telemetry_events = lambda: calls.append(1)\n"
        "import mimir.runtime.session\n"
        "print(len(calls))\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert finished.stdout.strip() == "1"
