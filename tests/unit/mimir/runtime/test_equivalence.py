import dataclasses
from pathlib import Path

import pytest

from mimir.policy.binding import LoadedRuntime
from mimir.policy.document import Configuration, Fingerprint
from mimir.runtime.equivalence import cache_directory, equivalence_key, is_cached, record_pass
from tests.conftest import build_policy

RUNTIME = LoadedRuntime(
    variant="fp32",
    graph_sha256="a" * 64,
    weights_sha256="b" * 64,
    opset=20,
    onnxruntime="1.30.0",
    provider="CPUExecutionProvider",
    options_sha256="c" * 64,
    hardware="arm64 Apple M4",
)
POLICY = build_policy(
    Fingerprint(
        model="m",
        revision="r",
        variant="fp32",
        graph_sha256="a" * 64,
        weights_sha256="b" * 64,
        opset=20,
        onnxruntime="1.30.0",
        configurations=(
            Configuration(provider="CPUExecutionProvider", options_sha256="c" * 64, hardware="x"),
        ),
    )
)


def test_cache_directory_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMIR_CACHE", str(tmp_path / "explicit"))
    assert cache_directory() == tmp_path / "explicit"
    monkeypatch.delenv("MIMIR_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_directory() == tmp_path / "xdg" / "mimir"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert cache_directory() == Path.home() / ".cache" / "mimir"


def test_key_changes_with_anything_that_changes_numerics() -> None:
    base = equivalence_key(RUNTIME, POLICY)
    assert base == equivalence_key(RUNTIME, POLICY)
    for changed in (
        dataclasses.replace(RUNTIME, hardware="x86_64 Xeon"),
        dataclasses.replace(RUNTIME, onnxruntime="1.31.0"),
        dataclasses.replace(RUNTIME, graph_sha256="d" * 64),
    ):
        assert equivalence_key(changed, POLICY) != base
    stricter = build_policy(POLICY.document.fingerprint, threshold=0.9)
    assert equivalence_key(RUNTIME, stricter) != base


def test_record_pass_marks_the_key_cached(isolated_cache: Path) -> None:
    key = equivalence_key(RUNTIME, POLICY)
    assert not is_cached(key)
    record_pass(key, RUNTIME, 200)
    assert is_cached(key)
    assert (isolated_cache / "equivalence" / f"{key}.json").read_text(encoding="utf-8").count(
        '"records": 200'
    ) == 1
