import dataclasses

import pytest

from mimir.core.errors import PolicyMismatchError, UncertifiedRuntimeError
from mimir.policy.binding import LoadedRuntime, bind
from mimir.policy.document import Configuration, Fingerprint

FINGERPRINT = Fingerprint(
    model="Mythologic/MIMIR-1",
    revision="v1.0",
    variant="fp16",
    graph_sha256="a" * 64,
    weights_sha256="b" * 64,
    opset=20,
    onnxruntime="1.30.0",
    configurations=(
        Configuration(
            provider="CUDAExecutionProvider", options_sha256="c" * 64, hardware="Tesla T4"
        ),
    ),
)
RUNTIME = LoadedRuntime(
    variant="fp16",
    graph_sha256="a" * 64,
    weights_sha256="b" * 64,
    opset=20,
    onnxruntime="1.30.0",
    provider="CUDAExecutionProvider",
    options_sha256="c" * 64,
    hardware="Tesla T4",
)


def test_exact_configuration_is_certified() -> None:
    assert bind(FINGERPRINT, RUNTIME) == "certified"


def test_other_hardware_needs_the_equivalence_check() -> None:
    assert bind(FINGERPRINT, dataclasses.replace(RUNTIME, hardware="NVIDIA L4")) == (
        "unlisted_hardware"
    )


@pytest.mark.parametrize(
    ("runtime", "field"),
    [
        (dataclasses.replace(RUNTIME, variant="fp32"), "variant"),
        (dataclasses.replace(RUNTIME, graph_sha256="d" * 64), "graph SHA-256"),
        (dataclasses.replace(RUNTIME, weights_sha256="d" * 64), "weights SHA-256"),
        (dataclasses.replace(RUNTIME, opset=21), "opset"),
    ],
)
def test_a_different_model_is_refused(runtime: LoadedRuntime, field: str) -> None:
    with pytest.raises(PolicyMismatchError, match=field):
        bind(FINGERPRINT, runtime)


def test_a_different_runtime_version_is_refused_with_the_fix() -> None:
    with pytest.raises(UncertifiedRuntimeError, match=r"onnxruntime==1\.30\.0"):
        bind(FINGERPRINT, dataclasses.replace(RUNTIME, onnxruntime="1.31.0"))


@pytest.mark.parametrize(
    "runtime",
    [
        dataclasses.replace(RUNTIME, provider="CPUExecutionProvider"),
        dataclasses.replace(RUNTIME, options_sha256="e" * 64),
    ],
)
def test_a_different_provider_or_options_is_refused(runtime: LoadedRuntime) -> None:
    with pytest.raises(UncertifiedRuntimeError, match="mimir calibrate"):
        bind(FINGERPRINT, runtime)
