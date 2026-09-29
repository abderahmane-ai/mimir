import dataclasses

import pytest

from mimir.core.errors import PolicyMismatchError
from mimir.policy.binding import LoadedRuntime, bind
from mimir.policy.document import Configuration, Fingerprint
from mimir.runtime.session import torch_version

FINGERPRINT = Fingerprint(
    model="Mythologic/MIMIR-1",
    revision="v1.0",
    variant="fp32",
    weights_sha256="b" * 64,
    torch=torch_version(),
    configurations=(Configuration(device="cuda", hardware="Tesla T4"),),
)
RUNTIME = LoadedRuntime(
    variant="fp32",
    weights_sha256="b" * 64,
    torch=torch_version(),
    device="cuda",
    hardware="Tesla T4",
)


def test_exact_configuration_is_certified() -> None:
    assert bind(FINGERPRINT, RUNTIME) == "certified"


def test_other_hardware_needs_the_equivalence_check() -> None:
    assert bind(FINGERPRINT, dataclasses.replace(RUNTIME, hardware="NVIDIA L4")) == (
        "unlisted_hardware"
    )


def test_another_torch_version_needs_the_equivalence_check() -> None:
    assert bind(FINGERPRINT, dataclasses.replace(RUNTIME, torch="0.0.0")) == "unlisted_hardware"


@pytest.mark.parametrize(
    ("runtime", "field"),
    [
        (dataclasses.replace(RUNTIME, variant="fp16"), "variant"),
        (dataclasses.replace(RUNTIME, weights_sha256="d" * 64), "weights SHA-256"),
    ],
)
def test_different_weights_are_refused(runtime: LoadedRuntime, field: str) -> None:
    with pytest.raises(PolicyMismatchError, match=field):
        bind(FINGERPRINT, runtime)


@pytest.mark.parametrize(
    "runtime",
    [
        dataclasses.replace(RUNTIME, device="cpu"),
        dataclasses.replace(RUNTIME, hardware="other"),
    ],
)
def test_a_different_device_or_hardware_needs_the_equivalence_check(
    runtime: LoadedRuntime,
) -> None:
    assert bind(FINGERPRINT, runtime) == "unlisted_hardware"
