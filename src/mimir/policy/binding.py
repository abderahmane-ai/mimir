"""Matching a policy's fingerprint against the loaded weights and runtime.

The variant and weights must match exactly. A device and hardware the fingerprint does not
list additionally requires the equivalence check (`mimir.runtime.equivalence`); a torch
version it does not name does too, since kernels change between versions.
"""

from dataclasses import dataclass
from typing import Literal

from mimir.core.errors import PolicyMismatchError
from mimir.policy.document import Fingerprint

Binding = Literal["certified", "unlisted_hardware"]


@dataclass(frozen=True, slots=True)
class LoadedRuntime:
    """The loaded weights (by SHA-256) and the runtime configuration."""

    variant: str
    weights_sha256: str
    torch: str
    device: Literal["cpu", "cuda"]
    hardware: str


def bind(fingerprint: Fingerprint, runtime: LoadedRuntime) -> Binding:
    """Match a fingerprint against the loaded runtime.

    Returns `certified` on a full match and `unlisted_hardware` when the configuration
    differs.

    Raises:
        PolicyMismatchError: the variant or weights differ.
    """
    mismatches = [
        f"{name} is {found!r}, the policy names {expected!r}"
        for name, found, expected in (
            ("variant", runtime.variant, fingerprint.variant),
            ("weights SHA-256", runtime.weights_sha256, fingerprint.weights_sha256),
        )
        if found != expected
    ]
    if mismatches:
        message = (
            f"the policy for {fingerprint.model}@{fingerprint.revision} does not match the "
            f"loaded weights: {'; '.join(mismatches)}"
        )
        raise PolicyMismatchError(message)
    matching = [
        configuration
        for configuration in fingerprint.configurations
        if configuration.device == runtime.device and configuration.hardware == runtime.hardware
    ]
    if matching and runtime.torch == fingerprint.torch:
        return "certified"
    return "unlisted_hardware"
