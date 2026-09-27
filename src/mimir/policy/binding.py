"""Matching a policy's fingerprint against the loaded model and runtime.

The graph, weights, variant, opset, ONNX Runtime version, provider and provider options must
all match. Unlisted hardware additionally requires the equivalence check
(`mimir.runtime.equivalence`).
"""

from dataclasses import dataclass
from typing import Literal

from mimir.core.errors import PolicyMismatchError, UncertifiedRuntimeError
from mimir.policy.document import Fingerprint

Binding = Literal["certified", "unlisted_hardware"]


@dataclass(frozen=True, slots=True)
class LoadedRuntime:
    """The loaded graph and weights (by SHA-256) and the runtime configuration."""

    variant: str
    graph_sha256: str
    weights_sha256: str
    opset: int
    onnxruntime: str
    provider: str
    options_sha256: str
    hardware: str


def bind(fingerprint: Fingerprint, runtime: LoadedRuntime) -> Binding:
    """Match a fingerprint against the loaded runtime.

    Returns `certified` on a full match and `unlisted_hardware` when only the hardware differs.

    Raises:
        PolicyMismatchError: the variant, graph, weights or opset differ.
        UncertifiedRuntimeError: the ONNX Runtime version, provider or options differ.
    """
    model_mismatches = [
        f"{name} is {found!r}, the policy names {expected!r}"
        for name, found, expected in (
            ("variant", runtime.variant, fingerprint.variant),
            ("graph SHA-256", runtime.graph_sha256, fingerprint.graph_sha256),
            ("weights SHA-256", runtime.weights_sha256, fingerprint.weights_sha256),
            ("opset", runtime.opset, fingerprint.opset),
        )
        if found != expected
    ]
    if model_mismatches:
        message = (
            f"the policy for {fingerprint.model}@{fingerprint.revision} does not match the "
            f"loaded model: {'; '.join(model_mismatches)}"
        )
        raise PolicyMismatchError(message)
    if runtime.onnxruntime != fingerprint.onnxruntime:
        message = (
            f"ONNX Runtime {runtime.onnxruntime} is loaded; the policy is certified on "
            f"{fingerprint.onnxruntime}: install onnxruntime=={fingerprint.onnxruntime}, or "
            f"onnxruntime-gpu=={fingerprint.onnxruntime} for CUDA"
        )
        raise UncertifiedRuntimeError(message)
    matching = [
        configuration
        for configuration in fingerprint.configurations
        if configuration.provider == runtime.provider
        and configuration.options_sha256 == runtime.options_sha256
    ]
    if not matching:
        certified = sorted({configuration.provider for configuration in fingerprint.configurations})
        message = (
            f"{runtime.provider} with options {runtime.options_sha256[:12]} is not certified; "
            f"certified providers: {certified}. Use a certified device, or create a policy "
            f"with `mimir calibrate`"
        )
        raise UncertifiedRuntimeError(message)
    if any(configuration.hardware == runtime.hardware for configuration in matching):
        return "certified"
    return "unlisted_hardware"
