"""ONNX Runtime sessions with fixed provider and session options.

The options are covered by certification: `options_sha256` is compared with the policy
fingerprint.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Final, Literal, Protocol

import numpy as np
import numpy.typing as npt
import onnxruntime

from mimir.core.errors import UncertifiedRuntimeError

# ONNX Runtime 1.30 uploads usage events to mobile.events.data.microsoft.com from a background
# thread; an upload still in flight when the process exits locks a destroyed mutex and aborts
# it ("recursive_mutex lock failed", seen in the integration suite on 2026-09-27).
onnxruntime.disable_telemetry_events()

Device = Literal["cpu", "cuda"]
Array = npt.NDArray[np.generic]

CPU: Final = "CPUExecutionProvider"
CUDA: Final = "CUDAExecutionProvider"
PROVIDERS: Final[Mapping[Device, str]] = {"cpu": CPU, "cuda": CUDA}
# TF32 would round fp32 matrix products to 10 mantissa bits on Ampere and later GPUs.
PROVIDER_OPTIONS: Final[Mapping[str, Mapping[str, str]]] = {CPU: {}, CUDA: {"use_tf32": "0"}}
SESSION_OPTIONS: Final[Mapping[str, str | bool]] = {
    "graph_optimization_level": "ORT_ENABLE_ALL",
    "use_deterministic_compute": True,
}
RUNTIME_DISTRIBUTIONS: Final = (
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-openvino",
    "onnxruntime-qnn",
)


class GraphSession(Protocol):
    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, Array]
    ) -> list[Array]: ...


def runtime_version() -> str:
    version: str = onnxruntime.__version__
    return version


def installed_runtimes() -> list[str]:
    """Return the installed ONNX Runtime distributions (all provide `onnxruntime`)."""
    found: list[str] = []
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        found.append(name)
    return found


def check_single_runtime() -> None:
    """Raise if more than one ONNX Runtime distribution is installed."""
    found = installed_runtimes()
    if len(found) > 1:
        keep = "onnxruntime-gpu" if "onnxruntime-gpu" in found else found[0]
        remove = [name for name in found if name != keep]
        message = (
            f"{found} are installed and share the module 'onnxruntime': "
            f"pip uninstall {' '.join(remove)}, then pip install --force-reinstall {keep}"
        )
        raise UncertifiedRuntimeError(message)


def options_sha256(provider: str) -> str:
    """Return the SHA-256 of the canonical JSON of the provider and all session options."""
    canonical = json.dumps(
        {
            "provider": provider,
            "provider_options": dict(PROVIDER_OPTIONS[provider]),
            "session_options": dict(SESSION_OPTIONS),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def resolve_device(device: str) -> Device:
    """Resolve `auto`, `cpu` or `cuda`; `auto` selects CUDA when available."""
    available = onnxruntime.get_available_providers()
    if device == "auto":
        return "cuda" if CUDA in available else "cpu"
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if CUDA not in available:
            message = (
                f"device 'cuda' needs {CUDA}; this ONNX Runtime offers {available}: "
                "pip install 'mimir-decisions[local-gpu]'"
            )
            raise UncertifiedRuntimeError(message)
        return "cuda"
    message = f"device {device!r} is not one of 'auto', 'cpu', 'cuda'"
    raise ValueError(message)


def open_session(path: Path, device: Device) -> GraphSession:
    """Open a session on the device's provider. Raises if ONNX Runtime falls back to another."""
    provider = PROVIDERS[device]
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.use_deterministic_compute = True
    session = onnxruntime.InferenceSession(
        str(path),
        sess_options=options,
        providers=[provider],
        provider_options=[dict(PROVIDER_OPTIONS[provider])],
    )
    active: list[str] = session.get_providers()
    if not active or active[0] != provider:
        message = f"{path}: ONNX Runtime opened the session on {active}, not {provider}"
        raise UncertifiedRuntimeError(message)
    opened: GraphSession = session
    return opened
