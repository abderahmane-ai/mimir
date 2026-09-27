"""Hardware names as recorded in policy fingerprints.

- CPU: architecture and processor brand, e.g. `x86_64 Intel(R) Xeon(R) CPU @ 2.20GHz`.
- CUDA: GPU name from `nvidia-smi`, e.g. `Tesla T4`.

Returns `unknown` when the name cannot be determined.
"""

import os
import platform
import subprocess
from pathlib import Path
from typing import Final

from mimir.runtime.session import Device

UNKNOWN: Final = "unknown"
CPUINFO: Final = Path("/proc/cpuinfo")
COMMAND_TIMEOUT_S: Final = 10


def _command_output(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            arguments, capture_output=True, text=True, check=True, timeout=COMMAND_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = completed.stdout.strip().splitlines()
    return lines[0].strip() if lines else None


def cpu_brand() -> str:
    system = platform.system()
    if system == "Linux" and CPUINFO.is_file():
        for line in CPUINFO.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name" and value.strip():
                return value.strip()
    if system == "Darwin":
        found = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if found:
            return found
    return platform.processor() or UNKNOWN


def gpu_name() -> str:
    """Return the name of the first GPU in `CUDA_VISIBLE_DEVICES` (or GPU 0)."""
    arguments = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible.isdigit():
        arguments += ["-i", visible]
    return _command_output(arguments) or UNKNOWN


def hardware_name(device: Device) -> str:
    if device == "cuda":
        return gpu_name()
    return f"{platform.machine()} {cpu_brand()}"
