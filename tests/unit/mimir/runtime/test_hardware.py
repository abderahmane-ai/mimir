import platform
import subprocess
from pathlib import Path

import pytest

from mimir.runtime import hardware
from mimir.runtime.hardware import UNKNOWN, cpu_brand, gpu_name, hardware_name


def test_cpu_name_is_architecture_then_brand() -> None:
    assert hardware_name("cpu") == f"{platform.machine()} {cpu_brand()}"
    assert cpu_brand()


def test_linux_brand_comes_from_cpuinfo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor : 0\nmodel name : Intel(R) Xeon(R) CPU @ 2.20GHz\n")
    monkeypatch.setattr(hardware, "CPUINFO", cpuinfo)
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert cpu_brand() == "Intel(R) Xeon(R) CPU @ 2.20GHz"


def test_gpu_name_reads_nvidia_smi_for_the_first_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        seen.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, stdout="Tesla T4\nTesla T4\n", stderr="")

    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert gpu_name() == "Tesla T4"
    assert seen[0][-2:] == ["-i", "1"]
    assert hardware_name("cuda") == "Tesla T4"


def test_gpu_name_is_unknown_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(arguments[0])

    monkeypatch.setattr("subprocess.run", run)
    assert gpu_name() == UNKNOWN
