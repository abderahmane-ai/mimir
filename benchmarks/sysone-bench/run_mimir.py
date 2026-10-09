"""Run the sealed sysone-bench manifest through its own orchestrator, for MIMIR."""

import sys
from pathlib import Path


def factory(name: str) -> object:
    from runners.mimir_runner import MimirRunner

    return MimirRunner()


def progress(model: str, suite: str, split: str, decisions: int) -> None:
    print(f"[{model}] {suite}/{split}: {decisions} decisions", flush=True)


def main() -> int:
    clone = Path(sys.argv[1]).resolve()
    run_id = sys.argv[2]
    sys.path.insert(0, str(clone))
    from benchmark.orchestrator import run_all_v2

    paths = run_all_v2(
        ["mimir"],
        clone / "datasets/v2/manifest.jsonl",
        clone / "results/v2/runs",
        runner_factory=factory,
        manifest_checksum_path=clone / "datasets/v2/manifest.sha256",
        run_id=run_id,
        progress=progress,
    )
    for path in paths:
        print(f"run artifacts: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
