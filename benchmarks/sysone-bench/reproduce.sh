#!/usr/bin/env bash
# Rerun MIMIR on sysone-bench: clone the benchmark at the measured commit, install the CPU
# engine, and run the sealed manifest through the benchmark's own orchestrator.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="${1:-$here/work}"
commit="9447a17fccb524957d1211ecdf46153821ff0810"

if [ ! -d "$work/.git" ]; then
  git clone https://github.com/instax-dutta/sysone-bench "$work"
fi
git -C "$work" fetch --depth 1 origin "$commit"
git -C "$work" checkout --detach "$commit"
cp "$here/mimir_runner.py" "$work/runners/mimir_runner.py"

python3 -m venv "$work/.venv"
"$work/.venv/bin/pip" install --quiet --upgrade pip
"$work/.venv/bin/pip" install --quiet numpy "mimir-decisions[local]==1.2.0"

run_id="mimir-1.2.0-$(date +%Y%m%d)"
PYTHONPATH="$work" "$work/.venv/bin/python" "$here/run_mimir.py" "$work" "$run_id"
"$work/.venv/bin/python" "$here/count_decisions.py" "$work/results/v2/runs/$run_id"
