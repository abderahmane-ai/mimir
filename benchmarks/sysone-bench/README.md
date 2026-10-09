# sysone-bench

MIMIR-1's run on the independent [sysone-bench](https://github.com/instax-dutta/sysone-bench), made through the benchmark's own protocol: its sealed manifest and checksum, seed 42, its v2 orchestrator, and the adapter `mimir_runner.py` beside this file. 952 evaluation cases, 1,240 decisions, all nine suites.

MIMIR scores **0.6597** on the evaluation split. The published totals on the same manifest are Jev 0.9065, Laya 0.6863 and Qwen PCD 0.6048. MIMIR is English-only, so `multilingual_intent` is out of distribution for it; on the eight English suites MIMIR scores 0.718 against Laya's 0.712.

`results.json` carries the run identity, the per-suite numbers and the provenance. `reproduce.sh` reruns it (CPU, the model from the Hub, about 2 GB of disk). The run is submitted upstream as a pull request; until that merges, `results.json` here is the record.

- Adapter: `mimir_runner.py` (same file submitted upstream).
- Run: `results/v2/runs/mimir-1.2.0-20261009/` in a sysone-bench checkout (`mimir-decisions` 1.2.0, Torch 2.14.0, CPU), with `metadata.json`, `predictions.jsonl`, `summary.json`, `usage.json` and `checksums.sha256`. An earlier run on 1.0.0 gave the same accuracy and the same per-suite values.
- Mode: decisions run in `certified` mode at risk 0.01. Of the 1,240 evaluation decisions, 200 were decided and 1,040 deferred; in 169 of the deferred choices no option applied, and the adapter answered with the argmax option. A deferred decision otherwise keeps the model's own answer. The benchmark contract has no abstention, so every one of these counts as an answer.
- Recount: `count_decisions.py <run directory>` recomputes accuracy and these counts from `predictions.jsonl` alone.
