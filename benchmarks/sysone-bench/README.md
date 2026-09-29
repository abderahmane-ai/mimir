# sysone-bench

MIMIR-1.0.0's run on the independent [sysone-bench](https://github.com/instax-dutta/sysone-bench), made through the benchmark's own protocol: its sealed manifest and checksum, seed 42, its v2 orchestrator, and the adapter `mimir_runner.py` beside this file. 952 evaluation cases, 1,240 decisions, all nine suites.

MIMIR scores **0.6597** on the evaluation split. The published totals on the same manifest are Jev 0.9065, Laya 0.6863 and Qwen PCD 0.6048. MIMIR is English-only, so `multilingual_intent` is out of distribution for it; on the eight English suites MIMIR scores 0.718 against Laya's 0.712.

`results.json` carries the run identity, the per-suite numbers and the provenance. `reproduce.sh` reruns it (CPU, the model from the Hub, about 2 GB of disk). The run is submitted upstream as a pull request; until that merges, `results.json` here is the record.

- Adapter: `mimir_runner.py` (same file submitted upstream).
- Run: `results/v2/runs/mimir-1.0.0-20260929/` in a sysone-bench checkout, with `metadata.json`, `predictions.jsonl`, `summary.json`, `usage.json` and `checksums.sha256`.
- Projection: a deferred or abstained decision becomes the argmax option, because the benchmark contract has no abstention.
