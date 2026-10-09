# sysone-bench

MIMIR-1's run on the independent [sysone-bench](https://github.com/instax-dutta/sysone-bench), made through the benchmark's own protocol: its sealed manifest and checksum, seed 42, its v2 orchestrator, and the adapter `mimir_runner.py` beside this file. 952 evaluation cases, 1,240 decisions, all nine suites.

MIMIR scores **0.6597** on the evaluation split. The published totals on the same manifest are Jev 0.9065, Laya 0.6863 and Qwen PCD 0.6048. MIMIR is English-only, so `multilingual_intent` is out of distribution for it; on the eight English suites MIMIR scores 0.718 against Laya's 0.712.

`results.json` carries the run identity, the per-suite numbers and the provenance. `reproduce.sh` reruns it (CPU, the model from the Hub, about 2 GB of disk). The run is submitted upstream as a pull request; until that merges, `results.json` here is the record.

- Adapter: `mimir_runner.py` (same file submitted upstream).
- Run: `results/raw/mimir-1.2.0-20261009/` in a sysone-bench checkout (`mimir-decisions` 1.2.0, Torch 2.14.0, CPU), with `metadata.json`, `predictions.jsonl`, `summary.json`, `usage.json` and `checksums.sha256`. Raw runs are gitignored upstream and mirrored by the maintainer. An earlier run on 1.0.0 gave the same accuracy and the same per-suite values.
- Gold: the run is scored against dataset 2.0.0 (0.6597). Rescored against the corrected 2.1.0 gold, which is what the published panel uses, the same answers score 0.6718 (`results.json`, `rescored_dataset_2_1_0`).
- Mode: decisions run in `certified` mode at risk 0.01. Of the 1,240 evaluation decisions, 200 were decided (130 correct, 0.650) and 1,040 deferred (688 correct, 0.662). Deferral does not separate right from wrong at this risk level: the deferred answers are as accurate as the decided ones. In 169 of the deferred choices no option applied, and the adapter answered with the argmax option; every other deferred decision keeps the model's own answer or probability. The benchmark contract has no abstention, so every decision counts as an answer.
- Recount: the counts are in the run's `metadata.json` under `decision_statuses`; `count_decisions.py <run directory>` recomputes them from `predictions.jsonl` alone.
