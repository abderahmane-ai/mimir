"""Recount a MIMIR run from its predictions: decisions and accuracy by status, per split.

Usage: count_decisions.py <run directory>

Reads only `predictions.jsonl` and shares no code with the adapter or the orchestrator, so it
checks the `decision_statuses` block in `metadata.json`. Exits non-zero when a row has no status
for one of its questions.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def is_correct(question: dict, answer: dict, expected: object) -> bool:
    kind = question["type"]
    if kind == "choice":
        return answer["choice"] == expected
    if kind == "noul":
        return int(answer["noul"] >= 0.5) == expected
    return math.floor(answer["score"] + 0.5) == expected


def main() -> int:
    rows = [
        json.loads(line)
        for line in (Path(sys.argv[1]) / "predictions.jsonl").read_text("utf-8").splitlines()
    ]
    report: dict[str, dict] = {}
    for split in ("calibration", "evaluation"):
        statuses: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "decisions": 0})
        no_option_applied = 0
        for row in (row for row in rows if row["split"] == split):
            recorded = row["_raw_model"]["decisions"]
            if len(recorded) != len(row["question_ids"]):
                expected_count = len(row["question_ids"])
                print(f"{row['case_id']}: {len(recorded)} statuses for {expected_count} questions")
                return 1
            for question, entry in zip(row["questions"], recorded, strict=True):
                qid = question["qid"]
                slot = statuses[entry["status"]]
                slot["decisions"] += 1
                slot["correct"] += is_correct(question, row["answers"][qid], row["expected"][qid])
                no_option_applied += entry["no_option_applied"]
        decisions = sum(slot["decisions"] for slot in statuses.values())
        correct = sum(slot["correct"] for slot in statuses.values())
        report[split] = {
            "accuracy": None if decisions == 0 else correct / decisions,
            "decisions": decisions,
            "no_option_applied": no_option_applied,
            "statuses": {
                status: {**slot, "accuracy": slot["correct"] / slot["decisions"]}
                for status, slot in sorted(statuses.items())
            },
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
