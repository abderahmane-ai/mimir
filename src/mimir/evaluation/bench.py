"""Accuracy, coverage and realised risk of results against labels (`mimir bench`).

Per spec type:

- `accuracy`: correct answers among all records, ignoring the policy;
- `coverage`: records not deferred;
- `risk`: wrong answers among records not deferred.

Each rate has a 95% Wilson interval. `estimate` also reports the mean absolute error.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from mimir.core.intervals import wilson_interval
from mimir.core.labels import Label, is_correct
from mimir.core.results import DecisionResult, EstimateResult, Status


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Rate(_Frozen):
    count: int
    total: int
    value: float | None
    interval: tuple[float, float] | None


class TypeReport(_Frozen):
    records: int
    accuracy: Rate
    coverage: Rate
    risk: Rate
    mean_absolute_error: float | None


class BenchReport(_Frozen):
    records: int
    types: dict[str, TypeReport]


def rate(count: int, total: int) -> Rate:
    return Rate(
        count=count,
        total=total,
        value=count / total if total else None,
        interval=wilson_interval(count, total),
    )


def bench(results: Sequence[DecisionResult], labels: Sequence[Label]) -> BenchReport:
    """Score results against their labels, grouped by spec type."""
    if len(results) != len(labels):
        message = f"{len(results)} results for {len(labels)} labels"
        raise ValueError(message)
    groups: dict[str, list[tuple[DecisionResult, Label]]] = {}
    for result, label in zip(results, labels, strict=True):
        groups.setdefault(result.type, []).append((result, label))
    types: dict[str, TypeReport] = {}
    for kind, members in sorted(groups.items()):
        correct = [is_correct(result, label) for result, label in members]
        taken = [result.status != Status.DEFERRED for result, _ in members]
        errors = sum(
            is_taken and not is_right for is_taken, is_right in zip(taken, correct, strict=True)
        )
        absolute = [
            abs(result.answer - float(label))
            for result, label in members
            if isinstance(result, EstimateResult) and isinstance(label, int | float)
        ]
        types[kind] = TypeReport(
            records=len(members),
            accuracy=rate(sum(correct), len(members)),
            coverage=rate(sum(taken), len(members)),
            risk=rate(errors, sum(taken)),
            mean_absolute_error=sum(absolute) / len(absolute) if absolute else None,
        )
    return BenchReport(records=len(results), types=types)
