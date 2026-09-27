"""Labelled decisions, the input format of `mimir bench` and `mimir calibrate`.

A JSONL file with one `{"context": ..., "decision": <spec>, "label": ...}` object per line.
Labels by spec type:

| Spec | Label |
|---|---|
| `choice` | an option id, or null when no option applies |
| `multi_choice` | the list of option ids that apply |
| `yes_no` | true or false |
| `verify` | `supported`, `contradicted` or `not_enough_information` |
| `rank` | the id of the best candidate, or the list of ids that are equally best |
| `rate` | a level id |
| `estimate` | a number within [low, high] |
"""

from pathlib import Path
from typing import Self, TypeGuard

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from mimir.core.context import ContextInput
from mimir.core.decisions import (
    VERDICTS,
    Choice,
    DecisionSpec,
    Estimate,
    MultiChoice,
    Rank,
    Rate,
    Verify,
    YesNo,
)
from mimir.core.results import (
    ChoiceResult,
    DecisionResult,
    EstimateResult,
    MultiChoiceResult,
    RankResult,
    RateResult,
    VerifyResult,
    YesNoResult,
)

Label = bool | float | str | list[str] | None


class LabelError(ValueError):
    """Invalid labelled file: bad JSON, an invalid spec, or a label that does not fit it."""


class LabelledDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    context: ContextInput
    decision: DecisionSpec
    label: Label

    @model_validator(mode="after")
    def _check_label_matches_spec(self) -> Self:
        spec, label = self.decision, self.label
        ids = set(spec.option_ids)
        match spec:
            case Choice():
                valid = label is None or (isinstance(label, str) and label in ids)
            case MultiChoice():
                valid = isinstance(label, list) and set(label) <= ids
            case YesNo():
                valid = isinstance(label, bool)
            case Verify():
                valid = isinstance(label, str) and label in VERDICTS
            case Rank():
                best = [label] if isinstance(label, str) else label
                valid = isinstance(best, list) and bool(best) and set(best) <= ids
            case Rate():
                valid = isinstance(label, str) and label in ids
            case Estimate():
                valid = _is_number(label) and spec.low <= float(label) <= spec.high
        if not valid:
            message = f"label {label!r} does not fit a {spec.type} decision over {sorted(ids)}"
            raise ValueError(message)
        return self


def _is_number(label: Label) -> TypeGuard[float]:
    return isinstance(label, int | float) and not isinstance(label, bool)


def is_correct(result: DecisionResult, label: Label) -> bool:
    """Return whether the answer matches the label.

    For `rank`, the top candidate must be one of the labelled best. For `estimate`, the label
    must lie in the prediction interval.
    """
    match result:
        case ChoiceResult() | YesNoResult() | VerifyResult() | RateResult():
            return result.answer == label
        case MultiChoiceResult():
            return isinstance(label, list) and set(result.answer) == set(label)
        case RankResult():
            best = [label] if isinstance(label, str) else label
            return isinstance(best, list) and result.answer[0] in best
        case EstimateResult(interval=(low, high)):
            return _is_number(label) and low <= float(label) <= high
        case EstimateResult():
            return False


def read_labelled(path: Path) -> list[LabelledDecision]:
    """Read a labelled JSONL file. Errors report the line number and field."""
    found: list[LabelledDecision] = []
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                found.append(LabelledDecision.model_validate_json(line))
            except ValidationError as error:
                message = f"{path}:{number}: {error}"
                raise LabelError(message) from error
    if not found:
        message = f"{path}: no labelled decisions"
        raise LabelError(message)
    return found
