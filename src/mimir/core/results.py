"""Decision results.

`answer` is always the model's prediction. `status` says whether it cleared the operating
floor:

- `DECIDED`: the answer cleared it (in `standard` mode there is no floor).
- `ABSTAINED`: the model found that no listed option applies.
- `DEFERRED`: the answer is below the floor (`threshold` mode's `min_confidence`, or the
  certified threshold in `certified` mode); review it before acting.

`actionable` is true for `DECIDED` and `ABSTAINED`. `certified` is true when the loaded
policy holds a threshold for this decision type at the requested risk and the score passes
it; `certificate` then carries the evidence that certified it. `confidence` is the
calibrated score the floor is compared with.

Results from `decide_uncertified` have raw model probabilities, a status taken from the answer
alone, and no certificate or prediction set.

`relevant_context` is sorted by relevance, highest first. Relevance is the share of the
model's evidence attention on each part of the context, not a causal attribution.
"""

from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from mimir.core.decisions import Verdict

SCHEMA_VERSION: Final = 1


class Status(StrEnum):
    DECIDED = "decided"
    ABSTAINED = "abstained"
    DEFERRED = "deferred"


ContextKind = Literal["passage", "table", "table_row", "field"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Certificate(_Frozen):
    """The certified threshold a decision was checked against.

    On `records` held-out decisions, `taken` reached the threshold and `errors` of those were
    wrong. The exact binomial test of `errors` out of `taken` at rate `risk` gives `p_value`:
    the error rate of taken decisions is at most `risk` with probability `confidence`.
    `coverage` is `taken / records`, with a 95% Wilson interval.
    """

    risk: float
    confidence: float
    threshold: float
    records: int
    taken: int
    errors: int
    p_value: float
    coverage: float
    coverage_interval: tuple[float, float] | None
    model: str
    revision: str
    variant: str
    origin: Literal["release", "custom"]


class ContextRelevance(_Frozen):
    """Relevance of one part of the context.

    `index` is the position of the passage, table or field in the context. `row` is the row
    position for `table_row` and None otherwise. A `table` part is the caption and header.
    """

    kind: ContextKind
    index: int
    row: int | None
    relevance: float
    text: str


class OptionSet(_Frozen):
    """A conformal prediction set. `abstain` is true when "none of the options" is included."""

    options: tuple[str, ...]
    abstain: bool


class _Result(_Frozen):
    schema_version: Literal[1] = SCHEMA_VERSION
    status: Status
    actionable: bool = Field(description="Whether the answer cleared the operating floor.")
    certified: bool = Field(
        description="Whether the answer passed the policy's threshold at the requested risk."
    )
    confidence: float | None = Field(
        description="Calibrated probability of the answer, compared with the floor."
    )
    relevant_context: tuple[ContextRelevance, ...]
    certificate: Certificate | None
    latency_ms: float


class ChoiceResult(_Result):
    type: Literal["choice"] = "choice"
    answer: str | None = Field(description="The option id, or None when no option applies.")
    probabilities: dict[str, float]
    abstain_probability: float
    prediction_set: OptionSet | None


class MultiChoiceResult(_Result):
    type: Literal["multi_choice"] = "multi_choice"
    answer: tuple[str, ...]
    probabilities: dict[str, float] = Field(
        description="Independent probability that each option applies."
    )


class YesNoResult(_Result):
    type: Literal["yes_no"] = "yes_no"
    answer: bool | None
    probabilities: dict[str, float]
    abstain_probability: float
    prediction_set: OptionSet | None


class VerifyResult(_Result):
    type: Literal["verify"] = "verify"
    answer: Verdict | None
    probabilities: dict[str, float]
    abstain_probability: float
    prediction_set: OptionSet | None


class RankResult(_Result):
    type: Literal["rank"] = "rank"
    answer: tuple[str, ...] = Field(description="Every candidate id, best first.")
    probabilities: dict[str, float] = Field(
        description="Each candidate's probability of being the best one."
    )


class RateResult(_Result):
    type: Literal["rate"] = "rate"
    answer: str
    probabilities: dict[str, float]
    prediction_set: tuple[str, ...] | None = Field(
        description="Contiguous levels in the conformal set; empty if the set is empty."
    )


class EstimateResult(_Result):
    type: Literal["estimate"] = "estimate"
    answer: float = Field(description="The mean of the predicted distribution.")
    unit: str | None
    interval: tuple[float, float] | tuple[()] | None = Field(
        description="Conformal interval at `alpha`; empty if the set is empty."
    )


DecisionResult = Annotated[
    ChoiceResult
    | MultiChoiceResult
    | YesNoResult
    | VerifyResult
    | RankResult
    | RateResult
    | EstimateResult,
    Field(discriminator="type"),
]
RESULT_FOR_SPEC: Final[dict[str, type[_Result]]] = {
    "choice": ChoiceResult,
    "multi_choice": MultiChoiceResult,
    "yes_no": YesNoResult,
    "verify": VerifyResult,
    "rank": RankResult,
    "rate": RateResult,
    "estimate": EstimateResult,
}
