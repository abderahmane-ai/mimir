"""Decision specs: the question and the options to decide between.

| Spec | Model decision type | Answer |
|---|---|---|
| `Choice` | categorical (binary for two options) | an option id, or None |
| `MultiChoice` | multilabel | the option ids that apply |
| `YesNo` | binary | `True` or `False` |
| `Verify` | categorical over three verdicts | a `Verdict` |
| `Rank` | ranking | candidate ids, best first |
| `Rate` | ordinal | a level id |
| `Estimate` | continuous | a number in `[low, high]` |

Options are a sequence of texts or a mapping of id to text. Results use the ids.
`YesNo` and `Verify` use fixed option texts matching the training data.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Final, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

ModelType = Literal["binary", "categorical", "multilabel", "ranking", "ordinal", "continuous"]
MODEL_TYPES: Final[tuple[ModelType, ...]] = (
    "binary",
    "categorical",
    "multilabel",
    "ranking",
    "ordinal",
    "continuous",
)
Verdict = Literal["supported", "contradicted", "not_enough_information"]
VERDICTS: Final[tuple[Verdict, ...]] = ("supported", "contradicted", "not_enough_information")
VERDICT_TEXTS: Final = (
    "the evidence supports the statement",
    "the evidence contradicts the statement",
    "the evidence neither supports nor contradicts the statement",
)
VERIFY_QUESTION: Final = "Judge this statement against the evidence: {claim}"
YES_NO_IDS: Final = ("no", "yes")
MIN_OPTIONS: Final = 2
MIN_LEVELS: Final = 3


def _options(value: object) -> object:
    """Map a sequence of texts to `{text: text}`, rejecting duplicates."""
    if isinstance(value, Mapping | str) or not isinstance(value, Sequence):
        return value
    texts = [str(text) for text in value]
    duplicates = sorted({text for text in texts if texts.count(text) > 1})
    if duplicates:
        message = f"duplicate options {duplicates}"
        raise ValueError(message)
    return {text: text for text in texts}


Options = Annotated[
    dict[str, str],
    BeforeValidator(_options, json_schema_input_type=list[str] | dict[str, str]),
]


def _check_options(options: dict[str, str], minimum: int, name: str) -> dict[str, str]:
    if len(options) < minimum:
        message = f"{name} needs at least {minimum} entries; got {len(options)}: {list(options)}"
        raise ValueError(message)
    empty_ids = [key for key in options if not key.strip()]
    empty_texts = [key for key, text in options.items() if not text.strip()]
    if empty_ids or empty_texts:
        message = f"{name} contain blank ids {empty_ids} or blank texts for ids {empty_texts}"
        raise ValueError(message)
    texts = list(options.values())
    repeated = sorted({text for text in texts if texts.count(text) > 1})
    if repeated:
        message = f"{name} contain duplicate texts {repeated}"
        raise ValueError(message)
    return options


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def _check_question(cls, question: str) -> str:
        if not question.strip():
            message = f"question {question!r} is blank"
            raise ValueError(message)
        return question

    @property
    def model_question(self) -> str:
        """The question text passed to the model."""
        return self.question


class Choice(_Spec):
    """Pick one option, or none."""

    type: Literal["choice"] = "choice"
    options: Options

    def __init__(
        self, question: str, options: Sequence[str] | Mapping[str, str], **fields: object
    ) -> None:
        BaseModel.__init__(self, question=question, options=options, **fields)

    @field_validator("options")
    @classmethod
    def _check_choice_options(cls, options: dict[str, str]) -> dict[str, str]:
        return _check_options(options, MIN_OPTIONS, "options")

    @property
    def model_type(self) -> ModelType:
        return "binary" if len(self.options) == MIN_OPTIONS else "categorical"

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(self.options)

    @property
    def option_texts(self) -> tuple[str, ...]:
        return tuple(self.options.values())


class MultiChoice(_Spec):
    """Pick every option that applies. The answer may be empty."""

    type: Literal["multi_choice"] = "multi_choice"
    options: Options

    def __init__(
        self, question: str, options: Sequence[str] | Mapping[str, str], **fields: object
    ) -> None:
        BaseModel.__init__(self, question=question, options=options, **fields)

    @field_validator("options")
    @classmethod
    def _check_multi_options(cls, options: dict[str, str]) -> dict[str, str]:
        return _check_options(options, MIN_OPTIONS, "options")

    @property
    def model_type(self) -> ModelType:
        return "multilabel"

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(self.options)

    @property
    def option_texts(self) -> tuple[str, ...]:
        return tuple(self.options.values())


class YesNo(_Spec):
    """Answer a yes/no question."""

    type: Literal["yes_no"] = "yes_no"

    def __init__(self, question: str, **fields: object) -> None:
        BaseModel.__init__(self, question=question, **fields)

    @property
    def model_type(self) -> ModelType:
        return "binary"

    @property
    def option_ids(self) -> tuple[str, ...]:
        return YES_NO_IDS

    @property
    def option_texts(self) -> tuple[str, ...]:
        return YES_NO_IDS


class Verify(BaseModel):
    """Check whether the context supports or contradicts a claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["verify"] = "verify"
    claim: str = Field(min_length=1)

    def __init__(self, claim: str, **fields: object) -> None:
        BaseModel.__init__(self, claim=claim, **fields)

    @field_validator("claim")
    @classmethod
    def _check_claim(cls, claim: str) -> str:
        if not claim.strip():
            message = f"claim {claim!r} is blank"
            raise ValueError(message)
        return claim

    @property
    def model_question(self) -> str:
        return VERIFY_QUESTION.format(claim=self.claim)

    @property
    def model_type(self) -> ModelType:
        return "categorical"

    @property
    def option_ids(self) -> tuple[str, ...]:
        return VERDICTS

    @property
    def option_texts(self) -> tuple[str, ...]:
        return VERDICT_TEXTS


class Rank(_Spec):
    """Rank candidates from best to worst."""

    type: Literal["rank"] = "rank"
    candidates: Options

    def __init__(
        self, question: str, candidates: Sequence[str] | Mapping[str, str], **fields: object
    ) -> None:
        BaseModel.__init__(self, question=question, candidates=candidates, **fields)

    @field_validator("candidates")
    @classmethod
    def _check_candidates(cls, candidates: dict[str, str]) -> dict[str, str]:
        return _check_options(candidates, MIN_OPTIONS, "candidates")

    @property
    def model_type(self) -> ModelType:
        return "ranking"

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(self.candidates)

    @property
    def option_texts(self) -> tuple[str, ...]:
        return tuple(self.candidates.values())


class Rate(_Spec):
    """Rate on an ordered scale. Levels are given from lowest to highest."""

    type: Literal["rate"] = "rate"
    levels: Options

    def __init__(
        self, question: str, levels: Sequence[str] | Mapping[str, str], **fields: object
    ) -> None:
        BaseModel.__init__(self, question=question, levels=levels, **fields)

    @field_validator("levels")
    @classmethod
    def _check_levels(cls, levels: dict[str, str]) -> dict[str, str]:
        return _check_options(levels, MIN_LEVELS, "levels")

    @property
    def model_type(self) -> ModelType:
        return "ordinal"

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(self.levels)

    @property
    def option_texts(self) -> tuple[str, ...]:
        return tuple(self.levels.values())


class Estimate(_Spec):
    """Estimate a value in `[low, high]`."""

    type: Literal["estimate"] = "estimate"
    low: float
    high: float
    unit: str | None = None

    def __init__(
        self, question: str, low: float, high: float, unit: str | None = None, **fields: object
    ) -> None:
        BaseModel.__init__(self, question=question, low=low, high=high, unit=unit, **fields)

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if not (math.isfinite(self.low) and math.isfinite(self.high) and self.low < self.high):
            message = f"low {self.low} must be below high {self.high}"
            raise ValueError(message)
        return self

    @property
    def model_type(self) -> ModelType:
        return "continuous"

    @property
    def option_ids(self) -> tuple[str, ...]:
        return ()

    @property
    def option_texts(self) -> tuple[str, ...]:
        return ()


DecisionSpec = Annotated[
    Choice | MultiChoice | YesNo | Verify | Rank | Rate | Estimate,
    Field(discriminator="type"),
]
SPEC_TYPES: Final = (Choice, MultiChoice, YesNo, Verify, Rank, Rate, Estimate)
