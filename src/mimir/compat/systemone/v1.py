"""Jev's `/v1/systemone` wire format (TypeSafe API reference, checked 2026-09-27).

`translate` converts a request into a `Context` and one spec per question; `respond` converts
the results back into Jev's answers. Question types map as follows:

| Jev | MIMIR spec |
|---|---|
| `choice` | `Choice` over the criteria keys |
| `noul` | `Choice` over `false` and `true` |
| `score`, 3 or more levels | `Rate` over levels `"0"`, `"1"`, ... |
| `score`, 2 levels | `Choice` over `"0"` and `"1"` |

Option texts follow the training data: `"<Key>: <description>"` when a description is given,
otherwise the key. Jev answers carry no policy status; `confidence` is
`(K * p_max - 1) / (K - 1)`, clipped to [0, 1].
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from mimir.core.context import Context, JsonState
from mimir.core.decisions import MIN_LEVELS, Choice, DecisionSpec, Rate
from mimir.core.results import ChoiceResult, DecisionResult, RateResult

MAX_CHOICE_OPTIONS: Final = 255
MIN_SCORE_LEVELS: Final = 2
MAX_SCORE_LEVELS: Final = 10
NOUL_KEYS: Final = ("false", "true")
CAMEL_BOUNDARY: Final = re.compile(r"(?<=[a-z])(?=[A-Z])")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NoulCriteria(_Model):
    true: JsonValue = None
    false: JsonValue = None


class NoulQuestion(_Model):
    type: Literal["noul"]
    instructions: JsonValue
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Model):
    type: Literal["choice"]
    instructions: JsonValue
    criteria: dict[str, JsonValue] = Field(min_length=1, max_length=MAX_CHOICE_OPTIONS)


class ScoreQuestion(_Model):
    type: Literal["score"]
    instructions: JsonValue
    criteria: list[JsonValue] = Field(min_length=MIN_SCORE_LEVELS, max_length=MAX_SCORE_LEVELS)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(_Model):
    state: str | dict[str, JsonValue] | list[JsonValue]
    model: str
    questions: dict[str, Question]


class NoulAnswer(_Model):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(_Model):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(_Model):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(_Model):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(_Model):
    model: str
    answers: dict[str, Answer]
    usage: Usage


@dataclass(frozen=True, slots=True)
class Translated:
    """One question as a spec, with what `respond` needs to answer in Jev's shape."""

    kind: Literal["noul", "choice", "score"]
    spec: DecisionSpec
    legend: dict[str, str]


def as_text(value: JsonValue) -> str:
    """Return a string as is, and any other JSON value as compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def humanize(label: str) -> str:
    """Turn a snake_case or camelCase key into a capitalised phrase, e.g. `needs_review` gives
    `Needs review`."""
    phrase = label
    if "_" in label or CAMEL_BOUNDARY.search(label):
        phrase = " ".join(CAMEL_BOUNDARY.sub(" ", label).replace("_", " ").split()).lower()
    return phrase[:1].upper() + phrase[1:]


def option_text(key: str, description: JsonValue, label: str | None = None) -> str:
    """Return `"<Label>: <description>"`, or the key alone without a description."""
    if description is None or description == "":
        return key
    return f"{label or humanize(key)}: {as_text(description)}"


def translate_question(question: Question, labels: Mapping[str, str] | None = None) -> Translated:
    """Convert one question; `labels` optionally renames the noul options' shown labels."""
    instructions = as_text(question.instructions)
    if isinstance(question, NoulQuestion):
        criteria = question.criteria or NoulCriteria()
        descriptions = {"false": criteria.false, "true": criteria.true}
        options = {
            key: option_text(key, descriptions[key], None if labels is None else labels.get(key))
            for key in NOUL_KEYS
        }
        return Translated("noul", Choice(instructions, options), {})
    if isinstance(question, ChoiceQuestion):
        options = {key: option_text(key, value) for key, value in question.criteria.items()}
        return Translated("choice", Choice(instructions, options), {})
    levels = {str(index): as_text(level) for index, level in enumerate(question.criteria)}
    spec: DecisionSpec = (
        Rate(instructions, levels) if len(levels) >= MIN_LEVELS else Choice(instructions, levels)
    )
    return Translated("score", spec, levels)


def state_context(state: str | dict[str, JsonValue] | list[JsonValue]) -> Context:
    if isinstance(state, str):
        return Context.coerce(state)
    return Context.coerce(JsonState(state=state))


def translate(request: SystemOneRequest) -> tuple[Context, dict[str, Translated]]:
    """Convert a request into a context and one translated question per question id."""
    return state_context(request.state), {
        name: translate_question(question) for name, question in request.questions.items()
    }


def _option_probabilities(result: DecisionResult) -> dict[str, float]:
    """Probabilities over the listed options only, renormalised to sum to 1."""
    if not isinstance(result, ChoiceResult | RateResult):
        message = f"a {result.type} result cannot answer a Jev question"
        raise TypeError(message)
    total = sum(result.probabilities.values())
    return {key: value / total for key, value in result.probabilities.items()}


def jev_confidence(probabilities: Mapping[str, float]) -> float:
    count = len(probabilities)
    if count < MIN_SCORE_LEVELS:
        return 1.0
    peak = max(probabilities.values())
    return min(max((count * peak - 1) / (count - 1), 0.0), 1.0)


def answer(
    translated: Translated, result: DecisionResult
) -> NoulAnswer | ChoiceAnswer | ScoreAnswer:
    """Express one result as Jev's answer for its question type."""
    probabilities = _option_probabilities(result)
    if translated.kind == "noul":
        return NoulAnswer(noul=probabilities["true"])
    confidence = jev_confidence(probabilities)
    if translated.kind == "choice":
        top = max(probabilities, key=lambda key: probabilities[key])
        return ChoiceAnswer(choice=top, probabilities=probabilities, confidence=confidence)
    score = sum(int(key) * value for key, value in probabilities.items())
    return ScoreAnswer(
        score=score, legend=translated.legend, probabilities=probabilities, confidence=confidence
    )


def respond(
    translated: Mapping[str, Translated],
    results: Mapping[str, DecisionResult],
    model: str,
    input_tokens: int,
) -> SystemOneResponse:
    """Build Jev's response from one result per question id."""
    return SystemOneResponse(
        model=model,
        answers={name: answer(item, results[name]) for name, item in translated.items()},
        usage=Usage(input_tokens=input_tokens, output_tokens=0),
    )
