"""A drop-in for Laya 0.3.20: `load(...).predict(state, questions)`.

Requests use Laya's question format (Jev's, plus list-valued choice criteria and noul
`labels`). Answers follow Laya 0.3.20's shape, with values rounded to 4 decimals. MIMIR has no
separate act head, so `action.act_probability` is 1.0 when the certified decision may be acted
on (`DECIDED` or `ABSTAINED`) and 0.0 when it is deferred.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from mimir.compat.systemone.v1 import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    Translated,
    state_context,
    translate_question,
)
from mimir.core.results import ChoiceResult, DecisionResult, RateResult, Status
from mimir.core.wire import DEFAULT_RISK
from mimir.extras import engine_class

if TYPE_CHECKING:
    from mimir.runtime.engine import Mimir

DIGITS: Final = 4
MIN_OPTIONS: Final = 2
PROBABILITY_FLOOR: Final = 1e-12
DEFAULT_REPOSITORY: Final = "Mythologic/MIMIR-1"


class LayaNoulQuestion(NoulQuestion):
    labels: dict[Literal["false", "true"], str] | None = None


class LayaChoiceQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["choice"]
    instructions: JsonValue
    criteria: dict[str, JsonValue] | list[str] = Field(min_length=1)


LayaQuestion = LayaNoulQuestion | LayaChoiceQuestion | ScoreQuestion
QUESTIONS: Final = TypeAdapter[dict[str, LayaQuestion]](dict[str, LayaQuestion])
LayaState = str | dict[str, JsonValue] | list[JsonValue]


def translate(question: LayaQuestion) -> Translated:
    """Convert one Laya question into a MIMIR spec."""
    if isinstance(question, LayaChoiceQuestion):
        criteria = question.criteria
        mapping: dict[str, JsonValue] = (
            dict.fromkeys(criteria) if isinstance(criteria, list) else dict(criteria)
        )
        return translate_question(
            ChoiceQuestion(type="choice", instructions=question.instructions, criteria=mapping)
        )
    if isinstance(question, LayaNoulQuestion):
        labels = (
            None
            if question.labels is None
            else {str(key): value for key, value in question.labels.items()}
        )
        return translate_question(question, labels)
    return translate_question(question)


def entropy_confidence(probabilities: Sequence[float]) -> float:
    """Return Laya's `confidence`: `1 - H(p) / log(k)`, clipped to [0, 1]."""
    count = len(probabilities)
    if count < MIN_OPTIONS:
        return 1.0
    entropy = -sum(p * math.log(min(max(p, PROBABILITY_FLOOR), 1.0)) for p in probabilities)
    return min(max(1.0 - entropy / math.log(count), 0.0), 1.0)


def answer(translated: Translated, result: DecisionResult) -> dict[str, JsonValue]:
    """Express one result as a Laya 0.3.20 answer."""
    if not isinstance(result, ChoiceResult | RateResult):
        message = f"a {result.type} result cannot answer a Laya question"
        raise TypeError(message)
    total = sum(result.probabilities.values())
    probabilities = {key: value / total for key, value in result.probabilities.items()}
    values = list(probabilities.values())
    peak = max(values)
    action: dict[str, JsonValue] = {
        "act_probability": 0.0 if result.status == Status.DEFERRED else 1.0
    }
    if translated.kind == "noul":
        true = probabilities["true"]
        return {
            "type": "noul",
            "noul": round(true, DIGITS),
            "confidence": round(max(true, 1.0 - true), DIGITS),
            "answer_confidence": round(peak, DIGITS),
            "action": action,
        }
    rounded: dict[str, JsonValue] = {
        key: round(value, DIGITS) for key, value in probabilities.items()
    }
    common: dict[str, JsonValue] = {
        "probabilities": rounded,
        "confidence": round(entropy_confidence(values), DIGITS),
        "answer_confidence": round(peak, DIGITS),
        "action": action,
    }
    if translated.kind == "choice":
        top = max(probabilities, key=lambda key: probabilities[key])
        return {"type": "choice", "choice": top, **common}
    legend: dict[str, JsonValue] = dict(translated.legend)
    score = sum(int(key) * value for key, value in probabilities.items())
    return {"type": "score", "score": round(score, DIGITS), "legend": legend, **common}


class Agent:
    """Laya-compatible wrapper around a local `Mimir` engine."""

    def __init__(self, engine: "Mimir", risk: float = DEFAULT_RISK) -> None:
        self.engine = engine
        self.risk = risk

    def predict(self, state: LayaState, questions: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Answer every question about one state, in Laya's response shape."""
        return self.predict_batch([state], questions)[0]

    def predict_batch(
        self, states: Sequence[LayaState], questions: Mapping[str, JsonValue]
    ) -> list[dict[str, JsonValue]]:
        """Answer the same questions about each state; one response per state, in order."""
        parsed = QUESTIONS.validate_python(dict(questions))
        translated = {name: translate(question) for name, question in parsed.items()}
        contexts = [state_context(state) for state in states]
        items = [(context, item.spec) for context in contexts for item in translated.values()]
        results = self.engine.decide_many(items, risk=self.risk)
        model = self.engine.info().model
        responses: list[dict[str, JsonValue]] = []
        width = len(translated)
        for position, context in enumerate(contexts):
            own = results[position * width : (position + 1) * width]
            answers: dict[str, JsonValue] = {
                name: answer(item, result)
                for (name, item), result in zip(translated.items(), own, strict=True)
            }
            tokens = sum(
                self.engine.count_tokens(context, item.spec) for item in translated.values()
            )
            usage: dict[str, JsonValue] = {"input_tokens": tokens, "output_tokens": 0}
            responses.append({"model": model, "answers": answers, "usage": usage})
        return responses


def load(
    model_id_or_path: str | Path = DEFAULT_REPOSITORY,
    device: str | None = None,
    *,
    risk: float = DEFAULT_RISK,
    revision: str | None = None,
) -> Agent:
    """Load MIMIR behind Laya's `load` signature; `device` None means `auto`."""
    engine = engine_class("mimir.compat.laya.v1").from_pretrained(
        str(model_id_or_path), revision=revision, device=device or "auto"
    )
    return Agent(engine, risk)
