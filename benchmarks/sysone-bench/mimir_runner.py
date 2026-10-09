"""MIMIR adapter: local inference through the mimir-decisions package.

Decisions run in `certified` mode at risk 0.01. The contract has no abstention, so every decision
is scored as an answer, whatever its status:

* `decided`: the answer cleared the certified floor.
* `deferred`: the answer is below the floor. A choice keeps the model's own answer; a noul and a
  score use the model's probabilities.
* A choice whose answer is None (the model found no option applies) is projected to the argmax
  option, and its `no_option_applied` flag is true. Noul and score decisions never carry it.

Each prediction row records `status` and `no_option_applied` per decision in `_raw_model`; the
orchestrator aggregates them into the run's `metadata.json`.
"""

import os
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from typing import Any

from runners.base import BaseRunner, renormalize_choice_probabilities

ADAPTER_VERSION = "2"
DEFAULT_MODEL = "Mythologic/MIMIR-1"
DEFAULT_DEVICE = "cpu"
DEFAULT_RISK = 0.01
PACKAGE = "mimir-decisions"


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _option_text(key: str, description: object) -> str:
    if description is None or description == "":
        return key
    return f"{key}: {_text(description)}"


class MimirRunner(BaseRunner):
    name = "mimir"

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        from mimir import Mimir

        requested = os.environ.get("MIMIR_MODEL", DEFAULT_MODEL) if model is None else model
        target = os.environ.get("MIMIR_DEVICE", DEFAULT_DEVICE) if device is None else device
        self.engine = Mimir.from_pretrained(requested, device=target)
        info = self.engine.info()
        self._metadata = {
            "runner": self.name,
            "model": info.model,
            "revision": info.revision,
            "serving": "local",
            "device": info.device,
            "adapter_version": ADAPTER_VERSION,
            "package": PACKAGE,
            "package_version": importlib_metadata.version(PACKAGE),
            "torch": info.runtime.torch,
            "variant": info.variant,
            "certification": info.certification,
            "mode": "certified",
            "risk": DEFAULT_RISK,
        }

    def _spec(self, question: Mapping[str, Any], qid: str) -> Any:
        from mimir import Choice, Rate, YesNo

        question_type = question.get("type")
        instructions = _text(question.get("instructions"))
        if question_type == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, Mapping) or not criteria:
                raise ValueError(f"{qid}: choice criteria must be a non-empty mapping")
            options = {str(key): _option_text(str(key), value) for key, value in criteria.items()}
            return Choice(instructions, options)
        if question_type == "noul":
            return YesNo(instructions)
        if question_type == "score":
            criteria = question.get("criteria")
            if not isinstance(criteria, list) or not criteria:
                raise ValueError(f"{qid}: score criteria must be a non-empty list")
            levels = {str(index): _text(level) for index, level in enumerate(criteria)}
            return Rate(instructions, levels) if len(levels) >= 3 else Choice(instructions, levels)
        raise ValueError(f"{qid}: unsupported question type {question_type!r}")

    def _project(self, kind: str, result: Any, qid: str) -> tuple[dict[str, Any], bool]:
        probabilities = {str(key): float(value) for key, value in result.probabilities.items()}
        if kind == "noul":
            return {"type": "noul", "noul": probabilities["yes"]}, False
        if kind == "score":
            score = sum(int(key) * value for key, value in probabilities.items())
            answer = {"type": "score", "score": score, "confidence": max(probabilities.values())}
            return answer, False
        choice = result.answer
        no_option_applied = choice is None
        if no_option_applied:
            choice = max(probabilities, key=lambda key: probabilities[key])
        answer = {
            "type": "choice",
            "choice": choice,
            "probabilities": renormalize_choice_probabilities(probabilities, qid),
            "confidence": max(probabilities.values()),
        }
        return answer, no_option_applied

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        from mimir import Mode
        from mimir.compat.systemone.v1 import state_context

        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)
        context = state_context(copied_state)
        answers: dict[str, Any] = {}
        decisions: list[dict[str, Any]] = []
        input_tokens = 0
        for qid, question in copied_questions.items():
            if not isinstance(question, Mapping):
                raise TypeError(f"{qid}: question must be a mapping")
            question_type = question.get("type")
            spec = self._spec(question, qid)
            result = self.engine.decide(context, spec, mode=Mode.CERTIFIED, risk=DEFAULT_RISK)
            answers[qid], no_option_applied = self._project(_text(question_type), result, qid)
            decisions.append({"status": str(result.status), "no_option_applied": no_option_applied})
            input_tokens += self.engine.count_tokens(context, spec)
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "calls": len(answers),
            },
            "_raw_model": {"decisions": decisions},
        }
