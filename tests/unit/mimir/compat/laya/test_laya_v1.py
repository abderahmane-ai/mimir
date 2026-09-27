import json
import math
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from mimir.compat.laya.v1 import QUESTIONS, Agent, answer, entropy_confidence, load, translate
from mimir.core.decisions import Choice, Rate
from mimir.core.errors import SignatureError
from mimir.core.results import ChoiceResult, RateResult, Status
from mimir.runtime.engine import Mimir

FIXTURE: Final = json.loads(
    (Path(__file__).parent / "fixtures" / "laya_0_3_20.json").read_text(encoding="utf-8")
)


def test_laya_questions_translate_including_lists_and_labels() -> None:
    questions = QUESTIONS.validate_python(FIXTURE["request"]["questions"])
    translated = {name: translate(question) for name, question in questions.items()}
    assert translated["team"].spec == Choice(
        "Which team should handle this?", {"billing": "billing", "security": "security"}
    )
    assert translated["urgency"].spec == Rate(
        "How urgent is this?", {"0": "low", "1": "medium", "2": "high"}
    )
    assert translated["refund"].spec == Choice(
        "Should we refund?", {"false": "false", "true": "A: refund now"}
    )


def _result(status: Status) -> ChoiceResult:
    return ChoiceResult(
        status=status,
        confidence=0.8,
        relevant_context=(),
        deferral=None,
        certificate=None,
        latency_ms=1.0,
        answer="billing",
        probabilities={"billing": 0.72, "security": 0.18},
        abstain_probability=0.1,
        prediction_set=None,
    )


def test_answers_have_laya_keys_for_every_type() -> None:
    documented = FIXTURE["response"]["answers"]
    questions = QUESTIONS.validate_python(FIXTURE["request"]["questions"])
    translated = {name: translate(question) for name, question in questions.items()}
    team = answer(translated["team"], _result(Status.DECIDED))
    assert set(team) == set(documented["team"])
    assert team["choice"] == "billing"
    assert team["probabilities"] == {"billing": 0.8, "security": 0.2}
    assert team["action"] == {"act_probability": 1.0}
    urgency = answer(
        translated["urgency"],
        RateResult(
            status=Status.DEFERRED,
            confidence=0.6,
            relevant_context=(),
            deferral=None,
            certificate=None,
            latency_ms=1.0,
            answer="0",
            probabilities={"0": 0.6, "1": 0.3, "2": 0.1},
            prediction_set=None,
        ),
    )
    assert set(urgency) == set(documented["urgency"])
    assert urgency["score"] == 0.5
    assert urgency["action"] == {"act_probability": 0.0}
    refund = answer(
        translated["refund"],
        ChoiceResult(
            status=Status.DEFERRED,
            confidence=0.7,
            relevant_context=(),
            deferral=None,
            certificate=None,
            latency_ms=1.0,
            answer="false",
            probabilities={"false": 0.7, "true": 0.3},
            abstain_probability=0.0,
            prediction_set=None,
        ),
    )
    assert set(refund) == set(documented["refund"])
    assert (refund["noul"], refund["confidence"]) == (0.3, 0.7)


def test_entropy_confidence() -> None:
    assert entropy_confidence([1.0]) == 1.0
    assert entropy_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert entropy_confidence([1.0, 0.0]) == pytest.approx(1.0)
    p = [0.8, 0.2]
    expected = 1 + sum(value * math.log(value) for value in p) / math.log(2)
    assert entropy_confidence(p) == pytest.approx(expected)


def test_invalid_laya_questions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        QUESTIONS.validate_python({"q": {"type": "choice", "instructions": "i", "criteria": []}})
    with pytest.raises(ValidationError):
        QUESTIONS.validate_python(
            {"q": {"type": "noul", "instructions": "i", "labels": {"maybe": "M"}}}
        )


def test_agent_predicts_in_laya_shape(release: Path) -> None:
    engine = Mimir.from_pretrained(str(release), device="cpu", allow_unsigned=True)
    agent = Agent(engine)
    documented = FIXTURE["response"]
    response = agent.predict(FIXTURE["request"]["state"], FIXTURE["request"]["questions"])
    assert set(response) == set(documented)
    answers = response["answers"]
    assert isinstance(answers, dict)
    assert set(answers) == set(documented["answers"])
    usage = response["usage"]
    assert isinstance(usage, dict)
    assert usage["output_tokens"] == 0
    assert isinstance(usage["input_tokens"], int)
    assert usage["input_tokens"] > 0
    batch = agent.predict_batch(["a", "b"], {"q": {"type": "noul", "instructions": "late?"}})
    assert len(batch) == 2


def test_load_verifies_the_release_like_from_pretrained(release: Path) -> None:
    with pytest.raises(SignatureError, match=r"manifest\.json\.sigstore is missing"):
        load(release, None)
