"""The package against the golden fixtures of the released model.

Each fixture holds a public record, the graph inputs the training pipeline laid out for it, the
PyTorch model's outputs, and its calibrated result under a reference policy. The package must
reproduce the inputs exactly and the outputs and results within measured tolerances.

Set `MIMIR_RELEASE_DIR` to the release directory and `MIMIR_FIXTURES_DIR` to the fixtures.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict
from tokenizers import Tokenizer

from mimir.core.context import Context
from mimir.core.decisions import Choice, DecisionSpec, Estimate, ModelType, MultiChoice, Rank, Rate
from mimir.core.results import (
    ChoiceResult,
    DecisionResult,
    EstimateResult,
    MultiChoiceResult,
    RankResult,
    RateResult,
    Status,
)
from mimir.policy.document import Policy, parse_policy, read_arrays
from mimir.runtime.engine import LoadedPolicy, Mimir
from mimir.runtime.layout import collate, tokenize
from mimir.runtime.readout import OUTPUTS
from mimir.runtime.rendering import render
from mimir.runtime.session import open_session

pytestmark = pytest.mark.integration

# Measured through this package on all 104 fixtures (fp32 graph, ONNX Runtime 1.30 CPU,
# Apple M4, 2026-09-27): worst output difference 4.5e-4 of max(1, |value|) (the workspace;
# utilities 3.6e-4), worst probability difference 2.3e-5, worst gate p-value difference 7.3e-5,
# and no decision or taken flag differed.
OUTPUT_TOLERANCE: Final = 1e-3
PROBABILITY_TOLERANCE: Final = 1e-4
RELEVANCE_TOLERANCE: Final = 1e-4
FIXTURES: Final = 104
RELEASE_ENVIRONMENT: Final = "MIMIR_RELEASE_DIR"
FIXTURES_ENVIRONMENT: Final = "MIMIR_FIXTURES_DIR"


class GoldenResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    probabilities: list[float]
    chosen: list[int]
    taken: bool
    prediction_set: list[int] | None
    evidence: list[float]


class GoldenRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    decision_type: ModelType
    question: str
    candidates: list[str]
    state: Context
    range: tuple[float, float] | None
    result: GoldenResult

    @property
    def spec(self) -> DecisionSpec:
        spec: DecisionSpec
        match self.decision_type:
            case "binary" | "categorical":
                spec = Choice(self.question, self.candidates)
            case "multilabel":
                spec = MultiChoice(self.question, self.candidates)
            case "ranking":
                spec = Rank(self.question, self.candidates)
            case "ordinal":
                spec = Rate(self.question, self.candidates)
            case "continuous":
                assert self.range is not None
                spec = Estimate(self.question, *self.range)
        assert spec.model_type == self.decision_type
        return spec


def _directory(name: str) -> Path:
    value = os.environ.get(name)
    if not value or not Path(value).is_dir():
        pytest.fail(f"set {name} to an existing directory; got {value!r}")
    return Path(value)


@dataclass(frozen=True, slots=True)
class Golden:
    records: list[GoldenRecord]
    arrays: dict[str, np.ndarray]
    policy: Policy


@pytest.fixture(scope="module")
def golden() -> Golden:
    root = _directory(FIXTURES_ENVIRONMENT)
    lines = (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
    records = [GoldenRecord.model_validate_json(line) for line in lines]
    document = json.loads((root / "policy.json").read_bytes())
    # The reference policy is fingerprinted to the PyTorch model, not to a runtime; the test
    # applies it directly, so its fingerprint is replaced with a valid placeholder.
    document["origin"] = "release"
    document["fingerprint"] = {
        "model": "golden",
        "revision": "fixtures",
        "variant": "fp32",
        "graph_sha256": "0" * 64,
        "weights_sha256": "0" * 64,
        "opset": 20,
        "onnxruntime": "1.30.0",
        "configurations": [
            {"provider": "CPUExecutionProvider", "options_sha256": "0" * 64, "hardware": "none"}
        ],
    }
    policy = parse_policy(json.dumps(document).encode(), read_arrays(root / "policy.npz"), "golden")
    assert len(records) == FIXTURES
    return Golden(records, read_arrays(root / "arrays.npz"), policy)


@pytest.fixture(scope="module")
def engine(golden: Golden) -> Mimir:
    release = _directory(RELEASE_ENVIRONMENT)
    loaded = Mimir.from_pretrained(str(release), device="cpu", allow_unsigned=True)
    snapshot = loaded.snapshot
    graph = snapshot.path(snapshot.config.variants[snapshot.variant].graph)
    return Mimir(
        snapshot=snapshot,
        session=open_session(graph, "cpu"),
        tokenizer=Tokenizer.from_file(str(snapshot.path(snapshot.config.tokenizer))),
        device="cpu",
        runtime=loaded.runtime,
        policy=LoadedPolicy(golden.policy, "certified"),
    )


def test_layout_reproduces_every_graph_input(golden: Golden, engine: Mimir) -> None:
    config = engine.snapshot.config
    tokenizer = Tokenizer.from_file(str(engine.snapshot.path(config.tokenizer)))
    mismatches: list[str] = []
    for record in golden.records:
        rendered = render(record.state, record.spec)
        batch = collate([tokenize(rendered, tokenizer, config)], config.layout.special_tokens)
        for name, value in batch.feed.items():
            expected = golden.arrays[f"{record.id}/inputs/{name}"]
            same = (
                expected.dtype == value.dtype
                and expected.shape == value.shape
                and np.array_equal(expected, value, equal_nan=expected.dtype.kind == "f")
            )
            if not same:
                mismatches.append(f"{record.id}/{name}")
        if not np.array_equal(golden.arrays[f"{record.id}/layout/key_segment"], batch.key_segments):
            mismatches.append(f"{record.id}/key_segment")
    assert mismatches == []


def test_graph_outputs_match_the_model(golden: Golden, engine: Mimir) -> None:
    config = engine.snapshot.config
    session = open_session(
        engine.snapshot.path(config.variants[engine.snapshot.variant].graph), "cpu"
    )
    names = [spec.name for spec in config.graph.inputs]
    worst = dict.fromkeys(OUTPUTS, 0.0)
    for record in golden.records:
        feed = {name: golden.arrays[f"{record.id}/inputs/{name}"] for name in names}
        found = dict(zip(OUTPUTS, session.run(list(OUTPUTS), feed), strict=True))
        for name in OUTPUTS:
            expected = golden.arrays[f"{record.id}/outputs/{name}"].astype(np.float64)
            value = np.asarray(found[name], dtype=np.float64)
            gap = np.abs(value - expected) / np.maximum(1.0, np.abs(expected))
            worst[name] = max(worst[name], float(gap.max(initial=0.0)))
    assert max(worst.values()) <= OUTPUT_TOLERANCE, worst


def _check_result(record: GoldenRecord, result: DecisionResult) -> None:
    expected = record.result
    candidates = record.candidates
    chosen = expected.chosen
    probabilities = np.array(expected.probabilities)
    assert (result.status != Status.DEFERRED) == expected.taken, record.id
    match result:
        case ChoiceResult():
            assert result.answer == (candidates[chosen[0]] if chosen else None)
            found = [
                *(result.probabilities[text] for text in candidates),
                result.abstain_probability,
            ]
            np.testing.assert_allclose(found, probabilities, atol=PROBABILITY_TOLERANCE)
            assert result.prediction_set is not None
            indices = [candidates.index(text) for text in result.prediction_set.options]
            if result.prediction_set.abstain:
                indices.append(len(candidates))
            assert sorted(indices) == expected.prediction_set
        case MultiChoiceResult():
            assert list(result.answer) == [candidates[index] for index in chosen]
            found = [result.probabilities[text] for text in candidates]
            np.testing.assert_allclose(found, probabilities, atol=PROBABILITY_TOLERANCE)
        case RankResult():
            assert result.answer[0] == candidates[chosen[0]]
            found = [result.probabilities[text] for text in candidates]
            np.testing.assert_allclose(found, probabilities, atol=PROBABILITY_TOLERANCE)
        case RateResult():
            assert result.answer == candidates[chosen[0]]
            found = [result.probabilities[text] for text in candidates]
            np.testing.assert_allclose(found, probabilities, atol=PROBABILITY_TOLERANCE)
            bounds = expected.prediction_set
            levels = [] if bounds is None else candidates[bounds[0] : bounds[1] + 1]
            assert list(result.prediction_set or ()) == levels
        case EstimateResult():
            assert record.range is not None
            low, high = record.range
            bins = probabilities.shape[0]
            mean = low + (high - low) * float(probabilities @ ((np.arange(bins) + 0.5) / bins))
            assert result.answer == pytest.approx(mean, abs=PROBABILITY_TOLERANCE * (high - low))
        case _:
            pytest.fail(f"{record.id}: unexpected result type {result.type}")


def test_results_match_the_reference_policy(golden: Golden, engine: Mimir) -> None:
    results = [
        engine.decide(record.state, record.spec, risk=0.01, alpha=0.1) for record in golden.records
    ]
    for record, result in zip(golden.records, results, strict=True):
        _check_result(record, result)


def test_relevance_matches_the_model_evidence(golden: Golden, engine: Mimir) -> None:
    for record in golden.records:
        result = engine.decide_uncertified(record.state, record.spec)
        state = render(record.state, record.spec).state
        expected = {
            segment.part: value
            for segment, value in zip(state, record.result.evidence, strict=True)
            if value > 0
        }
        found = {
            (item.kind, item.index, item.row): item.relevance for item in result.relevant_context
        }
        assert set(found) == {(part.kind, part.index, part.row) for part in expected}, record.id
        for part, value in expected.items():
            assert found[(part.kind, part.index, part.row)] == pytest.approx(
                value, abs=RELEVANCE_TOLERANCE
            )
