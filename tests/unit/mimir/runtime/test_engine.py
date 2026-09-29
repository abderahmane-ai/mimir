from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from mimir.core.context import Context
from mimir.core.decisions import (
    Choice,
    DecisionSpec,
    Estimate,
    MultiChoice,
    Rank,
    Rate,
    Verify,
    YesNo,
)
from mimir.core.errors import (
    EquivalenceError,
    IntegrityError,
    PolicyError,
    PolicyMismatchError,
    RiskLevelError,
)
from mimir.core.results import ChoiceResult, EstimateResult, Status
from mimir.core.wire import Mode
from mimir.policy.assessment import assess
from mimir.policy.document import write_policy
from mimir.runtime import engine as engine_module
from mimir.runtime.engine import Mimir
from mimir.runtime.layout import collate, tokenize
from mimir.runtime.readout import as_float64, readout_at
from mimir.runtime.release import ReleaseConfig
from mimir.runtime.rendering import render
from mimir.runtime.session import resolve_device
from tests.conftest import (
    SPECIAL,
    build_policy,
    load_engine,
    local_fingerprint,
    scripted_outputs,
    write_manifest,
    write_npz,
)

TEXT = "my card was charged twice"
SPECS: list[DecisionSpec] = [
    Choice("which team", ["billing", "security", "shipping"]),
    YesNo("is it late"),
    Verify("the invoice is late"),
    MultiChoice("which", ["refund", "order"]),
    Rank("best", ["refund", "order"]),
    Rate("rate", ["low", "medium", "high"]),
    Estimate("price", 0, 100),
]


def test_a_matching_release_policy_is_certified(engine: Mimir) -> None:
    info = engine.info()
    assert (info.certification, info.policy, info.variant, info.device) == (
        "certified",
        "release",
        "fp32",
        "cpu",
    )
    assert info.risk_levels == (0.01,)
    assert info.revision == "local"
    assert info.runtime.hardware != "unknown"


@pytest.mark.parametrize("spec", SPECS)
def test_decide_answers_every_spec_in_standard_mode(engine: Mimir, spec: DecisionSpec) -> None:
    result = engine.decide(TEXT, spec)
    assert result.type == spec.type
    assert result.status in (Status.DECIDED, Status.ABSTAINED)
    assert result.actionable is True
    assert result.relevant_context
    assert result.relevant_context[0].text == TEXT
    # The scripted outputs spread evidence evenly over the [CLS] key and the five state tokens.
    assert result.relevant_context[0].relevance == pytest.approx(5 / 6, abs=1e-6)


def test_scripted_values_flow_into_calibrated_results(engine: Mimir) -> None:
    # Scripted outputs give utility 2 to each of the K options and K / 4 to abstain.
    result = engine.decide(TEXT, Choice("which team", ["billing", "security", "shipping"]))
    assert isinstance(result, ChoiceResult)
    expected = np.exp([2.0, 2.0, 2.0, 0.75])
    expected /= expected.sum()
    assert result.probabilities["billing"] == pytest.approx(expected[0], abs=1e-6)
    assert result.abstain_probability == pytest.approx(expected[3], abs=1e-6)
    assert result.status == Status.DECIDED
    assert result.answer == "billing"
    assert result.certified is False
    assert result.certificate is None


def test_continuous_is_decided_without_a_threshold(engine: Mimir) -> None:
    result = engine.decide(TEXT, Estimate("price", 0, 100))
    assert isinstance(result, EstimateResult)
    assert result.status == Status.DECIDED
    assert result.answer == pytest.approx(50.0)
    assert result.certified is False


def test_threshold_mode_defers_below_the_floor(engine: Mimir) -> None:
    spec = Choice("which team", ["billing", "security", "shipping"])
    assert engine.decide(TEXT, spec, mode=Mode.THRESHOLD, min_confidence=0.99).status == (
        Status.DEFERRED
    )
    decided = engine.decide(TEXT, spec, mode=Mode.THRESHOLD, min_confidence=0.1)
    assert decided.status == Status.DECIDED
    assert decided.answer == "billing"


def test_certified_mode_defers_below_the_threshold(engine: Mimir) -> None:
    spec = Choice("which team", ["billing", "security", "shipping"])
    result = engine.decide(TEXT, spec, mode=Mode.CERTIFIED)
    assert result.status == Status.DEFERRED
    assert result.answer == "billing"
    assert result.certified is False


def test_risk_alpha_and_mode_are_validated(engine: Mimir) -> None:
    with pytest.raises(RiskLevelError, match=r"choose one of \[0\.01\]"):
        engine.decide(TEXT, YesNo("q"), risk=0.02)
    with pytest.raises(ValueError, match="alpha"):
        engine.decide(TEXT, YesNo("q"), alpha=1.0)
    with pytest.raises(ValueError, match="min_confidence"):
        engine.decide(TEXT, YesNo("q"), mode=Mode.THRESHOLD)
    with pytest.raises(ValueError, match="min_confidence"):
        engine.decide(TEXT, YesNo("q"), mode=Mode.CERTIFIED, min_confidence=0.5)


def test_uncertified_works_without_a_policy(
    release_builder: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = load_engine(release_builder(with_policy=False), monkeypatch)
    assert engine.info().certification == "none"
    assert engine.info().risk_levels == ()
    with pytest.raises(PolicyError, match="decide_uncertified"):
        engine.decide(TEXT, YesNo("q"))
    result = engine.decide_uncertified(TEXT, YesNo("q"))
    assert result.certificate is None
    assert result.status in {Status.DECIDED, Status.ABSTAINED}


def test_decide_many_keeps_order_across_batches(engine: Mimir) -> None:
    items = [(f"{TEXT} " * (index + 1), spec) for index, spec in enumerate(SPECS)]
    together = engine.decide_many(items)
    one_by_one = engine.decide_many(items, batch_size=1)
    assert [result.type for result in together] == [spec.type for spec in SPECS]
    assert [result.model_dump(exclude={"latency_ms"}) for result in together] == [
        result.model_dump(exclude={"latency_ms"}) for result in one_by_one
    ]


def test_the_engine_is_safe_to_share_across_threads(engine: Mimir) -> None:
    expected = engine.decide(TEXT, SPECS[0]).model_dump(exclude={"latency_ms"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        found = list(pool.map(lambda _: engine.decide(TEXT, SPECS[0]), range(16)))
    assert all(result.model_dump(exclude={"latency_ms"}) == expected for result in found)


def test_count_tokens_matches_the_layout(
    engine: Mimir, release: Path, tokenizer: Tokenizer
) -> None:
    config = ReleaseConfig.model_validate_json((release / "config.json").read_bytes())
    expected = tokenize(render(Context.coerce(TEXT), SPECS[1]), tokenizer, config).encoded_tokens
    assert engine.count_tokens(TEXT, SPECS[1]) == expected


def test_a_custom_policy_for_this_runtime_is_used(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = build_policy(local_fingerprint(release), threshold=0.1, origin="custom")
    write_policy(custom, tmp_path / "custom.json", tmp_path / "custom.npz")
    engine = load_engine(release, monkeypatch, policy=tmp_path / "custom.json")
    assert engine.info().policy == "custom"
    result = engine.decide(TEXT, YesNo("is it late"))
    assert result.status == Status.DECIDED
    assert result.certificate is not None
    assert result.certificate.origin == "custom"


def test_a_policy_for_other_weights_is_refused(
    release: Path,
    release_builder: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = release_builder("other")
    (other / "weights" / "model.safetensors").write_bytes(b"\x00" * 1024)
    fingerprint = local_fingerprint(other)
    write_policy(build_policy(fingerprint), tmp_path / "foreign.json", tmp_path / "foreign.npz")
    with pytest.raises(PolicyMismatchError, match="weights SHA-256"):
        load_engine(release, monkeypatch, policy=tmp_path / "foreign.json")


def _write_equivalence(root: Path, tokenizer: Tokenizer, *, flip: bool = False) -> None:
    # The engine fixture patches the model run, so these expected outputs come from the same
    # scripted outputs the engine will read back.
    config = ReleaseConfig.model_validate_json((root / "config.json").read_bytes())
    policy = build_policy(local_fingerprint(root, hardware="Tesla T4"))
    inputs: dict[str, np.ndarray] = {}
    expected: dict[str, np.ndarray] = {}
    for index, spec in enumerate(SPECS):
        request = f"e{index:04d}"
        tokenized = tokenize(render(Context.coerce(TEXT), spec), tokenizer, config)
        feed = collate([tokenized], SPECIAL).feed
        inputs |= {f"{request}/inputs/{name}": value for name, value in feed.items()}
        outputs = as_float64(
            {name: np.asarray(value) for name, value in scripted_outputs(feed).items()}
        )
        readout = readout_at(outputs, 0, spec.model_type, len(spec.option_ids))
        found = [assess(readout, policy, risk, config.default_alpha) for risk in config.risk_levels]
        chosen = found[0].decision.chosen
        if flip and index == 0:
            chosen = (1,)
        expected[f"fp32/{request}/chosen"] = np.array(chosen, dtype=np.int64)
        expected[f"fp32/{request}/taken"] = np.array([item.certified for item in found])
    (root / "equivalence").mkdir()
    write_npz(root / "equivalence" / "inputs.npz", inputs)
    write_npz(root / "equivalence" / "expected.npz", expected)
    write_manifest(root)


def test_unlisted_hardware_without_an_equivalence_set_is_refused(
    release_builder: Callable[..., Path],
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = release_builder(hardware="Tesla T4")
    with pytest.raises(IntegrityError, match="no equivalence set"):
        load_engine(root, monkeypatch)
    assert not isolated_cache.exists()


def test_unlisted_hardware_passing_the_equivalence_check_is_cached(
    release_builder: Callable[..., Path],
    tokenizer: Tokenizer,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = release_builder(hardware="Tesla T4")
    _write_equivalence(root, tokenizer)
    assert load_engine(root, monkeypatch).info().certification == "equivalent"
    assert len(list((isolated_cache / "equivalence").glob("*.json"))) == 1

    def refuse(*_: object) -> int:
        message = "the cached pass should skip the check"
        raise AssertionError(message)

    monkeypatch.setattr(engine_module, "check_equivalence", refuse)
    assert load_engine(root, monkeypatch).info().certification == "equivalent"


def test_unlisted_hardware_with_different_decisions_is_refused(
    release_builder: Callable[..., Path],
    tokenizer: Tokenizer,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = release_builder(hardware="Tesla T4")
    _write_equivalence(root, tokenizer, flip=True)
    with pytest.raises(EquivalenceError, match="1 of 7 equivalence decisions differ"):
        load_engine(root, monkeypatch)
    assert not (isolated_cache / "equivalence").exists()


def test_resolve_device_follows_torch_cuda_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert resolve_device("cuda") == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    with pytest.raises(ValueError, match="needs a CUDA torch build"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="not one of"):
        resolve_device("tpu")
