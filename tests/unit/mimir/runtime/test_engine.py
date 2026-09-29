from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
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
    UncertifiedRuntimeError,
)
from mimir.core.results import ChoiceResult, EstimateResult, Status
from mimir.policy.assessment import assess
from mimir.policy.document import write_policy
from mimir.runtime import engine as engine_module
from mimir.runtime.engine import Mimir
from mimir.runtime.layout import collate, tokenize
from mimir.runtime.readout import OUTPUTS, as_float64, readout_at
from mimir.runtime.release import ReleaseConfig, Variant
from mimir.runtime.rendering import render
from mimir.runtime.session import Device, GraphSession, open_session, resolve_device
from tests.conftest import SPECIAL, build_policy, local_fingerprint, write_manifest, write_npz

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


def load(root: Path, **options: object) -> Mimir:
    policy = options.get("policy")
    return Mimir.from_pretrained(
        str(root),
        device="cpu",
        allow_unsigned=True,
        policy=policy if isinstance(policy, Path) else None,
    )


def test_a_matching_release_policy_is_certified(release: Path) -> None:
    info = load(release).info()
    assert (info.certification, info.policy, info.variant, info.device) == (
        "certified",
        "release",
        "fp32",
        "cpu",
    )
    assert info.risk_levels == (0.01,)
    assert info.revision == "local"


@pytest.mark.parametrize("spec", SPECS)
def test_decide_returns_the_result_type_of_each_spec(release: Path, spec: DecisionSpec) -> None:
    result = load(release).decide(TEXT, spec)
    assert result.type == spec.type
    assert result.relevant_context
    assert result.relevant_context[0].text == TEXT
    # The test graph spreads evidence evenly over the [CLS] key and the five state tokens.
    assert result.relevant_context[0].relevance == pytest.approx(5 / 6, abs=1e-6)


def test_graph_values_flow_into_calibrated_results(release: Path) -> None:
    # The test graph gives utility 2 to each of the K options and K / 4 to abstain.
    result = load(release).decide(TEXT, Choice("which team", ["billing", "security", "shipping"]))
    assert isinstance(result, ChoiceResult)
    expected = np.exp([2.0, 2.0, 2.0, 0.75])
    expected /= expected.sum()
    assert result.probabilities["billing"] == pytest.approx(expected[0], abs=1e-6)
    assert result.abstain_probability == pytest.approx(expected[3], abs=1e-6)
    assert result.status == Status.DEFERRED
    assert result.deferral is not None
    assert result.deferral.reason == "below_threshold"


def test_continuous_is_deferred_without_a_threshold(release: Path) -> None:
    result = load(release).decide(TEXT, Estimate("price", 0, 100))
    assert isinstance(result, EstimateResult)
    assert result.deferral is not None
    assert result.deferral.reason == "no_certified_threshold"
    assert result.answer == pytest.approx(50.0)


def test_risk_and_alpha_are_validated(release: Path) -> None:
    engine = load(release)
    with pytest.raises(RiskLevelError, match=r"choose one of \[0\.01\]"):
        engine.decide(TEXT, YesNo("q"), risk=0.02)
    with pytest.raises(ValueError, match="alpha"):
        engine.decide(TEXT, YesNo("q"), alpha=1.0)


def test_uncertified_works_without_a_policy(release_builder: Callable[..., Path]) -> None:
    engine = load(release_builder(with_policy=False))
    assert engine.info().certification == "none"
    assert engine.info().risk_levels == ()
    with pytest.raises(PolicyError, match="decide_uncertified"):
        engine.decide(TEXT, YesNo("q"))
    result = engine.decide_uncertified(TEXT, YesNo("q"))
    assert result.certificate is None
    assert result.status in {Status.DECIDED, Status.ABSTAINED}


def test_decide_many_keeps_order_across_batches(release: Path) -> None:
    engine = load(release)
    items = [(f"{TEXT} " * (index + 1), spec) for index, spec in enumerate(SPECS)]
    together = engine.decide_many(items)
    one_by_one = engine.decide_many(items, batch_size=1)
    assert [result.type for result in together] == [spec.type for spec in SPECS]
    assert [result.model_dump(exclude={"latency_ms"}) for result in together] == [
        result.model_dump(exclude={"latency_ms"}) for result in one_by_one
    ]


def test_the_engine_is_safe_to_share_across_threads(release: Path) -> None:
    engine = load(release)
    expected = engine.decide(TEXT, SPECS[0]).model_dump(exclude={"latency_ms"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        found = list(pool.map(lambda _: engine.decide(TEXT, SPECS[0]), range(16)))
    assert all(result.model_dump(exclude={"latency_ms"}) == expected for result in found)


def test_count_tokens_matches_the_layout(release: Path, tokenizer: Tokenizer) -> None:
    engine = load(release)
    config = ReleaseConfig.model_validate_json((release / "config.json").read_bytes())
    expected = tokenize(render(Context.coerce(TEXT), SPECS[1]), tokenizer, config).encoded_tokens
    assert engine.count_tokens(TEXT, SPECS[1]) == expected


def test_a_custom_policy_for_this_runtime_is_used(release: Path, tmp_path: Path) -> None:
    custom = build_policy(local_fingerprint(release), threshold=0.1, origin="custom")
    write_policy(custom, tmp_path / "custom.json", tmp_path / "custom.npz")
    engine = load(release, policy=tmp_path / "custom.json")
    assert engine.info().policy == "custom"
    result = engine.decide(TEXT, YesNo("is it late"))
    assert result.status == Status.DECIDED
    assert result.certificate is not None
    assert result.certificate.origin == "custom"


def test_a_policy_for_another_graph_is_refused(
    release: Path, release_builder: Callable[..., Path], tmp_path: Path
) -> None:
    other = release_builder("other")
    (other / "onnx" / "model.onnx_data").write_bytes(b"\x00" * 1024)
    fingerprint = local_fingerprint(other)
    write_policy(build_policy(fingerprint), tmp_path / "foreign.json", tmp_path / "foreign.npz")
    with pytest.raises(PolicyMismatchError, match="weights SHA-256"):
        load(release, policy=tmp_path / "foreign.json")


def _write_equivalence(root: Path, tokenizer: Tokenizer, *, flip: bool = False) -> None:
    config = ReleaseConfig.model_validate_json((root / "config.json").read_bytes())
    policy = build_policy(local_fingerprint(root, hardware="Tesla T4"))
    session = open_session(root / "onnx" / "model.onnx", "cpu")
    inputs: dict[str, np.ndarray] = {}
    expected: dict[str, np.ndarray] = {}
    for index, spec in enumerate(SPECS):
        request = f"e{index:04d}"
        tokenized = tokenize(render(Context.coerce(TEXT), spec), tokenizer, config)
        feed = collate([tokenized], SPECIAL).feed
        inputs |= {f"{request}/inputs/{name}": value for name, value in feed.items()}
        outputs = as_float64(dict(zip(OUTPUTS, session.run(list(OUTPUTS), feed), strict=True)))
        readout = readout_at(outputs, 0, spec.model_type, len(spec.option_ids))
        found = [assess(readout, policy, risk, config.default_alpha) for risk in config.risk_levels]
        chosen = found[0].decision.chosen
        if flip and index == 0:
            chosen = (1,)
        expected[f"fp32/{request}/chosen"] = np.array(chosen, dtype=np.int64)
        expected[f"fp32/{request}/taken"] = np.array([item.taken for item in found])
    (root / "equivalence").mkdir()
    write_npz(root / "equivalence" / "inputs.npz", inputs)
    write_npz(root / "equivalence" / "expected.npz", expected)
    write_manifest(root)


def test_unlisted_hardware_without_an_equivalence_set_is_refused(
    release_builder: Callable[..., Path], isolated_cache: Path
) -> None:
    root = release_builder(hardware="Tesla T4")
    with pytest.raises(IntegrityError, match="no equivalence set"):
        load(root)
    assert not isolated_cache.exists()


def test_unlisted_hardware_passing_the_equivalence_check_is_cached(
    release_builder: Callable[..., Path],
    tokenizer: Tokenizer,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = release_builder(hardware="Tesla T4")
    _write_equivalence(root, tokenizer)
    assert load(root).info().certification == "equivalent"
    assert len(list((isolated_cache / "equivalence").glob("*.json"))) == 1

    def refuse(*_: object) -> int:
        message = "the cached pass should skip the check"
        raise AssertionError(message)

    monkeypatch.setattr(engine_module, "check_equivalence", refuse)
    assert load(root).info().certification == "equivalent"


def test_unlisted_hardware_with_different_decisions_is_refused(
    release_builder: Callable[..., Path], tokenizer: Tokenizer, isolated_cache: Path
) -> None:
    root = release_builder(hardware="Tesla T4")
    _write_equivalence(root, tokenizer, flip=True)
    with pytest.raises(EquivalenceError, match="1 of 7 equivalence decisions differ"):
        load(root)
    assert not (isolated_cache / "equivalence").exists()


def _add_cuda_variant(release: Path, policy: str) -> None:
    config = ReleaseConfig.model_validate_json((release / "config.json").read_bytes())
    variants = dict(config.variants)
    variants["fp16"] = Variant(graph="onnx/model.onnx", policy=policy, devices=("cuda",))
    (release / "config.json").write_text(
        config.model_copy(update={"variants": variants}).model_dump_json(indent=2),
        encoding="utf-8",
    )
    write_manifest(release)


def test_auto_falls_back_to_cpu_when_cuda_cannot_open(
    release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_cuda_variant(release, "policy/fp32")

    def resolve(device: str) -> Device:
        return "cuda" if device == "auto" else resolve_device(device)

    opened: list[Device] = []

    def open_strict(path: Path, device: Device) -> GraphSession:
        opened.append(device)
        if device == "cuda":
            message = f"{path}: opened on ['CPUExecutionProvider'], not CUDAExecutionProvider"
            raise UncertifiedRuntimeError(message)
        return open_session(path, device)

    monkeypatch.setattr(engine_module, "resolve_device", resolve)
    monkeypatch.setattr(engine_module, "open_session", open_strict)
    engine = Mimir.from_pretrained(str(release), device="auto", allow_unsigned=True)
    assert opened == ["cuda", "cpu"]
    assert (engine.info().device, engine.info().variant) == ("cpu", "fp32")
    assert engine.info().certification == "certified"


def test_auto_prefers_the_certified_release_when_cuda_has_no_policy(
    release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_cuda_variant(release, "policy/fp16")

    def resolve(device: str) -> Device:
        return "cuda" if device == "auto" else resolve_device(device)

    opened: list[Device] = []

    def open_any(path: Path, device: Device) -> GraphSession:
        opened.append(device)
        return open_session(path, "cpu")

    monkeypatch.setattr(engine_module, "resolve_device", resolve)
    monkeypatch.setattr(engine_module, "open_session", open_any)
    engine = Mimir.from_pretrained(str(release), device="auto", allow_unsigned=True)
    assert opened == ["cuda", "cpu"]
    assert (engine.info().device, engine.info().variant) == ("cpu", "fp32")
    assert engine.info().certification == "certified"
    assert engine.info().risk_levels == (0.01,)


def test_an_explicit_cuda_variant_stays_strict(
    release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_cuda_variant(release, "policy/fp16")

    def resolve(device: str) -> Device:
        return "cuda" if device == "auto" else resolve_device(device)

    def open_any(path: Path, _device: Device) -> GraphSession:
        return open_session(path, "cpu")

    monkeypatch.setattr(engine_module, "resolve_device", resolve)
    monkeypatch.setattr(engine_module, "open_session", open_any)
    engine = Mimir.from_pretrained(str(release), device="auto", variant="fp16", allow_unsigned=True)
    assert (engine.info().device, engine.info().variant) == ("cuda", "fp16")
    assert engine.info().certification == "none"
    with pytest.raises(PolicyError, match="no policy for variant fp16"):
        engine.decide(TEXT, YesNo("q"))
