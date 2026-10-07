"""Shared test builders: a word-level tokenizer, scripted model outputs, a complete release
directory with a policy bound to this machine, and the engine on it."""

import hashlib
import os
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from mimir.core.context import Context
from mimir.core.decider import Decider
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
from mimir.core.results import (
    ChoiceResult,
    DecisionResult,
    EstimateResult,
    MultiChoiceResult,
    RankResult,
    RateResult,
    Status,
    VerifyResult,
    YesNoResult,
)
from mimir.core.wire import InputLimits, Mode, ModelInfo, RuntimeInfo
from mimir.policy.document import (
    CertifiedThreshold,
    Configuration,
    Fingerprint,
    Policy,
    PolicyDocument,
    Scaling,
    TypePolicy,
    write_policy,
)
from mimir.runtime.engine import Mimir
from mimir.runtime.hardware import hardware_name
from mimir.runtime.release import (
    Layout,
    Limits,
    Manifest,
    ReleaseConfig,
    SpecialTokens,
    Variant,
)
from mimir.runtime.session import Array, Floats, LoadedModel, torch_version

SPECIAL: Final = SpecialTokens(cls=0, sep=1, pad=2, mask=3, newline=4)
UNKNOWN: Final = "[UNK]"
WORDS: Final = [
    "the",
    "which",
    "team",
    "should",
    "handle",
    "this",
    "ticket",
    "billing",
    "security",
    "shipping",
    "my",
    "card",
    "was",
    "charged",
    "twice",
    "refund",
    "order",
    "is",
    "it",
    "true",
    "that",
    "customer",
    "plan",
    "seats",
    "invoice",
    "total",
    "due",
    "date",
    "late",
    "yes",
    "no",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "and",
    "or",
    "not",
    "how",
    "many",
    "what",
    "rate",
    "estimate",
    "price",
    "rank",
    "best",
    "answer",
    "question",
    "passage",
    "table",
    "row",
    "field",
    "value",
    "level",
    "low",
    "medium",
    "high",
    "urgent",
    "claim",
    "supported",
    "evidence",
    ".",
    ",",
    ":",
    "|",
    "?",
    "!",
    "$",
    "%",
    "-",
    "1",
    "2",
    "3",
    "4",
    "5",
    "10",
    "100",
    "1200",
    "2026",
    "03",
    "04",
    "march",
]
WORKSPACE_WIDTH: Final = 4
HISTOGRAM_BINS: Final = 64
MODEL_TYPES: Final = ("binary", "categorical", "multilabel", "ranking", "ordinal", "continuous")


def build_tokenizer() -> Tokenizer:
    """A lowercasing word-level tokenizer whose special ids match `SPECIAL`."""
    vocab = {"[CLS]": 0, "[SEP]": 1, "[PAD]": 2, "[MASK]": 3, "\n": 4, UNKNOWN: 5}
    for word in WORDS:
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token=UNKNOWN))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def scripted_outputs(feed: Mapping[str, Array]) -> dict[str, Floats]:
    """Deterministic model outputs for a collated feed.

    Utilities are 2 for present options, abstain is K / 4, the ordinal score K / 10, the
    histogram flat, evidence uniform over live keys, and the workspace constant K / 4.
    """
    present = np.asarray(feed["candidate_present"], dtype=np.bool_)
    count = present.sum(axis=1, keepdims=True).astype(np.float32)
    utilities = (present * 2.0).astype(np.float32)
    keys = np.asarray(feed["key_mask"], dtype=np.float32)
    total = keys.sum(axis=1, keepdims=True)
    return {
        "utilities": utilities,
        "thresholds": utilities[:, :-1].astype(np.float32),
        "abstain": (count / 4).reshape(-1).astype(np.float32),
        "ordinal_score": (count / 10).reshape(-1).astype(np.float32),
        "histogram_logits": np.broadcast_to(
            (count / 4).reshape(-1, 1), (count.shape[0], HISTOGRAM_BINS)
        ).astype(np.float32),
        "evidence": (keys / np.maximum(total, 1)).astype(np.float32),
        "workspace": np.broadcast_to(
            (count / 4).reshape(-1, 1), (count.shape[0], WORKSPACE_WIDTH)
        ).astype(np.float32),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_policy(
    fingerprint: Fingerprint, *, threshold: float | None = 0.5, origin: str = "release"
) -> Policy:
    """A policy over all six types: identity scaling and one threshold at risks 0.01 and 0.05
    for every type but continuous."""
    types: dict[str, TypePolicy] = {}
    arrays: dict[str, np.ndarray] = {}
    for name in MODEL_TYPES:
        parameters = (1.0, 0.0) if name == "multilabel" else (1.0,)
        conformal: Literal["bucket", "type"] | None = None
        if name in {"binary", "categorical"}:
            conformal = "bucket"
        elif name in {"ordinal", "continuous"}:
            conformal = "type"
        entry = CertifiedThreshold(
            threshold=threshold,
            records=1000,
            taken=500,
            errors=2,
            p_value=0.001,
            coverage=0.5,
            coverage_interval=(0.47, 0.53),
        )
        types[name] = TypePolicy(
            scaling=(Scaling(bucket=0, parameters=parameters),),
            conformal=conformal,
            thresholds={} if name == "continuous" else {"0.01": entry, "0.05": entry},
        )
        if conformal == "bucket":
            arrays[f"conformal/{name}/0"] = np.linspace(0.0, 1.0, 99)
        elif conformal == "type":
            arrays[f"conformal/{name}"] = np.linspace(0.0, 1.0, 99)
    document = PolicyDocument.model_validate(
        {
            "format_version": 1,
            "origin": origin,
            "fingerprint": fingerprint.model_dump(),
            "confidence": 0.95,
            "label_taken": 0.5,
            "decision_types": {name: policy.model_dump() for name, policy in types.items()},
        }
    )
    return Policy(document, arrays)


def local_fingerprint(root: Path, *, hardware: str | None = None) -> Fingerprint:
    return Fingerprint(
        model="mimir-test",
        revision="local",
        variant="fp32",
        weights_sha256=sha256(root / "weights" / "model.safetensors"),
        torch=torch_version(),
        configurations=(
            Configuration(
                device="cpu",
                hardware=hardware_name("cpu") if hardware is None else hardware,
            ),
        ),
    )


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in arrays.items():
            with archive.open(f"{name}.npy", "w") as member:
                np.lib.format.write_array(member, value, allow_pickle=False)


def write_manifest(root: Path) -> None:
    files = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.startswith("manifest.json")
    }
    manifest = Manifest(format_version=1, files=files, loadable_by={"mimir-decisions": ">=1,<2"})
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def build_release(root: Path, *, with_policy: bool = True, hardware: str | None = None) -> Path:
    """Write a complete, unsigned release directory for the fp32 variant on the CPU.

    The weights are placeholder bytes: tests never load them, the engine runs scripted
    outputs instead (see `load_engine`).
    """
    weights = root / "weights" / "model.safetensors"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"placeholder weights")
    (root / "encoder_config.json").write_text("{}", encoding="utf-8")
    build_tokenizer().save(str(root / "tokenizer.json"))
    config = ReleaseConfig(
        format_version=1,
        variants={
            "fp32": Variant(
                weights="weights/model.safetensors", policy="policy/fp32", devices=("cpu",)
            )
        },
        tokenizer="tokenizer.json",
        decision_types=MODEL_TYPES,
        risk_levels=(0.005, 0.01, 0.02, 0.05),
        default_alpha=0.1,
        limits=Limits(options=8, levels=6, context_tokens=400),
        layout=Layout(
            chunk_tokens=64, question_cap=16, crossing_tokens=256, special_tokens=SPECIAL
        ),
    )
    (root / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    if with_policy:
        policy = build_policy(local_fingerprint(root, hardware=hardware))
        write_policy(policy, root / "policy" / "fp32.json", root / "policy" / "fp32.npz")
    write_manifest(root)
    return root


def result_for(spec: DecisionSpec, status: Status = Status.DECIDED) -> DecisionResult:
    """A valid result for any spec: the first option chosen with probability 0.9."""
    ids = spec.option_ids
    probabilities = {
        option: (0.9 if index == 0 else 0.1 / len(ids)) for index, option in enumerate(ids)
    }
    common = {
        "status": status,
        "actionable": status in (Status.DECIDED, Status.ABSTAINED),
        "certified": False,
        "confidence": 0.9,
        "relevant_context": (),
        "certificate": None,
        "latency_ms": 1.0,
    }
    match spec:
        case Choice():
            return ChoiceResult.model_validate(
                {
                    **common,
                    "answer": ids[0],
                    "probabilities": probabilities,
                    "abstain_probability": 0.0,
                    "prediction_set": None,
                }
            )
        case YesNo():
            return YesNoResult.model_validate(
                {
                    **common,
                    "answer": False,
                    "probabilities": probabilities,
                    "abstain_probability": 0.0,
                    "prediction_set": None,
                }
            )
        case Verify():
            return VerifyResult.model_validate(
                {
                    **common,
                    "answer": "supported",
                    "probabilities": probabilities,
                    "abstain_probability": 0.0,
                    "prediction_set": None,
                }
            )
        case MultiChoice():
            return MultiChoiceResult.model_validate(
                {**common, "answer": (ids[0],), "probabilities": probabilities}
            )
        case Rank():
            return RankResult.model_validate(
                {**common, "answer": ids, "probabilities": probabilities}
            )
        case Rate():
            return RateResult.model_validate(
                {**common, "answer": ids[0], "probabilities": probabilities, "prediction_set": None}
            )
        case Estimate():
            return EstimateResult.model_validate(
                {
                    **common,
                    "answer": (spec.low + spec.high) / 2,
                    "unit": spec.unit,
                    "interval": (spec.low, spec.high),
                }
            )


@dataclass(frozen=True, slots=True)
class Call:
    requests: tuple[tuple[Context, DecisionSpec], ...]
    mode: Mode
    min_confidence: float | None
    risk: float | None
    alpha: float | None
    batch_size: int | None


class RecordingDecider(Decider):
    """A Decider that records each `_run` call and answers with `result_for`."""

    def __init__(self) -> None:
        self.calls: list[Call] = []

    def _run(
        self,
        requests: Sequence[tuple[Context, DecisionSpec]],
        *,
        mode: Mode,
        min_confidence: float | None,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        self.calls.append(Call(tuple(requests), mode, min_confidence, risk, alpha, batch_size))
        return [result_for(spec) for _, spec in requests]

    def info(self) -> ModelInfo:
        return ModelInfo(
            model="recording",
            revision="r",
            variant="fp32",
            device="cpu",
            runtime=RuntimeInfo(torch="0", hardware="test"),
            certification="certified",
            policy="release",
            risk_levels=(0.01,),
            decision_types=MODEL_TYPES,
            limits=InputLimits(options=8, levels=6, context_tokens=400),
        )


class YesNoDecider(RecordingDecider):
    """A `RecordingDecider` answering every `YesNo` with `answer` at `status`."""

    def __init__(self, *, answer: bool | None, status: Status = Status.DECIDED) -> None:
        super().__init__()
        self.answer = answer
        self.status = status

    def _run(
        self,
        requests: Sequence[tuple[Context, DecisionSpec]],
        *,
        mode: Mode,
        min_confidence: float | None,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        results = super()._run(
            requests,
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
            batch_size=batch_size,
        )
        update = {
            "answer": self.answer,
            "status": self.status,
            "actionable": self.status in (Status.DECIDED, Status.ABSTAINED),
            "certified": False,
        }
        return [
            YesNoResult.model_validate({**result.model_dump(), **update})
            if isinstance(result, YesNoResult)
            else result
            for result in results
        ]


@pytest.fixture
def tokenizer() -> Tokenizer:
    return build_tokenizer()


@pytest.fixture
def release(tmp_path: Path) -> Path:
    return build_release(tmp_path / "release")


@pytest.fixture(scope="module")
def hub_release() -> Path:
    """The real release tree; subprocess servers load real weights, which no patch reaches."""
    value = os.environ.get("MIMIR_RELEASE_DIR")
    if not value or not Path(value).is_dir():
        pytest.fail(f"set MIMIR_RELEASE_DIR to an existing directory; got {value!r}")
    return Path(value)


@pytest.fixture
def release_builder(tmp_path: Path) -> Callable[..., Path]:
    def build(name: str = "release", **options: object) -> Path:
        with_policy = bool(options.get("with_policy", True))
        hardware = options.get("hardware")
        return build_release(
            tmp_path / name,
            with_policy=with_policy,
            hardware=hardware if isinstance(hardware, str) else None,
        )

    return build


class BatchRecorder:
    """A `BatchObserver` keeping every batch's results."""

    def __init__(self) -> None:
        self.batches: list[list[DecisionResult]] = []

    def observe_batch(self, results: Sequence[DecisionResult]) -> None:
        self.batches.append(list(results))


def fake_run(
    session: LoadedModel, feed: Mapping[str, Array], question_cap: int
) -> dict[str, Floats]:
    """Scripted model outputs, ignoring the session; `question_cap` is read by the engine."""
    del session, question_cap
    return {
        name: np.asarray(value, dtype=np.float32) for name, value in scripted_outputs(feed).items()
    }


def count_model_runs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """A list that grows by one entry per model run through the session module."""
    import mimir.runtime.session as session_module

    runs: list[int] = []
    original = session_module.run

    def counted(
        session: LoadedModel, feed: Mapping[str, Array], question_cap: int
    ) -> dict[str, Floats]:
        runs.append(1)
        return original(session, feed, question_cap)

    monkeypatch.setattr(session_module, "run", counted)
    return runs


def load_engine(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, policy: Path | None = None
) -> Mimir:
    """The engine on a release directory from `build_release`, running scripted outputs."""
    import mimir.runtime.engine as engine_module
    import mimir.runtime.session as session_module

    monkeypatch.setattr(engine_module, "load_model", lambda *_: None)
    monkeypatch.setattr(session_module, "run", fake_run)
    return Mimir.from_pretrained(str(root), device="cpu", allow_unsigned=True, policy=policy)


@pytest.fixture
def engine(release: Path, monkeypatch: pytest.MonkeyPatch) -> Mimir:
    return load_engine(release, monkeypatch)


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setenv("MIMIR_CACHE", str(cache))
    return cache


EXAMPLE_TOOLS: Final = Path(__file__).parents[1] / "examples" / "tools.yaml"
# The MCP SDK 1 environment holds clients only; `make test` points it at the main
# environment's `mimir` through MIMIR_SERVER.
MIMIR_SERVER: Final = os.environ.get("MIMIR_SERVER", str(Path(sys.executable).parent / "mimir"))


def mcp_server_arguments(release: Path) -> list[str]:
    """`mimir mcp` arguments serving the examples' tools file on a test release."""
    return [
        "mcp",
        "--tools",
        str(EXAMPLE_TOOLS),
        "--model",
        str(release),
        "--allow-unsigned",
        "--device",
        "cpu",
    ]
