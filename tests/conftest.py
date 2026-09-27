"""Shared test builders: a word-level tokenizer, a small ONNX graph with the release contract,
a complete release directory with a policy bound to this machine, and the engine on it."""

import hashlib
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
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
from mimir.core.wire import InputLimits, ModelInfo, RuntimeInfo
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
    GraphContract,
    Layout,
    Limits,
    Manifest,
    ReleaseConfig,
    SpecialTokens,
    TensorSpec,
    Variant,
)
from mimir.runtime.session import CPU, options_sha256, runtime_version

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
INPUTS: Final = (
    ("chunk_ids", TensorProto.INT64, 2),
    ("chunk_mask", TensorProto.BOOL, 2),
    ("chunk_record", TensorProto.INT64, 1),
    ("question_length", TensorProto.INT64, 1),
    ("key_chunk", TensorProto.INT64, 2),
    ("key_position", TensorProto.INT64, 2),
    ("key_mask", TensorProto.BOOL, 2),
    ("candidate_ids", TensorProto.INT64, 2),
    ("candidate_mask", TensorProto.BOOL, 2),
    ("candidate_text_mask", TensorProto.BOOL, 2),
    ("candidate_index", TensorProto.INT64, 2),
    ("candidate_present", TensorProto.BOOL, 2),
    ("decision_type", TensorProto.INT64, 1),
    ("typed_kind", TensorProto.INT64, 1),
    ("typed_number", TensorProto.FLOAT, 1),
    ("typed_year", TensorProto.INT64, 1),
    ("typed_month", TensorProto.INT64, 1),
    ("typed_day", TensorProto.INT64, 1),
    ("typed_second", TensorProto.INT64, 1),
    ("typed_chunk", TensorProto.INT64, 1),
    ("typed_position", TensorProto.INT64, 1),
)
OUTPUTS: Final = (
    ("utilities", 2),
    ("thresholds", 2),
    ("abstain", 1),
    ("ordinal_score", 1),
    ("histogram_logits", 2),
    ("evidence", 2),
    ("workspace", 2),
)
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


def _tensor(name: str, element: int, rank: int) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, element, [f"{name}_{axis}" for axis in range(rank)])


def build_graph(path: Path) -> None:
    """Write a small graph with the release signature; its one large weight goes to
    `model.onnx_data` beside it.

    Utilities are 2 for present options, abstain is K / 4, the ordinal score K / 10, the
    histogram flat, evidence uniform over live keys, and the workspace constant K / 4.
    """
    initializers = [
        numpy_helper.from_array(np.full(256, 2.0, dtype=np.float32), "scale_table"),
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "quarter"),
        numpy_helper.from_array(np.array([0.1], dtype=np.float32), "tenth"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axis_one"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "start"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "end"),
        numpy_helper.from_array(np.array([HISTOGRAM_BINS], dtype=np.int64), "bins"),
        numpy_helper.from_array(np.array([WORKSPACE_WIDTH], dtype=np.int64), "width"),
    ]
    nodes = [
        helper.make_node("ReduceMax", ["scale_table"], ["scale"], keepdims=0),
        helper.make_node("Cast", ["candidate_present"], ["present"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["present", "scale"], ["utilities"]),
        helper.make_node("Slice", ["utilities", "start", "end", "axis_one"], ["thresholds"]),
        helper.make_node("ReduceSum", ["present", "axis_one"], ["count"], keepdims=0),
        helper.make_node("Mul", ["count", "quarter"], ["abstain"]),
        helper.make_node("Mul", ["count", "tenth"], ["ordinal_score"]),
        helper.make_node("Unsqueeze", ["abstain", "axis_one"], ["column"]),
        helper.make_node("Shape", ["abstain"], ["records"]),
        helper.make_node("Concat", ["records", "bins"], ["histogram_shape"], axis=0),
        helper.make_node("Expand", ["column", "histogram_shape"], ["histogram_logits"]),
        helper.make_node("Concat", ["records", "width"], ["workspace_shape"], axis=0),
        helper.make_node("Expand", ["column", "workspace_shape"], ["workspace"]),
        helper.make_node("Cast", ["key_mask"], ["keys"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["keys", "axis_one"], ["key_total"], keepdims=1),
        helper.make_node("Div", ["keys", "key_total"], ["evidence"]),
    ]
    graph = helper.make_graph(
        nodes,
        "release",
        [_tensor(name, element, rank) for name, element, rank in INPUTS],
        [_tensor(name, TensorProto.FLOAT, rank) for name, rank in OUTPUTS],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=512,
    )


def graph_contract(path: Path) -> GraphContract:
    model = onnx.load(str(path), load_external_data=False)
    return GraphContract(
        opset={"ai.onnx": 20},
        operators=tuple(sorted({f"ai.onnx::{node.op_type}" for node in model.graph.node})),
        inputs=tuple(
            TensorSpec(
                name=name, dtype=np.dtype(helper.tensor_dtype_to_np_dtype(kind)).name, rank=rank
            )
            for name, kind, rank in INPUTS
        ),
        outputs=tuple(TensorSpec(name=name, dtype="float32", rank=rank) for name, rank in OUTPUTS),
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_policy(
    fingerprint: Fingerprint, *, threshold: float | None = 0.5, origin: str = "release"
) -> Policy:
    """A policy over all six types: identity scaling, a one-centroid gate at the origin under an
    identity precision, and a threshold at risk 0.01 for every type but continuous."""
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
            thresholds={} if name == "continuous" else {"0.01": entry},
        )
        arrays[f"gate/{name}/centroids"] = np.zeros((1, WORKSPACE_WIDTH))
        arrays[f"gate/{name}/precision"] = np.eye(WORKSPACE_WIDTH)
        arrays[f"gate/{name}/reference"] = np.linspace(0.0, 100.0, 99)
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
            "gate_level": 0.01,
            "label_taken": 0.5,
            "decision_types": {name: policy.model_dump() for name, policy in types.items()},
        }
    )
    return Policy(document, arrays)


def local_fingerprint(root: Path, *, hardware: str | None = None) -> Fingerprint:
    return Fingerprint(
        model="vathosai/mimir-test",
        revision="local",
        variant="fp32",
        graph_sha256=sha256(root / "onnx" / "model.onnx"),
        weights_sha256=sha256(root / "onnx" / "model.onnx_data"),
        opset=20,
        onnxruntime=runtime_version(),
        configurations=(
            Configuration(
                provider=CPU,
                options_sha256=options_sha256(CPU),
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
    manifest = Manifest(format_version=1, files=files, loadable_by={"mimirai": ">=1,<2"})
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def build_release(root: Path, *, with_policy: bool = True, hardware: str | None = None) -> Path:
    """Write a complete, unsigned release directory for the fp32 variant on the CPU."""
    graph = root / "onnx" / "model.onnx"
    build_graph(graph)
    build_tokenizer().save(str(root / "tokenizer.json"))
    config = ReleaseConfig(
        format_version=1,
        variants={"fp32": Variant(graph="onnx/model.onnx", policy="policy/fp32", devices=("cpu",))},
        tokenizer="tokenizer.json",
        decision_types=MODEL_TYPES,
        risk_levels=(0.005, 0.01, 0.02, 0.05),
        default_alpha=0.1,
        limits=Limits(options=8, levels=6, context_tokens=400),
        layout=Layout(
            chunk_tokens=64, question_cap=16, crossing_tokens=256, special_tokens=SPECIAL
        ),
        graph=graph_contract(graph),
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
        "confidence": 0.9,
        "relevant_context": (),
        "deferral": None,
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
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        self.calls.append(Call(tuple(requests), risk, alpha, batch_size))
        return [result_for(spec) for _, spec in requests]

    def info(self) -> ModelInfo:
        return ModelInfo(
            model="recording",
            revision="r",
            variant="fp32",
            device="cpu",
            runtime=RuntimeInfo(
                onnxruntime="1.30.0", provider=CPU, options_sha256="0" * 64, hardware="test"
            ),
            certification="certified",
            policy="release",
            risk_levels=(0.01,),
            decision_types=MODEL_TYPES,
            limits=InputLimits(options=8, levels=6, context_tokens=400),
        )


@pytest.fixture
def tokenizer() -> Tokenizer:
    return build_tokenizer()


@pytest.fixture
def release(tmp_path: Path) -> Path:
    return build_release(tmp_path / "release")


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


def load_engine(root: Path) -> Mimir:
    """The engine on a release directory from `build_release`."""
    return Mimir.from_pretrained(str(root), device="cpu", allow_unsigned=True)


@pytest.fixture
def engine(release: Path) -> Mimir:
    return load_engine(release)


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setenv("MIMIR_CACHE", str(cache))
    return cache
