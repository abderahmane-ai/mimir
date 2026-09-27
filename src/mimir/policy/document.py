"""Policy files: a JSON document and an npz of arrays, one pair per variant.

The document contains, per model decision type, the scaling for each option-count bucket, the
conformal score keying, and the certified threshold for each risk level. It also contains the
fingerprint of the configuration it was certified on (see `mimir.policy.binding`).

Array names:

- `gate/<type>/centroids`, `gate/<type>/precision`, `gate/<type>/reference`;
- `conformal/<type>/<bucket>` or `conformal/<type>`, sorted scores.
"""

import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mimir.core.decisions import ModelType
from mimir.core.errors import FormatVersionError, PolicyError
from mimir.policy.distributions import LABEL_TAKEN

Floats = npt.NDArray[np.float64]

FORMAT_VERSION: Final = 1
MATRIX_RANK: Final = 2


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Scaling(_Frozen):
    """Scaling parameters for one option-count bucket, `ceil(log2(K))`."""

    bucket: int = Field(ge=0)
    parameters: tuple[float, ...] = Field(min_length=1, max_length=2)


class CertifiedThreshold(_Frozen):
    """A selective threshold and its test results. `threshold` is None if none was certified."""

    threshold: float | None
    records: int = Field(ge=0)
    taken: int = Field(ge=0)
    errors: int = Field(ge=0)
    p_value: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    coverage_interval: tuple[float, float] | None


class TypePolicy(_Frozen):
    scaling: tuple[Scaling, ...] = Field(min_length=1)
    conformal: Literal["bucket", "type"] | None
    thresholds: dict[str, CertifiedThreshold]

    @model_validator(mode="after")
    def _check_buckets_ascend(self) -> Self:
        buckets = [entry.bucket for entry in self.scaling]
        if buckets != sorted(set(buckets)):
            message = f"scaling buckets {buckets} are not strictly ascending"
            raise ValueError(message)
        return self

    def scaling_of(self, count: int) -> Scaling:
        """Return the scaling for K options: the largest bucket not above `ceil(log2(K))`, or
        the smallest bucket if none is."""
        bucket = max(count - 1, 0).bit_length()
        under = [entry for entry in self.scaling if entry.bucket <= bucket]
        return under[-1] if under else self.scaling[0]


class Configuration(_Frozen):
    """A certified execution configuration."""

    provider: str
    options_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware: str


class Fingerprint(_Frozen):
    """Identifies the model, graph and runtime a policy was certified on."""

    model: str
    revision: str
    variant: str
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    opset: int
    onnxruntime: str
    configurations: tuple[Configuration, ...] = Field(min_length=1)


class PolicyDocument(_Frozen):
    format_version: int
    origin: Literal["release", "custom"]
    fingerprint: Fingerprint
    confidence: float = Field(gt=0, lt=1)
    gate_level: float = Field(gt=0, lt=1)
    label_taken: float
    decision_types: dict[ModelType, TypePolicy]

    @model_validator(mode="after")
    def _check_label_taken(self) -> Self:
        if self.label_taken != LABEL_TAKEN:
            message = f"label_taken is {self.label_taken}; this package uses {LABEL_TAKEN}"
            raise ValueError(message)
        return self


def risk_key(risk: float) -> str:
    """Return the document key for a risk level, e.g. `0.01 -> "0.01"`."""
    return f"{risk:g}"


def expected_arrays(document: PolicyDocument) -> set[str]:
    """Return the array names required by a document."""
    keys: set[str] = set()
    for name, policy in document.decision_types.items():
        keys |= {f"gate/{name}/{part}" for part in ("centroids", "precision", "reference")}
        if policy.conformal == "bucket":
            keys |= {f"conformal/{name}/{entry.bucket}" for entry in policy.scaling}
        elif policy.conformal == "type":
            keys.add(f"conformal/{name}")
    return keys


def _check_arrays(document: PolicyDocument, arrays: Mapping[str, Floats]) -> None:
    expected = expected_arrays(document)
    if set(arrays) != expected:
        missing, extra = sorted(expected - set(arrays)), sorted(set(arrays) - expected)
        message = f"policy arrays: missing {missing}, unexpected {extra}"
        raise PolicyError(message)
    for name, value in arrays.items():
        if value.dtype != np.float64 or not np.all(np.isfinite(value)):
            message = f"policy array {name} is {value.dtype} or holds a non-finite value"
            raise PolicyError(message)
    for name in document.decision_types:
        centroids = arrays[f"gate/{name}/centroids"]
        precision = arrays[f"gate/{name}/precision"]
        width = centroids.shape[-1]
        if centroids.ndim != MATRIX_RANK or precision.shape != (width, width):
            message = f"gate {name}: centroids {centroids.shape}, precision {precision.shape}"
            raise PolicyError(message)
    for name, value in arrays.items():
        is_sorted_scores = name.startswith("conformal/") or name.endswith("/reference")
        if is_sorted_scores and (value.ndim != 1 or np.any(np.diff(value) < 0)):
            message = f"policy array {name} of shape {value.shape} is not sorted and 1-D"
            raise PolicyError(message)


@dataclass(frozen=True, slots=True)
class Policy:
    """A policy document and its arrays, validated together."""

    document: PolicyDocument
    arrays: Mapping[str, Floats]

    def __post_init__(self) -> None:
        _check_arrays(self.document, self.arrays)

    def type_policy(self, model_type: ModelType) -> TypePolicy:
        found = self.document.decision_types.get(model_type)
        if found is None:
            message = f"the policy has no entry for {model_type} decisions"
            raise PolicyError(message)
        return found

    @property
    def risk_levels(self) -> tuple[float, ...]:
        """The risk levels with an entry for at least one decision type, ascending."""
        keys = {
            key for policy in self.document.decision_types.values() for key in policy.thresholds
        }
        return tuple(sorted(float(key) for key in keys))

    def conformal_scores(self, model_type: ModelType, bucket: int) -> Floats | None:
        """Return the sorted conformal scores for a type and bucket, or None if it has none."""
        policy = self.type_policy(model_type)
        if policy.conformal == "bucket":
            return self.arrays[f"conformal/{model_type}/{bucket}"]
        if policy.conformal == "type":
            return self.arrays[f"conformal/{model_type}"]
        return None


def parse_policy(document: bytes, arrays: Mapping[str, Floats], where: str) -> Policy:
    """Parse and validate a policy. Errors are prefixed with `where`."""
    try:
        raw = json.loads(document)
    except json.JSONDecodeError as error:
        message = f"{where}: not JSON: {error}"
        raise PolicyError(message) from error
    version = raw.get("format_version") if isinstance(raw, dict) else None
    if version != FORMAT_VERSION:
        message = f"{where}: policy format {version!r}; this package reads {FORMAT_VERSION}"
        raise FormatVersionError(message)
    try:
        parsed = PolicyDocument.model_validate(raw)
    except ValidationError as error:
        message = f"{where}: {error}"
        raise PolicyError(message) from error
    return Policy(parsed, arrays)


def read_arrays(path: Path) -> dict[str, Floats]:
    """Load all arrays from an npz file with pickling disabled."""
    with np.load(path, allow_pickle=False) as found:
        return {name: np.asarray(found[name]) for name in found.files}


def read_policy(document_path: Path, arrays_path: Path) -> Policy:
    return parse_policy(document_path.read_bytes(), read_arrays(arrays_path), str(document_path))


def write_policy(policy: Policy, document_path: Path, arrays_path: Path) -> None:
    """Write both files atomically (temporary file, then rename)."""
    document_path.parent.mkdir(parents=True, exist_ok=True)
    part = document_path.with_name(document_path.name + ".part")
    part.write_text(policy.document.model_dump_json(indent=2) + "\n", encoding="utf-8")
    part.replace(document_path)
    arrays_part = arrays_path.with_name(arrays_path.name + ".part")
    with zipfile.ZipFile(arrays_part, "w") as archive:
        for name, value in policy.arrays.items():
            with archive.open(f"{name}.npy", "w") as member:
                np.lib.format.write_array(member, value, allow_pickle=False)
    arrays_part.replace(arrays_path)
