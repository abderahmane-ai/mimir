"""Models for a release's `config.json` and `manifest.json`.

`config.json` lists the variants, decision types (in graph index order), certified risk levels,
input limits, default alpha, input layout and graph contract. `manifest.json` lists the SHA-256
of every other file and the `mimir-decisions` versions that can load the release.
"""

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from mimir.core.decisions import MODEL_TYPES, ModelType

CONFIG_FILE: Final = "config.json"
MANIFEST_FILE: Final = "manifest.json"
SIGNATURE_FILE: Final = "manifest.json.sigstore"
FORMAT_VERSION: Final = 1
PACKAGE: Final = "mimir-decisions"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Variant(_Frozen):
    """A graph and its policy. `policy` is the path without the `.json`/`.npz` suffix."""

    graph: str
    policy: str
    devices: tuple[str, ...]


class Limits(_Frozen):
    options: int = Field(gt=0)
    levels: int = Field(gt=0)
    context_tokens: int = Field(gt=0)


class SpecialTokens(_Frozen):
    cls: int
    sep: int
    pad: int
    mask: int
    newline: int


class Layout(_Frozen):
    """Chunking parameters used to build graph inputs."""

    chunk_tokens: int = Field(gt=0)
    question_cap: int = Field(gt=0)
    crossing_tokens: int = Field(gt=0)
    special_tokens: SpecialTokens


class TensorSpec(_Frozen):
    name: str
    dtype: str
    rank: int = Field(ge=0)


class GraphContract(_Frozen):
    """Allowed opsets and operators (`domain::type`), and the expected inputs and outputs."""

    opset: dict[str, int]
    operators: tuple[str, ...]
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]


class ReleaseConfig(_Frozen):
    format_version: int
    variants: dict[str, Variant] = Field(min_length=1)
    tokenizer: str
    decision_types: tuple[ModelType, ...]
    risk_levels: tuple[float, ...] = Field(min_length=1)
    default_alpha: float = Field(gt=0, lt=1)
    limits: Limits
    layout: Layout
    graph: GraphContract

    def type_index(self, model_type: ModelType) -> int:
        """Return the graph's integer index for a decision type."""
        return self.decision_types.index(model_type)


class Manifest(_Frozen):
    format_version: int
    files: dict[str, str]
    loadable_by: dict[str, str]


def is_known_type_order(config: ReleaseConfig) -> bool:
    """Return whether the release uses exactly the decision types this package supports."""
    return sorted(config.decision_types) == sorted(MODEL_TYPES)
