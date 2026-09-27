"""HTTP request, response and error models shared by the client and the server."""

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from mimir.core.context import ContextInput
from mimir.core.decisions import DecisionSpec, ModelType
from mimir.core.results import DecisionResult

DEFAULT_RISK: Final = 0.01
MAX_BODY_BYTES: Final = 4 * 1024 * 1024
MAX_BATCH_ITEMS: Final = 64


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class DecideRequest(_Frozen):
    """Body of `POST /v1/decide`. `alpha=None` uses the release default."""

    context: ContextInput
    decision: DecisionSpec
    risk: float = DEFAULT_RISK
    alpha: float | None = Field(default=None, gt=0, lt=1)


class UncertifiedRequest(_Frozen):
    """Body of `POST /v1/decide/uncertified`."""

    context: ContextInput
    decision: DecisionSpec


class BatchItem(_Frozen):
    context: ContextInput
    decision: DecisionSpec


class BatchRequest(_Frozen):
    """Body of `POST /v1/decide/batch`. Results are returned in item order."""

    items: list[BatchItem] = Field(min_length=1)
    risk: float = DEFAULT_RISK
    alpha: float | None = Field(default=None, gt=0, lt=1)


class BatchResponse(_Frozen):
    results: list[DecisionResult]


class InputLimits(_Frozen):
    """Maximum input sizes the release was tested at. Larger inputs are rejected."""

    options: int
    levels: int
    context_tokens: int


class RuntimeInfo(_Frozen):
    """The ONNX Runtime configuration in use."""

    onnxruntime: str
    provider: str
    options_sha256: str
    hardware: str


class ModelInfo(_Frozen):
    """Response of `GET /v1/models` and `Decider.info()`.

    `certification` is one of:

    - `certified`: the policy's fingerprint lists this runtime and hardware;
    - `equivalent`: the hardware is not listed, but the equivalence check passed;
    - `none`: no policy is loaded; only `decide_uncertified` is available.
    """

    model: str
    revision: str
    variant: str
    device: Literal["cpu", "cuda"]
    runtime: RuntimeInfo
    certification: Literal["certified", "equivalent", "none"]
    policy: Literal["release", "custom"] | None
    risk_levels: tuple[float, ...]
    decision_types: tuple[ModelType, ...]
    limits: InputLimits


class ErrorDetail(_Frozen):
    type: str
    message: str


class ErrorBody(_Frozen):
    """Body of every error response."""

    error: ErrorDetail
