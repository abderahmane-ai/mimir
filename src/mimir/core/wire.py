"""HTTP request, response and error models shared by the client and the server."""

from enum import StrEnum
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mimir.core.context import ContextInput
from mimir.core.decisions import DecisionSpec, ModelType
from mimir.core.results import DecisionResult

DEFAULT_RISK: Final = 0.05
MAX_BODY_BYTES: Final = 4 * 1024 * 1024
MAX_BATCH_ITEMS: Final = 64


class Mode(StrEnum):
    STANDARD = "standard"
    THRESHOLD = "threshold"
    CERTIFIED = "certified"


def check_mode(mode: Mode, min_confidence: float | None) -> None:
    """Raise if `mode` and `min_confidence` do not combine: `threshold` needs a floor in
    (0, 1), and no other mode takes one."""
    if mode == Mode.THRESHOLD:
        if min_confidence is None or not 0 < min_confidence < 1:
            message = f"threshold mode needs min_confidence in (0, 1); got {min_confidence}"
            raise ValueError(message)
    elif min_confidence is not None:
        message = f"min_confidence applies only to threshold mode, not {mode.value}"
        raise ValueError(message)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class _Decision(_Frozen):
    mode: Mode = Mode.STANDARD
    min_confidence: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        check_mode(self.mode, self.min_confidence)
        return self


class DecideRequest(_Decision):
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


class BatchRequest(_Decision):
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
    """The Torch runtime in use."""

    torch: str
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
