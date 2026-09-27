"""JSON Schemas of all public models, as printed by `mimir schema`."""

from typing import Final

from pydantic import BaseModel, JsonValue, TypeAdapter

from mimir.core.context import Context, ContextInput
from mimir.core.decisions import SPEC_TYPES, DecisionSpec
from mimir.core.labels import LabelledDecision
from mimir.core.results import RESULT_FOR_SPEC, DecisionResult
from mimir.core.tools import ToolArguments
from mimir.core.wire import (
    BatchRequest,
    BatchResponse,
    DecideRequest,
    ErrorBody,
    ModelInfo,
    UncertifiedRequest,
)

MODELS: Final[tuple[type[BaseModel], ...]] = (
    Context,
    *SPEC_TYPES,
    *RESULT_FOR_SPEC.values(),
    DecideRequest,
    UncertifiedRequest,
    BatchRequest,
    BatchResponse,
    ModelInfo,
    ErrorBody,
    ToolArguments,
    LabelledDecision,
)


def json_schemas() -> dict[str, JsonValue]:
    """Return schemas keyed by model name, including the `ContextInput`, `DecisionSpec` and
    `DecisionResult` unions. Results use serialization mode; everything else validation mode.
    """
    schemas: dict[str, JsonValue] = {}
    results = set(RESULT_FOR_SPEC.values())
    for model in MODELS:
        schemas[model.__name__] = model.model_json_schema(
            mode="serialization" if model in results else "validation"
        )
    schemas["ContextInput"] = TypeAdapter[ContextInput](ContextInput).json_schema()
    schemas["DecisionSpec"] = TypeAdapter[DecisionSpec](DecisionSpec).json_schema()
    schemas["DecisionResult"] = TypeAdapter[DecisionResult](DecisionResult).json_schema(
        mode="serialization"
    )
    return schemas
