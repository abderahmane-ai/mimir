"""Imports that depend on optional extras, raising `MissingExtraError` when absent."""

from typing import TYPE_CHECKING, Final

from mimir.core.errors import MissingExtraError

if TYPE_CHECKING:
    from mimir.runtime.engine import Mimir

LOCAL_MODULES: Final = frozenset(
    {"huggingface_hub", "numpy", "onnx", "onnxruntime", "sigstore", "tokenizers"}
)


def engine_class(feature: str) -> "type[Mimir]":
    """Import `Mimir`, or raise `MissingExtraError` naming `feature` and the `local` extra."""
    try:
        from mimir.runtime.engine import Mimir
    except ModuleNotFoundError as error:
        root = (error.name or "").split(".")[0]
        if root not in LOCAL_MODULES:
            raise
        raise MissingExtraError(feature, "local", root) from error
    return Mimir
