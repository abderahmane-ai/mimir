"""Imports that depend on optional extras, raising `MissingExtraError` when absent."""

import importlib.util
from typing import TYPE_CHECKING, Final

from mimir.core.errors import MissingExtraError

if TYPE_CHECKING:
    from mimir.runtime.engine import Mimir

LOCAL_MODULES: Final = frozenset(
    {"huggingface_hub", "numpy", "onnx", "onnxruntime", "sigstore", "tokenizers"}
)
SERVER_MODULES: Final = frozenset({"fastapi", "prometheus_client", "uvicorn", "yaml"})
MCP_MODULES: Final = frozenset({"mcp", "uvicorn", "yaml"})


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


def require_extra(feature: str, extra: str, modules: frozenset[str]) -> None:
    """Raise `MissingExtraError` naming `feature` and `extra` if any of `modules` is missing."""
    for module in sorted(modules):
        if importlib.util.find_spec(module) is None:
            raise MissingExtraError(feature, extra, module)
