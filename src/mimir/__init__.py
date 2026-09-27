"""Typed, calibrated and certified decisions with MIMIR.

`Mimir` (local engine) and `MimirClient` (HTTP client) are imported lazily, so importing the
data models does not load ONNX Runtime.
"""

from importlib import metadata
from typing import TYPE_CHECKING

from mimir.core.checks import CheckOutcome, Permission, ToolCallCheck
from mimir.core.context import Cell, Context, Field, JsonState, Passage, Table
from mimir.core.decider import Decider
from mimir.core.decisions import Choice, Estimate, MultiChoice, Rank, Rate, Verify, YesNo
from mimir.core.errors import MimirError
from mimir.core.results import (
    Certificate,
    ChoiceResult,
    ContextRelevance,
    Deferral,
    EstimateResult,
    MultiChoiceResult,
    OptionSet,
    RankResult,
    RateResult,
    Status,
    VerifyResult,
    YesNoResult,
)
from mimir.core.tools import DecisionTool

if TYPE_CHECKING:
    from mimir.client.http import MimirClient
    from mimir.runtime.engine import Mimir

__version__ = metadata.version("mimirai")

__all__ = [
    "Cell",
    "Certificate",
    "CheckOutcome",
    "Choice",
    "ChoiceResult",
    "Context",
    "ContextRelevance",
    "Decider",
    "DecisionTool",
    "Deferral",
    "Estimate",
    "EstimateResult",
    "Field",
    "JsonState",
    "Mimir",
    "MimirClient",
    "MimirError",
    "MultiChoice",
    "MultiChoiceResult",
    "OptionSet",
    "Passage",
    "Permission",
    "Rank",
    "RankResult",
    "Rate",
    "RateResult",
    "Status",
    "Table",
    "ToolCallCheck",
    "Verify",
    "VerifyResult",
    "YesNo",
    "YesNoResult",
    "__version__",
]


def __getattr__(name: str) -> object:
    if name == "Mimir":
        from mimir.extras import engine_class

        return engine_class("Mimir")
    if name == "MimirClient":
        from mimir.client.http import MimirClient

        return MimirClient
    message = f"module 'mimir' has no attribute {name!r}"
    raise AttributeError(message)
