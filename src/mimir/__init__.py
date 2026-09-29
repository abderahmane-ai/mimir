"""Typed, calibrated and certified decisions with MIMIR.

`Mimir` (local engine) and `MimirClient` (HTTP client) are imported lazily, so importing the
data models does not load ONNX Runtime.
"""

import os
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
from mimir.core.wire import Mode

# The Hub cache's shared blob store splits a graph and its external data across shard
# directories, and ONNX Runtime refuses external data outside the model's resolved
# directory; the classic per-repo cache keeps the pair in one. Read at huggingface_hub
# import time, so it is set before any submodule imports it.
os.environ.setdefault("HF_HUB_DISABLE_SHARED_BLOBS", "1")

if TYPE_CHECKING:
    from mimir.client.http import MimirClient
    from mimir.runtime.engine import Mimir

__version__ = metadata.version("mimir-decisions")

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
    "Estimate",
    "EstimateResult",
    "Field",
    "JsonState",
    "Mimir",
    "MimirClient",
    "MimirError",
    "Mode",
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
