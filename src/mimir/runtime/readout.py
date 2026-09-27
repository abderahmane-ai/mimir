"""Splitting batched graph outputs into per-request readouts and relevance scores."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from mimir.core.decisions import ModelType
from mimir.policy.distributions import Readout
from mimir.runtime.layout import Batch

OUTPUTS: Final = (
    "utilities",
    "thresholds",
    "abstain",
    "ordinal_score",
    "histogram_logits",
    "evidence",
    "workspace",
)

Array = npt.NDArray[np.generic]
Floats = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """Graph outputs for one request. `relevance[s]` is the evidence mass on state segment s."""

    readout: Readout
    relevance: Floats


def as_float64(outputs: Mapping[str, Array]) -> dict[str, Floats]:
    """Return the named graph outputs as float64 arrays."""
    return {name: np.asarray(outputs[name], dtype=np.float64) for name in OUTPUTS}


def readout_at(outputs: Mapping[str, Floats], row: int, kind: ModelType, count: int) -> Readout:
    """Return the readout of batch row `row`, which has `count` options."""
    return Readout(
        model_type=kind,
        utilities=outputs["utilities"][row, :count],
        thresholds=outputs["thresholds"][row, : max(count - 1, 0)],
        abstain=float(outputs["abstain"][row]),
        ordinal_score=float(outputs["ordinal_score"][row]),
        histogram=outputs["histogram_logits"][row],
        workspace=outputs["workspace"][row],
    )


def split_outputs(
    outputs: Mapping[str, Array],
    batch: Batch,
    model_types: Sequence[ModelType],
    segment_counts: Sequence[int],
) -> list[RequestOutput]:
    """Return one `RequestOutput` per request in the batch, in order."""
    values = as_float64(outputs)
    live = (batch.key_segments >= 0) & batch.feed["key_mask"].astype(np.bool_)
    found: list[RequestOutput] = []
    for row, (kind, count, segments) in enumerate(
        zip(model_types, batch.counts, segment_counts, strict=True)
    ):
        relevance = np.zeros(segments, dtype=np.float64)
        np.add.at(relevance, batch.key_segments[row][live[row]], values["evidence"][row][live[row]])
        found.append(RequestOutput(readout_at(values, row, kind, count), relevance))
    return found
