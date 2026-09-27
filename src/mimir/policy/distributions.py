"""Answer distributions computed from graph outputs.

Scaling parameters and distribution per model decision type:

| Type | Scaling | Distribution |
|---|---|---|
| binary, categorical | `[T]` | softmax over options and abstain (last entry) |
| ranking | `[T]` | softmax over candidates (Plackett-Luce top-1 marginal) |
| ordinal | `[T]` | `P(y <= k) = sigmoid((theta_k - s) / T)` |
| multilabel | `[a, b]` | `sigmoid(a * (u_i - u_abstain) + b)` per option |
| continuous | `[T]` | softmax over histogram bins spanning `[low, high]` |

The identity scaling (`[1]`, or `[1, 0]` for multilabel) gives the raw model distribution.
All arithmetic is float64, matching how the policy was fitted.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from mimir.core.decisions import ModelType

Floats = npt.NDArray[np.float64]

CHOICE_TYPES: Final[frozenset[ModelType]] = frozenset({"binary", "categorical"})
# A multilabel option is chosen when its probability is above this.
LABEL_TAKEN: Final = 0.5


@dataclass(frozen=True, slots=True)
class Readout:
    """Graph outputs for one request with K options.

    `thresholds` are the K - 1 ordinal cut points, `histogram` the continuous bin logits and
    `workspace` the pooled representation used by the distance gate.
    """

    model_type: ModelType
    utilities: Floats
    thresholds: Floats
    abstain: float
    ordinal_score: float
    histogram: Floats
    workspace: Floats

    @property
    def count(self) -> int:
        return int(self.utilities.shape[0])


@dataclass(frozen=True, slots=True)
class Decision:
    """Chosen option indices and the decision's score.

    `chosen` is empty for abstain or an empty multilabel set. `score` is None for continuous.
    """

    chosen: tuple[int, ...]
    score: float | None


def identity(model_type: ModelType) -> tuple[float, ...]:
    """Return the scaling that leaves the distribution unchanged."""
    return (1.0, 0.0) if model_type == "multilabel" else (1.0,)


def softmax(logits: Floats) -> Floats:
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    normalised: Floats = weights / weights.sum()
    return normalised


def sigmoid(values: Floats) -> Floats:
    found: Floats = np.exp(-np.logaddexp(0.0, -values))
    return found


def probabilities(readout: Readout, scaling: Sequence[float]) -> Floats:
    """Return the scaled distribution.

    Length is K + 1 for binary and categorical (abstain last), the bin count for continuous,
    and K otherwise.
    """
    kind = readout.model_type
    temperature = float(scaling[0])
    if kind in CHOICE_TYPES:
        logits = np.concatenate([readout.utilities, [readout.abstain]]) / temperature
        return softmax(logits)
    if kind == "ranking":
        return softmax(readout.utilities / temperature)
    if kind == "ordinal":
        below = sigmoid((readout.thresholds - readout.ordinal_score) / temperature)
        cumulative = np.concatenate([below, [1.0]])
        found: Floats = np.diff(cumulative, prepend=0.0)
        return found
    if kind == "multilabel":
        margin = readout.utilities - readout.abstain
        return sigmoid(temperature * margin + float(scaling[1]))
    return softmax(readout.histogram / temperature)


def decide(model_type: ModelType, probs: Floats) -> Decision:
    """Return the decision for a distribution.

    Multilabel chooses every option above `LABEL_TAKEN`; its score is the product of
    `max(p, 1 - p)` over options. Other types choose the argmax, scored by its probability.
    """
    if model_type == "multilabel":
        chosen = tuple(int(index) for index in np.flatnonzero(probs > LABEL_TAKEN))
        certainty = np.maximum(probs, 1 - probs)
        return Decision(chosen, float(np.prod(certainty)))
    top = int(probs.argmax())
    if model_type == "continuous":
        return Decision((top,), None)
    if model_type in CHOICE_TYPES and top == probs.shape[0] - 1:
        return Decision((), float(probs[top]))
    return Decision((top,), float(probs[top]))


def option_count_bucket(count: int) -> int:
    """Return `ceil(log2(K))`, or 0 for K <= 1."""
    return max(count - 1, 0).bit_length()
