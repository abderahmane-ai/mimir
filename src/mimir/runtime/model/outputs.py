"""Read a decision out of the workspace and candidates: utilities, abstain, ordinal, histogram."""

from dataclasses import dataclass

import torch
from torch import nn

from mimir.runtime.model.blocks import Norm
from mimir.runtime.model.config import HeadConfig


@dataclass(frozen=True, slots=True)
class Readout:
    utilities: torch.Tensor
    abstain: torch.Tensor
    ordinal_score: torch.Tensor
    thresholds: torch.Tensor
    histogram_logits: torch.Tensor


def readout_mlp(inputs: int, width: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        Norm(inputs),
        nn.Linear(inputs, width, bias=False),
        nn.SiLU(),
        nn.Linear(width, outputs),
    )


class DecisionReadout(nn.Module):
    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        width = config.width
        self.utility = readout_mlp(width, width, 1)
        self.abstain = readout_mlp(width, width, 1)
        self.ordinal_score = readout_mlp(width, width, 1)
        self.threshold = readout_mlp(2 * width, width, 1)
        self.histogram = readout_mlp(width, width, config.histogram_bins)

    def forward(self, pooled: torch.Tensor, candidates: torch.Tensor) -> Readout:
        """`pooled` is the workspace averaged over its slots."""
        pairs = torch.cat([candidates[:, :-1], candidates[:, 1:]], dim=-1)
        raw = self.threshold(pairs).squeeze(-1)
        steps = nn.functional.softplus(raw[:, 1:])
        thresholds = torch.cat([raw[:, :1], raw[:, :1] + steps.cumsum(dim=1)], dim=1)
        return Readout(
            utilities=self.utility(candidates).squeeze(-1),
            abstain=self.abstain(pooled).squeeze(-1),
            ordinal_score=self.ordinal_score(pooled).squeeze(-1),
            thresholds=thresholds,
            histogram_logits=self.histogram(pooled),
        )
