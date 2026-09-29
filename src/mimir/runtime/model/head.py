"""The decision head: projection, state memory, candidate set, reasoner and readout per depth.

Copied from the training repository; held to it by the golden fixtures. The quantile edges
arrive as tensors: they are buffers of the shipped weights.
"""

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from mimir.runtime.model.candidates import CandidateProcessor
from mimir.runtime.model.config import HeadConfig
from mimir.runtime.model.outputs import DecisionReadout, Readout
from mimir.runtime.model.reasoner import Reasoner
from mimir.runtime.model.state_memory import StateMemory
from mimir.runtime.model.typed_values import TypedValueEncoder

STATE_ROLE: Final = 0
QUESTION_ROLE: Final = 1
CANDIDATE_ROLE: Final = 2


@dataclass(frozen=True, slots=True)
class HeadInputs:
    """Encoder states per record: state keys, question tokens and pooled candidates."""

    tokens: torch.Tensor
    token_mask: torch.Tensor
    question: torch.Tensor
    question_mask: torch.Tensor
    candidates: torch.Tensor
    candidate_mask: torch.Tensor
    decision_type: torch.Tensor


@dataclass(frozen=True, slots=True)
class DepthOutput:
    """One depth's decision; `workspace` is the slot-averaged workspace the readout reads."""

    readout: Readout
    evidence_scores: torch.Tensor
    workspace: torch.Tensor


class DecisionHead(nn.Module):
    def __init__(
        self,
        config: HeadConfig,
        number_edges: torch.Tensor,
        day_edges: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.encoder_width, config.width, bias=False)
        self.roles = nn.Embedding(3, config.width)
        self.typed_values = TypedValueEncoder(config, number_edges, day_edges)
        self.state_memory = StateMemory(config)
        self.candidate_processor = CandidateProcessor(config)
        self.reasoner = Reasoner(config)
        self.readout = DecisionReadout(config)

    def _project(self, states: torch.Tensor, role: int) -> torch.Tensor:
        projected: torch.Tensor = self.projection(states)
        return projected + self.roles.weight[role]

    def forward(self, inputs: HeadInputs) -> list[DepthOutput]:
        tokens = self._project(inputs.tokens, STATE_ROLE)
        question = self._project(inputs.question, QUESTION_ROLE)
        candidates = self._project(inputs.candidates, CANDIDATE_ROLE)
        weight = inputs.question_mask[..., None].to(question.dtype)
        question_mean = (question * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)
        memory = self.state_memory(question_mean, tokens, inputs.token_mask)
        candidates = self.candidate_processor(
            candidates, inputs.candidate_mask, question, inputs.question_mask
        )
        context = self.reasoner.cell.context(
            memory,
            question,
            inputs.question_mask,
            tokens,
            inputs.token_mask,
            inputs.candidate_mask,
        )
        iterations = self.reasoner(question_mean, inputs.decision_type, candidates, context)
        outputs: list[DepthOutput] = []
        for step in iterations:
            pooled = step.workspace.mean(dim=1)
            outputs.append(
                DepthOutput(
                    readout=self.readout(pooled, step.candidates),
                    evidence_scores=step.gap_weights.mean(dim=(1, 2)),
                    workspace=pooled,
                )
            )
        return outputs
