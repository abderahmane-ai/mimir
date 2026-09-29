"""Candidates as a set: each reads the question, then all attend to all, with no positions."""

import torch
from torch import nn

from mimir.runtime.model.blocks import AttentionBlock, FeedForwardBlock
from mimir.runtime.model.config import HeadConfig


class CandidateProcessor(nn.Module):
    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.read_question = AttentionBlock(config)
        self.interact = AttentionBlock(config)
        self.feedforward = FeedForwardBlock(config)

    def forward(
        self,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        question: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> torch.Tensor:
        candidates = self.read_question(candidates, question, question_mask)
        candidates = self.interact(candidates, candidates, candidate_mask)
        refined: torch.Tensor = self.feedforward(candidates)
        return refined
