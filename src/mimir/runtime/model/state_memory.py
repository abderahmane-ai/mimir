"""Compress a record's chunk tokens into question-conditioned latents Z (DESIGN §2)."""

import torch
from torch import nn

from mimir.runtime.model.blocks import AttentionBlock, FeedForwardBlock
from mimir.runtime.model.config import HeadConfig


class StateMemory(nn.Module):
    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.empty(config.latents, config.width))
        nn.init.orthogonal_(self.latents)
        self.question = nn.Linear(config.width, config.width, bias=False)
        self.read = AttentionBlock(config)
        self.read_feedforward = FeedForwardBlock(config)
        self.mix = AttentionBlock(config)
        self.mix_feedforward = FeedForwardBlock(config)

    def forward(
        self, question: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        latents: torch.Tensor = self.latents + self.question(question)[:, None]
        latents = self.read_feedforward(self.read(latents, tokens, token_mask))
        every = torch.ones(latents.shape[:2], dtype=torch.bool, device=latents.device)
        memory: torch.Tensor = self.mix_feedforward(self.mix(latents, latents, every))
        return memory
