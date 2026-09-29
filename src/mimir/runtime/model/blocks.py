"""Pre-norm residual attention and SwiGLU blocks with query-key normalisation.

Every residual branch ends in a zero-initialised projection, so a new block is the identity.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from mimir.runtime.model.config import HeadConfig


class Norm(nn.RMSNorm):
    """RMSNorm in float32, returned in the input's dtype.

    Under fp16 autocast a half input meets the float32 weight and misses the fused kernel
    (torch warns "Mismatch dtype between input and weight", Kaggle smoke 2026-09-24).
    """

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        normed: torch.Tensor = super().forward(states.float())
        return normed.to(states.dtype)


@dataclass(frozen=True, slots=True)
class Memory:
    """Keys (normalised) and values over [batch, heads, length, size], with the key mask."""

    key: torch.Tensor
    value: torch.Tensor
    mask: torch.Tensor

    def extend(self, other: "Memory") -> "Memory":
        return Memory(
            torch.cat([self.key, other.key], dim=2),
            torch.cat([self.value, other.value], dim=2),
            torch.cat([self.mask, other.mask], dim=1),
        )


def masked_attention(query: torch.Tensor, memory: Memory) -> torch.Tensor:
    """Fused attention; a row with no valid key reads zeros instead of NaN."""
    bias = torch.zeros(memory.mask.shape, dtype=query.dtype, device=query.device)
    bias = bias.masked_fill(~memory.mask, torch.finfo(query.dtype).min)[:, None, None, :]
    read: torch.Tensor = nn.functional.scaled_dot_product_attention(
        query, memory.key, memory.value.to(query.dtype), attn_mask=bias
    )
    return read * memory.mask.any(dim=1)[:, None, None, None]


def attention_weights(query: torch.Tensor, memory: Memory) -> torch.Tensor:
    """The same attention's weights in float32, zero on masked keys and on keyless rows."""
    logits = (query @ memory.key.transpose(-1, -2)).float() / math.sqrt(query.shape[-1])
    mask = memory.mask[:, None, None, :]
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return logits.softmax(dim=-1) * mask.any(dim=-1, keepdim=True)


class Attention(nn.Module):
    """Multi-head attention from queries onto a memory, with RMSNorm on queries and keys."""

    def __init__(self, config: HeadConfig, *, zero_output: bool) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_size = config.head_size
        self.query = nn.Linear(config.width, config.width, bias=False)
        self.key_value = nn.Linear(config.width, 2 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)
        self.query_norm = Norm(config.head_size)
        self.key_norm = Norm(config.head_size)
        if zero_output:
            nn.init.zeros_(self.output.weight)

    def _split(self, states: torch.Tensor) -> torch.Tensor:
        batch, length, _ = states.shape
        return states.view(batch, length, self.heads, self.head_size).transpose(1, 2)

    def _merge(self, read: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = read.shape
        width = self.heads * self.head_size
        merged: torch.Tensor = self.output(read.transpose(1, 2).reshape(batch, length, width))
        return merged

    def memory(self, states: torch.Tensor, mask: torch.Tensor) -> Memory:
        key, value = self.key_value(states).chunk(2, dim=-1)
        return Memory(self.key_norm(self._split(key)), self._split(value), mask)

    def queries(self, states: torch.Tensor) -> torch.Tensor:
        split: torch.Tensor = self.query_norm(self._split(self.query(states)))
        return split

    def read(self, states: torch.Tensor, memory: Memory) -> torch.Tensor:
        return self._merge(masked_attention(self.queries(states), memory))

    def read_with_weights(
        self, states: torch.Tensor, memory: Memory
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = attention_weights(self.queries(states), memory)
        return self._merge(weights.to(memory.value.dtype) @ memory.value), weights


class AttentionBlock(nn.Module):
    """states + Attention(norm(states), norm(memory)); self-attention when memory is states."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.query_norm = Norm(config.width)
        self.memory_norm = Norm(config.width)
        self.attention = Attention(config, zero_output=True)

    def memory(self, states: torch.Tensor, mask: torch.Tensor) -> Memory:
        return self.attention.memory(self.memory_norm(states), mask)

    def read(self, states: torch.Tensor, memory: Memory) -> torch.Tensor:
        return states + self.attention.read(self.query_norm(states), memory)

    def forward(
        self, states: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.read(states, self.memory(memory, memory_mask))


class FeedForwardBlock(nn.Module):
    """states + SwiGLU(norm(states))."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.norm = Norm(config.width)
        self.gate_up = nn.Linear(config.width, 2 * config.feedforward_hidden, bias=False)
        self.down = nn.Linear(config.feedforward_hidden, config.width, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(self.norm(states)).chunk(2, dim=-1)
        update: torch.Tensor = self.down(nn.functional.silu(gate) * up)
        return states + update
