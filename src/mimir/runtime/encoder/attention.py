"""Attention masks for padded batches: padding keys hidden, local layers within their window."""

import torch

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


def padded_dense_masks(
    attention_mask: torch.Tensor, sliding_window: int
) -> dict[str, torch.Tensor]:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    keys = attention_mask.bool()[:, None, None, :]
    near = (positions[:, None] - positions[None, :]).abs() <= sliding_window
    return {FULL_ATTENTION: keys, SLIDING_ATTENTION: keys & near}
