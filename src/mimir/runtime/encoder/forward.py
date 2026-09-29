"""Run the encoder's layers with one prepared mask per attention kind.

Copied from the training repository; held to it by the golden fixtures.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from transformers import ModernBertModel

Masks = Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class Injection:
    """Hidden-state additions entering the encoder at the input of one layer."""

    layer: int
    hidden: torch.Tensor


def encode(
    model: ModernBertModel,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    masks: Masks,
    injection: Injection | None = None,
) -> torch.Tensor:
    layer_types = model.config.layer_types
    if layer_types is None:
        message = "encoder config names no layer types"
        raise ValueError(message)
    if injection is not None and not 0 <= injection.layer < len(model.layers):
        message = f"injection at layer {injection.layer} of {len(model.layers)}"
        raise ValueError(message)
    hidden: torch.Tensor = model.embeddings(input_ids=input_ids)
    rotary = {kind: model.rotary_emb(hidden, position_ids, kind) for kind in set(layer_types)}
    for index, (layer, kind) in enumerate(zip(model.layers, layer_types, strict=True)):
        if injection is not None and index == injection.layer:
            hidden = hidden + injection.hidden.to(hidden.dtype)
        hidden = layer(hidden, attention_mask=masks[kind], position_embeddings=rotary[kind])
    output: torch.Tensor = model.final_norm(hidden)
    return output
