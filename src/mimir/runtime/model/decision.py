"""The decision model: the encoder over chunks and candidates, typed values injected, the head.

Copied from the training repository; held to it by the golden fixtures. Only the inference
path is carried: one padded encoder call per row kind, the exit depth's readout, the evidence
and the final workspace.
"""

import torch
from torch import nn
from transformers import ModernBertModel

from mimir.runtime.encoder.attention import padded_dense_masks
from mimir.runtime.encoder.forward import Injection, encode
from mimir.runtime.model.batch import DecisionBatch
from mimir.runtime.model.head import DecisionHead, DepthOutput, HeadInputs


class ModelError(ValueError):
    """An encoder and a head that do not fit together."""


def _mean_question(
    question: torch.Tensor, batch: DecisionBatch
) -> tuple[torch.Tensor, torch.Tensor]:
    """Question positions 1.. of every chunk, averaged over the chunks of each record.

    `question` is each chunk's states from position 1, at least as wide as its longest
    question; wider columns are masked, so they change no state the head reads. Chunks are
    summed into their record by a one-hot product, not a scatter-add.
    """
    positions = torch.arange(question.shape[1], device=question.device)
    per_chunk = (positions < batch.question_length[:, None]).to(question.dtype)
    records = torch.arange(batch.records, device=question.device)
    owner = (batch.chunk_record[None, :] == records[:, None]).to(question.dtype)
    summed = torch.einsum("rc,cqd->rqd", owner, question * per_chunk[..., None])
    counts = owner @ per_chunk
    return summed / counts.clamp(min=1)[..., None], counts > 0


def _pool_candidates(hidden: torch.Tensor, batch: DecisionBatch) -> torch.Tensor:
    """Each candidate row's mean state over its own text tokens."""
    weight = batch.candidate_text_mask[..., None].to(hidden.dtype)
    return (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)


class DecisionModel(nn.Module):
    def __init__(self, encoder: ModernBertModel, head: DecisionHead) -> None:
        super().__init__()
        config = head.config
        if config.encoder_width != encoder.config.hidden_size:
            message = (
                f"head reads width {config.encoder_width}, encoder has {encoder.config.hidden_size}"
            )
            raise ModelError(message)
        if config.encoder_layers != len(encoder.layers):
            message = (
                f"head expects {config.encoder_layers} layers, encoder has {len(encoder.layers)}"
            )
            raise ModelError(message)
        self.encoder = encoder
        self.head = head
        self.window = int(encoder.config.sliding_window)

    def _encode(
        self, input_ids: torch.Tensor, mask: torch.Tensor, injection: Injection | None
    ) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.expand(input_ids.shape[0], -1)
        masks = padded_dense_masks(mask.long(), self.window)
        return encode(self.encoder, input_ids, positions, masks, injection)

    def _injection(self, batch: DecisionBatch) -> Injection | None:
        if batch.typed_chunk.numel() == 0:
            return None
        values = self.head.typed_values(batch.typed)
        hidden = values.new_zeros((*batch.chunk_ids.shape, values.shape[-1]))
        hidden = hidden.index_put(
            (batch.typed_chunk, batch.typed_position), values, accumulate=True
        )
        return Injection(self.head.config.injection_layer, hidden)

    def forward_padded(self, batch: DecisionBatch, question_cap: int) -> list[DepthOutput]:
        """One padded encoder call per row kind and no shape read from values.

        The question states are `question_cap` wide whatever the chunk width, so no slice
        bound depends on it. The batch must hold at least one candidate row.
        """
        hidden = self._encode(batch.chunk_ids, batch.chunk_mask, self._injection(batch))
        question = nn.functional.pad(hidden, (0, 0, 0, question_cap + 1))[:, 1 : 1 + question_cap]
        candidates = self._encode(batch.candidate_ids, batch.candidate_mask, None)
        return self._decide(batch, hidden, question, _pool_candidates(candidates, batch))

    def _decide(
        self,
        batch: DecisionBatch,
        hidden: torch.Tensor,
        question: torch.Tensor,
        pooled: torch.Tensor,
    ) -> list[DepthOutput]:
        question, question_mask = _mean_question(question, batch)
        tokens = hidden[batch.key_chunk, batch.key_position]
        if pooled.shape[0]:
            candidates = pooled[batch.candidate_index]
        else:
            candidates = pooled.new_zeros((*batch.candidate_index.shape, pooled.shape[-1]))
        inputs = HeadInputs(
            tokens=tokens,
            token_mask=batch.key_mask,
            question=question,
            question_mask=question_mask,
            candidates=candidates,
            candidate_mask=batch.candidate_present,
            decision_type=batch.targets.decision_type,
        )
        outputs: list[DepthOutput] = self.head(inputs)
        return outputs
