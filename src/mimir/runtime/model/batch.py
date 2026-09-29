"""The decision model's input batch, copied from the training repository.

Held to it by the golden fixtures.
"""

from dataclasses import dataclass

import torch

from mimir.runtime.model.typed_values import TypedValues


@dataclass(frozen=True, slots=True)
class DecisionTargets:
    decision_type: torch.Tensor
    choice: torch.Tensor
    labels: torch.Tensor
    grades: torch.Tensor
    value: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor


@dataclass(frozen=True, slots=True)
class DecisionBatch:
    chunk_ids: torch.Tensor
    chunk_mask: torch.Tensor
    chunk_record: torch.Tensor
    question_length: torch.Tensor
    key_chunk: torch.Tensor
    key_position: torch.Tensor
    key_mask: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_text_mask: torch.Tensor
    candidate_index: torch.Tensor
    candidate_present: torch.Tensor
    typed: TypedValues
    typed_chunk: torch.Tensor
    typed_position: torch.Tensor
    targets: DecisionTargets

    @property
    def records(self) -> int:
        return self.candidate_present.shape[0]
