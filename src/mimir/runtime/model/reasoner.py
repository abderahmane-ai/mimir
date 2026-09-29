"""The shared reasoning cell over workspace slots, applied once per iteration.

What never changes across iterations (Z, the question, the chunk tokens) is projected to keys
and values once per record; each iteration projects only the workspace and the evidence.

Copied from the training repository; held to it by the golden fixtures.
"""

from dataclasses import dataclass

import torch
from torch import nn

from mimir.core.decisions import MODEL_TYPES
from mimir.runtime.model.blocks import Attention, AttentionBlock, FeedForwardBlock, Memory, Norm
from mimir.runtime.model.config import HeadConfig

RELATIONS = ("supports", "contradicts", "irrelevant")


@dataclass(frozen=True, slots=True)
class Context:
    """The fixed memories every iteration reads, already projected."""

    recall: Memory
    tokens: Memory
    candidate_state: Memory
    candidate_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class Iteration:
    workspace: torch.Tensor
    candidates: torch.Tensor
    evidence: torch.Tensor
    gap_weights: torch.Tensor


def _every(states: torch.Tensor) -> torch.Tensor:
    return torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)


class ReasonerCell(nn.Module):
    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.recall = AttentionBlock(config)
        self.slot_norm = Norm(config.width)
        self.token_norm = Norm(config.width)
        self.evidence_norm = Norm(config.width)
        self.gap = Attention(config, zero_output=False)
        self.relation = nn.Linear(3 * config.width, len(RELATIONS), bias=False)
        nn.init.zeros_(self.relation.weight)
        self.support = nn.Linear(config.width, config.width, bias=False)
        self.contradict = nn.Linear(config.width, config.width, bias=False)
        nn.init.zeros_(self.support.weight)
        nn.init.zeros_(self.contradict.weight)
        self.workspace_feedforward = FeedForwardBlock(config)
        self.read_candidates = AttentionBlock(config)
        self.candidates_read = AttentionBlock(config)
        self.interact = AttentionBlock(config)
        self.candidate_feedforward = FeedForwardBlock(config)

    def context(
        self,
        memory: torch.Tensor,
        question: torch.Tensor,
        question_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Context:
        recall = torch.cat([memory, question], dim=1)
        state = torch.cat([memory, tokens], dim=1)
        return Context(
            recall=self.recall.memory(recall, torch.cat([_every(memory), question_mask], dim=1)),
            tokens=self.gap.memory(self.token_norm(tokens), token_mask),
            candidate_state=self.candidates_read.memory(
                state, torch.cat([_every(memory), token_mask], dim=1)
            ),
            candidate_mask=candidate_mask,
        )

    def forward(
        self,
        workspace: torch.Tensor,
        candidates: torch.Tensor,
        evidence: torch.Tensor,
        context: Context,
    ) -> Iteration:
        own = torch.cat([workspace, evidence], dim=1)
        workspace = self.recall.read(
            workspace, self.recall.memory(own, _every(own)).extend(context.recall)
        )
        query = self.slot_norm(workspace)
        found, gap_weights = self.gap.read_with_weights(query, context.tokens)
        found_norm = self.evidence_norm(found)
        relation = self.relation(torch.cat([query, found_norm, query * found_norm], dim=-1))
        weights = relation.float().softmax(dim=-1).to(workspace.dtype)
        workspace = (
            workspace
            + weights[..., 0:1] * self.support(found_norm)
            + weights[..., 1:2] * self.contradict(found_norm)
        )
        evidence = torch.cat([evidence, found], dim=1)
        workspace = self.workspace_feedforward(workspace)
        workspace = self.read_candidates(workspace, candidates, context.candidate_mask)
        own = torch.cat([workspace, evidence], dim=1)
        candidates = self.candidates_read.read(
            candidates,
            self.candidates_read.memory(own, _every(own)).extend(context.candidate_state),
        )
        candidates = self.interact(candidates, candidates, context.candidate_mask)
        candidates = self.candidate_feedforward(candidates)
        return Iteration(workspace, candidates, evidence, gap_weights)


class Reasoner(nn.Module):
    """Workspace slots seeded by the schema and the question, then the cell once per iteration."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.iterations = config.iterations
        self.slots = nn.Parameter(torch.empty(config.slots, config.width))
        nn.init.orthogonal_(self.slots)
        self.schemas = nn.Embedding(len(MODEL_TYPES), config.width)
        self.question = nn.Linear(config.width, config.width, bias=False)
        self.cell = ReasonerCell(config)

    def forward(
        self,
        question_mean: torch.Tensor,
        decision_type: torch.Tensor,
        candidates: torch.Tensor,
        context: Context,
    ) -> list[Iteration]:
        seed = self.schemas(decision_type) + self.question(question_mean)
        workspace = self.slots + seed[:, None]
        evidence = workspace.new_zeros((workspace.shape[0], 0, workspace.shape[2]))
        iterations: list[Iteration] = []
        for _ in range(self.iterations):
            step = self.cell(workspace, candidates, evidence, context)
            iterations.append(step)
            workspace, candidates, evidence = step.workspace, step.candidates, step.evidence
        return iterations
