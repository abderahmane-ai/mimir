"""Splitting tokenized state into encoder chunks and building candidate rows.

Each chunk is `[CLS] question [SEP] piece \\n piece ... [SEP]`. A table row is placed in the
same chunk as its table header whenever both fit. Candidates are encoded after the first chunk
when the whole request fits `crossing_tokens`, and after `[CLS] question [SEP]` otherwise.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mimir.core.errors import InputLimitError
from mimir.runtime.release import SpecialTokens
from mimir.runtime.rendering import TABLE_HEAD, TABLE_ROW

Tokens = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Piece:
    """Tokens `[token_start, token_end)` of state segment `segment`."""

    segment: int
    token_start: int
    token_end: int


@dataclass(frozen=True, slots=True)
class ChunkLayout:
    """One chunk's tokens and the offset of each piece within them."""

    tokens: Tokens
    question_tokens: int
    pieces: tuple[tuple[Piece, int], ...]


class _Planner:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.chunks: list[tuple[Piece, ...]] = []
        self.current: list[Piece] = []
        self.used = 0

    def close(self) -> None:
        if self.current:
            self.chunks.append(tuple(self.current))
        self.current = []
        self.used = 0

    def cost(self, length: int) -> int:
        return length + (1 if self.current else 0)

    def place(self, piece: Piece) -> None:
        length = piece.token_end - piece.token_start
        if self.used + self.cost(length) > self.budget:
            self.close()
        self.used += self.cost(length)
        self.current.append(piece)

    def holds(self, segment: int) -> bool:
        return any(piece.segment == segment for piece in self.current)


def body_budget(chunk_tokens: int, question_tokens: int) -> int:
    """Return the state tokens that fit in a chunk after the question and three specials."""
    budget = chunk_tokens - 3 - question_tokens
    if budget < 1:
        raise InputLimitError(
            limit="question tokens", value=question_tokens, maximum=chunk_tokens - 4
        )
    return budget


def plan_chunks(
    lengths: Sequence[int], kinds: Sequence[int], groups: Sequence[int], budget: int
) -> list[tuple[Piece, ...]]:
    """Assign state segments to chunks of at most `budget` tokens, in order.

    Segments longer than the budget are split. A table row is preceded by its header unless
    the header is already in the current chunk.
    """
    heads = {groups[index]: index for index, kind in enumerate(kinds) if kind == TABLE_HEAD}
    planner = _Planner(budget)
    for index, length in enumerate(lengths):
        if length > budget:
            planner.close()
            for start in range(0, length, budget):
                planner.place(Piece(index, start, min(length, start + budget)))
            continue
        head = heads.get(groups[index]) if kinds[index] == TABLE_ROW else None
        if head is not None and lengths[head] + 1 + length <= budget:
            stays = planner.used + planner.cost(length) <= budget and planner.holds(head)
            if not stays:
                if planner.used + planner.cost(lengths[head] + 1 + length) > budget:
                    planner.close()
                planner.place(Piece(head, 0, lengths[head]))
        planner.place(Piece(index, 0, length))
    planner.close()
    return planner.chunks or [()]


def chunk_layouts(
    question: Tokens,
    state: Sequence[Tokens],
    kinds: Sequence[int],
    groups: Sequence[int],
    chunk_tokens: int,
    special: SpecialTokens,
) -> list[ChunkLayout]:
    """Build the chunks of one request; `question` must already be capped."""
    lengths = [len(segment) for segment in state]
    budget = body_budget(chunk_tokens, len(question))
    layouts: list[ChunkLayout] = []
    for pieces in plan_chunks(lengths, kinds, groups, budget):
        parts: list[Tokens] = [np.array([special.cls]), question, np.array([special.sep])]
        offset = 2 + len(question)
        placed: list[tuple[Piece, int]] = []
        for position, piece in enumerate(pieces):
            if position:
                parts.append(np.array([special.newline]))
                offset += 1
            parts.append(state[piece.segment][piece.token_start : piece.token_end])
            placed.append((piece, offset))
            offset += piece.token_end - piece.token_start
        if pieces:
            parts.append(np.array([special.sep]))
        layouts.append(
            ChunkLayout(np.concatenate(parts).astype(np.int64), len(question), tuple(placed))
        )
    return layouts


def is_crossed(chunks: Sequence[int], candidates: Sequence[int], crossing_tokens: int) -> bool:
    """Return whether candidates are encoded after the first chunk.

    True when the request, with every candidate after its first chunk, fits `crossing_tokens`.
    """
    if not candidates:
        return False
    rows = sum(chunks[0] + length + 1 for length in candidates)
    return sum(chunks) + rows <= crossing_tokens


def candidate_rows(
    question: Tokens,
    candidates: Sequence[Tokens],
    layouts: Sequence[ChunkLayout],
    crossing_tokens: int,
    special: SpecialTokens,
) -> tuple[list[Tokens], int]:
    """Return each candidate's token row and the offset of the candidate text in it."""
    lengths = [len(layout.tokens) for layout in layouts]
    if is_crossed(lengths, [len(candidate) for candidate in candidates], crossing_tokens):
        prefix = layouts[0].tokens
    else:
        prefix = np.concatenate([[special.cls], question, [special.sep]]).astype(np.int64)
    rows = [
        np.concatenate([prefix, candidate, [special.sep]]).astype(np.int64)
        for candidate in candidates
    ]
    return rows, len(prefix)
