"""Tokenizing rendered requests and collating them into graph inputs.

`tokenize` turns one rendered request into token arrays and checks the release's input limits.
`collate` pads a batch of tokenized requests into the named arrays of the graph contract.
Identical candidate rows within a batch are encoded once.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt
from tokenizers import Encoding, Tokenizer

from mimir.core.errors import ContextError, InputLimitError
from mimir.runtime.chunking import ChunkLayout, candidate_rows, chunk_layouts
from mimir.runtime.release import ReleaseConfig, SpecialTokens
from mimir.runtime.rendering import Rendered

NO_SEGMENT: Final = -1

Tokens = npt.NDArray[np.int64]
Array = npt.NDArray[np.generic]


@dataclass(frozen=True, slots=True)
class TypedToken:
    """A typed value located at token `token_start` of state segment `segment`."""

    segment: int
    token_start: int
    kind: int
    number: float
    year: int
    month: int
    day: int
    second: int


@dataclass(frozen=True, slots=True)
class Tokenized:
    """One request as tokens, with its chunks and candidate rows already built."""

    rendered: Rendered
    type_index: int
    state_tokens: int
    typed: tuple[TypedToken, ...]
    layouts: tuple[ChunkLayout, ...]
    rows: tuple[Tokens, ...]
    text_start: int

    @property
    def encoded_tokens(self) -> int:
        """Tokens the encoder processes for this request, before padding."""
        return sum(len(layout.tokens) for layout in self.layouts) + sum(
            len(row) for row in self.rows
        )


@dataclass(frozen=True, slots=True)
class Batch:
    """Graph inputs for a batch, plus the state segment of each key position.

    `key_segments[r, k]` is the state segment of key `k` of request `r`, or -1 for the `[CLS]`
    key, separators and padding.
    """

    feed: dict[str, Array]
    key_segments: npt.NDArray[np.int64]
    counts: tuple[int, ...]


def _ids(encoding: Encoding) -> Tokens:
    return np.array(encoding.ids, dtype=np.int64)


def _token_start(encoding: Encoding, char_start: int, char_end: int, where: str) -> int:
    offsets = np.array(encoding.offsets, dtype=np.int64).reshape(-1, 2)
    overlapping = np.flatnonzero((offsets[:, 0] < char_end) & (offsets[:, 1] > char_start))
    if overlapping.size == 0:
        message = f"{where}: characters [{char_start}, {char_end}) map to no token"
        raise ContextError(message)
    return int(overlapping[0])


def check_limits(rendered: Rendered, state_tokens: int, config: ReleaseConfig) -> None:
    """Raise `InputLimitError` if the request exceeds the release's tested limits."""
    limits = config.limits
    count = len(rendered.candidates)
    if rendered.model_type == "ordinal":
        if count > limits.levels:
            raise InputLimitError(limit="levels", value=count, maximum=limits.levels)
    elif count > limits.options:
        raise InputLimitError(limit="options", value=count, maximum=limits.options)
    if state_tokens > limits.context_tokens:
        raise InputLimitError(
            limit="context tokens", value=state_tokens, maximum=limits.context_tokens
        )


def tokenize(rendered: Rendered, tokenizer: Tokenizer, config: ReleaseConfig) -> Tokenized:
    """Tokenize one rendered request and build its chunks and candidate rows."""
    state_texts = [segment.text for segment in rendered.state]
    encodings = tokenizer.encode_batch(
        [rendered.question, *state_texts, *rendered.candidates], add_special_tokens=False
    )
    question_encoding = encodings[0]
    state_encodings = encodings[1 : 1 + len(state_texts)]
    candidate_encodings = encodings[1 + len(state_texts) :]
    state = [_ids(encoding) for encoding in state_encodings]
    state_tokens = sum(len(tokens) for tokens in state)
    check_limits(rendered, state_tokens, config)
    typed = tuple(
        TypedToken(
            segment=value.segment,
            token_start=_token_start(
                state_encodings[value.segment],
                value.char_start,
                value.char_end,
                f"state segment {value.segment}",
            ),
            kind=value.kind,
            number=value.number,
            year=value.year,
            month=value.month,
            day=value.day,
            second=value.second,
        )
        for value in rendered.typed
    )
    layout = config.layout
    question = _ids(question_encoding)[: layout.question_cap]
    layouts = chunk_layouts(
        question,
        state,
        [segment.kind for segment in rendered.state],
        [segment.group for segment in rendered.state],
        layout.chunk_tokens,
        layout.special_tokens,
    )
    rows, text_start = candidate_rows(
        question,
        [_ids(encoding) for encoding in candidate_encodings],
        layouts,
        layout.crossing_tokens,
        layout.special_tokens,
    )
    return Tokenized(
        rendered=rendered,
        type_index=config.type_index(rendered.model_type),
        state_tokens=state_tokens,
        typed=typed,
        layouts=tuple(layouts),
        rows=tuple(rows),
        text_start=text_start,
    )


def _key_positions(layout: ChunkLayout) -> list[int]:
    """Return the chunk's key positions: `[CLS]` and every state token."""
    if not layout.pieces:
        return [0]
    return [0, *range(layout.question_tokens + 2, len(layout.tokens) - 1)]


def _key_segments(layout: ChunkLayout) -> list[int]:
    owner = np.full(len(layout.tokens), NO_SEGMENT, dtype=np.int64)
    for piece, offset in layout.pieces:
        owner[offset : offset + piece.token_end - piece.token_start] = piece.segment
    return [int(owner[position]) if position else NO_SEGMENT for position in _key_positions(layout)]


def _typed_positions(
    layouts: Sequence[ChunkLayout], first_chunk: int, segment: int, token: int
) -> list[tuple[int, int]]:
    hits = [
        (first_chunk + chunk, offset + token - piece.token_start)
        for chunk, layout in enumerate(layouts)
        for piece, offset in layout.pieces
        if piece.segment == segment and piece.token_start <= token < piece.token_end
    ]
    if not hits:
        message = f"typed value at segment {segment} token {token} is in no chunk"
        raise ContextError(message)
    return hits


@dataclass
class _Collector:
    chunks: list[Tokens] = field(default_factory=list)
    chunk_request: list[int] = field(default_factory=list)
    question_length: list[int] = field(default_factory=list)
    keys: list[list[tuple[int, int]]] = field(default_factory=list)
    key_segments: list[list[int]] = field(default_factory=list)
    unique: dict[bytes, int] = field(default_factory=dict)
    candidates: list[Tokens] = field(default_factory=list)
    candidate_starts: list[int] = field(default_factory=list)
    candidate_index: list[list[int]] = field(default_factory=list)
    typed: list[tuple[int, int, int, float, int, int, int, int]] = field(default_factory=list)
    decision_type: list[int] = field(default_factory=list)

    def add(self, request: Tokenized) -> None:
        row = len(self.decision_type)
        first_chunk = len(self.chunks)
        keys: list[tuple[int, int]] = []
        segments: list[int] = []
        for chunk, layout in enumerate(request.layouts):
            self.chunks.append(layout.tokens)
            self.chunk_request.append(row)
            self.question_length.append(layout.question_tokens)
            keys.extend((first_chunk + chunk, position) for position in _key_positions(layout))
            segments.extend(_key_segments(layout))
        self.keys.append(keys)
        self.key_segments.append(segments)
        indices: list[int] = []
        for sequence in request.rows:
            key = sequence.tobytes()
            if key not in self.unique:
                self.unique[key] = len(self.candidates)
                self.candidates.append(sequence)
                self.candidate_starts.append(request.text_start)
            indices.append(self.unique[key])
        self.candidate_index.append(indices)
        for value in request.typed:
            for chunk, position in _typed_positions(
                request.layouts, first_chunk, value.segment, value.token_start
            ):
                self.typed.append(
                    (
                        chunk,
                        position,
                        value.kind,
                        value.number,
                        value.year,
                        value.month,
                        value.day,
                        value.second,
                    )
                )
        self.decision_type.append(request.type_index)


def _pad(rows: Sequence[Sequence[int] | Tokens], width: int, fill: int) -> Tokens:
    out = np.full((len(rows), width), fill, dtype=np.int64)
    for index, row in enumerate(rows):
        out[index, : len(row)] = row
    return out


def collate(requests: Sequence[Tokenized], special: SpecialTokens) -> Batch:
    """Pad a non-empty batch of requests into graph inputs."""
    if not requests:
        message = "collate needs at least one request"
        raise ValueError(message)
    collector = _Collector()
    for request in requests:
        collector.add(request)
    records = len(requests)
    counts = tuple(len(indices) for indices in collector.candidate_index)
    width = max(counts)
    present = np.zeros((records, width), dtype=np.bool_)
    for row, count in enumerate(counts):
        present[row, :count] = True
    chunk_width = max(len(chunk) for chunk in collector.chunks)
    key_width = max(len(keys) for keys in collector.keys)
    typed = np.array(collector.typed, dtype=np.float64).reshape(-1, 8)
    if collector.candidates:
        candidate_width = max(len(sequence) for sequence in collector.candidates)
        candidate_ids = _pad(collector.candidates, candidate_width, special.pad)
        lengths = np.array([len(sequence) for sequence in collector.candidates])
        starts = np.array(collector.candidate_starts)
        positions = np.arange(candidate_width)
        candidate_mask = positions < lengths[:, None]
        candidate_text_mask = (positions >= starts[:, None]) & (positions < lengths[:, None] - 1)
        candidate_index = _pad(collector.candidate_index, width, 0)
    else:
        # Continuous requests alone have no candidates; the graph still needs one row, which
        # no request's mask admits, so every output is unchanged.
        candidate_ids = np.array([[special.cls, special.sep]], dtype=np.int64)
        candidate_mask = np.ones((1, 2), dtype=np.bool_)
        candidate_text_mask = np.zeros((1, 2), dtype=np.bool_)
        candidate_index = np.zeros((records, 1), dtype=np.int64)
        present = np.zeros((records, 1), dtype=np.bool_)
    key_mask = _pad([[1] * len(keys) for keys in collector.keys], key_width, 0).astype(np.bool_)
    feed: dict[str, Array] = {
        "chunk_ids": _pad(collector.chunks, chunk_width, special.pad),
        "chunk_mask": _pad([[1] * len(chunk) for chunk in collector.chunks], chunk_width, 0).astype(
            np.bool_
        ),
        "chunk_record": np.array(collector.chunk_request, dtype=np.int64),
        "question_length": np.array(collector.question_length, dtype=np.int64),
        "key_chunk": _pad([[chunk for chunk, _ in keys] for keys in collector.keys], key_width, 0),
        "key_position": _pad([[pos for _, pos in keys] for keys in collector.keys], key_width, 0),
        "key_mask": key_mask,
        "candidate_ids": candidate_ids,
        "candidate_mask": candidate_mask,
        "candidate_text_mask": candidate_text_mask,
        "candidate_index": candidate_index,
        "candidate_present": present,
        "decision_type": np.array(collector.decision_type, dtype=np.int64),
        "typed_kind": typed[:, 2].astype(np.int64),
        "typed_number": typed[:, 3].astype(np.float32),
        "typed_year": typed[:, 4].astype(np.int64),
        "typed_month": typed[:, 5].astype(np.int64),
        "typed_day": typed[:, 6].astype(np.int64),
        "typed_second": typed[:, 7].astype(np.int64),
        "typed_chunk": typed[:, 0].astype(np.int64),
        "typed_position": typed[:, 1].astype(np.int64),
    }
    return Batch(
        feed=feed,
        key_segments=_pad(collector.key_segments, key_width, NO_SEGMENT),
        counts=counts,
    )
