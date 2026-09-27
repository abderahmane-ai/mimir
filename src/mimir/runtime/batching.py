"""Packing requests into batches by padded token count."""

from collections.abc import Sequence

from mimir.runtime.layout import Tokenized


def padded_tokens(requests: Sequence[Tokenized]) -> int:
    """Return the encoder tokens of a batch after padding chunks and candidate rows."""
    chunks = [len(layout.tokens) for request in requests for layout in request.layouts]
    rows = [len(row) for request in requests for row in request.rows]
    return len(chunks) * max(chunks, default=0) + len(rows) * max(rows, default=0)


def pack(
    requests: Sequence[Tokenized], token_budget: int, batch_size: int | None
) -> list[list[int]]:
    """Group request indices into batches, longest first.

    A batch grows while its padded token count stays within `token_budget` and it has fewer
    than `batch_size` requests. A request over the budget runs alone.
    """
    if batch_size is not None and batch_size < 1:
        message = f"batch_size must be at least 1; got {batch_size}"
        raise ValueError(message)
    order = sorted(
        range(len(requests)), key=lambda index: requests[index].encoded_tokens, reverse=True
    )
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        candidate = [*current, index]
        is_full = batch_size is not None and len(current) >= batch_size
        over = padded_tokens([requests[position] for position in candidate]) > token_budget
        if current and (is_full or over):
            batches.append(current)
            candidate = [index]
        current = candidate
    if current:
        batches.append(current)
    return batches
