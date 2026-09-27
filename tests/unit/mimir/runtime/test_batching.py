from pathlib import Path

import pytest
from tokenizers import Tokenizer

from mimir.core.context import Context
from mimir.core.decisions import YesNo
from mimir.runtime.batching import pack, padded_tokens
from mimir.runtime.layout import Tokenized, tokenize
from mimir.runtime.release import ReleaseConfig
from mimir.runtime.rendering import render


@pytest.fixture
def requests(release: Path, tokenizer: Tokenizer) -> list[Tokenized]:
    config = ReleaseConfig.model_validate_json((release / "config.json").read_bytes())
    texts = ["card", "card twice card twice card twice", "card twice"]
    return [tokenize(render(Context.coerce(text), YesNo("q")), tokenizer, config) for text in texts]


def test_padded_tokens_pad_every_chunk_and_row_to_the_longest(requests: list[Tokenized]) -> None:
    short, long = requests[0], requests[1]
    chunk = max(len(short.layouts[0].tokens), len(long.layouts[0].tokens))
    row = max(len(row) for request in (short, long) for row in request.rows)
    assert padded_tokens([short, long]) == 2 * chunk + 4 * row
    assert padded_tokens([]) == 0


def test_pack_sorts_longest_first_and_covers_every_request(requests: list[Tokenized]) -> None:
    batches = pack(requests, token_budget=10_000, batch_size=None)
    assert batches == [[1, 2, 0]]


def test_pack_respects_batch_size_and_budget(requests: list[Tokenized]) -> None:
    assert pack(requests, token_budget=10_000, batch_size=2) == [[1, 2], [0]]
    single = pack(requests, token_budget=1, batch_size=None)
    assert single == [[1], [2], [0]]
    assert sorted(index for batch in single for index in batch) == [0, 1, 2]


def test_pack_rejects_a_non_positive_batch_size(requests: list[Tokenized]) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        pack(requests, token_budget=100, batch_size=0)
