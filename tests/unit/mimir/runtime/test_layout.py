import math
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from mimir.core.context import Context, Field, Table
from mimir.core.decisions import Choice, Estimate, Rate, YesNo
from mimir.core.errors import InputLimitError
from mimir.runtime.layout import collate, tokenize
from mimir.runtime.release import ReleaseConfig
from mimir.runtime.rendering import render
from tests.conftest import SPECIAL

INPUT_NAMES = {
    "chunk_ids",
    "chunk_mask",
    "chunk_record",
    "question_length",
    "key_chunk",
    "key_position",
    "key_mask",
    "candidate_ids",
    "candidate_mask",
    "candidate_text_mask",
    "candidate_index",
    "candidate_present",
    "decision_type",
    "typed_kind",
    "typed_number",
    "typed_year",
    "typed_month",
    "typed_day",
    "typed_second",
    "typed_chunk",
    "typed_position",
}


@pytest.fixture
def config(release: Path) -> ReleaseConfig:
    return ReleaseConfig.model_validate_json((release / "config.json").read_bytes())


def test_feed_matches_the_model_inputs(config: ReleaseConfig, tokenizer: Tokenizer) -> None:
    rendered = render(
        Context.coerce("my card was charged twice"), Choice("which team", ["billing", "security"])
    )
    batch = collate([tokenize(rendered, tokenizer, config)], SPECIAL)
    assert set(batch.feed) == INPUT_NAMES
    assert batch.feed["decision_type"].tolist() == [0]
    assert batch.counts == (2,)


def test_chunk_and_candidate_tokens(config: ReleaseConfig, tokenizer: Tokenizer) -> None:
    rendered = render(Context.coerce("card twice"), YesNo("is it late"))
    tokenized = tokenize(rendered, tokenizer, config)
    vocab = tokenizer.get_vocab()
    question = [vocab[word] for word in ("is", "it", "late")]
    state = [vocab["card"], vocab["twice"]]
    chunk = [SPECIAL.cls, *question, SPECIAL.sep, *state, SPECIAL.sep]
    assert tokenized.layouts[0].tokens.tolist() == chunk
    assert [row.tolist() for row in tokenized.rows] == [
        [*chunk, vocab["no"], SPECIAL.sep],
        [*chunk, vocab["yes"], SPECIAL.sep],
    ]
    assert tokenized.text_start == len(chunk)
    assert tokenized.state_tokens == 2
    assert tokenized.encoded_tokens == len(chunk) + 2 * (len(chunk) + 2)
    batch = collate([tokenized], SPECIAL)
    text_mask = batch.feed["candidate_text_mask"]
    assert text_mask.sum(axis=1).tolist() == [1, 1]


def test_identical_candidate_rows_are_shared_across_requests(
    config: ReleaseConfig, tokenizer: Tokenizer
) -> None:
    requests = [
        tokenize(render(Context(), YesNo("is it late")), tokenizer, config),
        tokenize(render(Context(), YesNo("is it late")), tokenizer, config),
    ]
    batch = collate(requests, SPECIAL)
    assert batch.feed["candidate_ids"].shape[0] == 2
    assert batch.feed["candidate_index"].tolist() == [[0, 1], [0, 1]]
    assert batch.feed["chunk_record"].tolist() == [0, 1]


def test_typed_values_point_at_their_first_token(
    config: ReleaseConfig, tokenizer: Tokenizer
) -> None:
    context = Context(
        tables=(Table.from_rows([["total", "1200"]]),),
        fields=(Field(key="due", kind="datetime", value="2026-03-04"),),
    )
    tokenized = tokenize(render(context, Choice("q", ["yes", "no"])), tokenizer, config)
    batch = collate([tokenized], SPECIAL)
    feed = batch.feed
    ids = feed["chunk_ids"]
    positions = list(
        zip(feed["typed_chunk"].tolist(), feed["typed_position"].tolist(), strict=True)
    )
    vocab = tokenizer.get_vocab()
    assert [int(ids[chunk, position]) for chunk, position in positions] == [
        vocab["1200"],
        vocab["2026"],
    ]
    assert feed["typed_kind"].tolist() == [0, 1]
    assert feed["typed_number"][0] == np.float32(1200.0)
    assert math.isnan(float(feed["typed_number"][1]))
    assert (feed["typed_year"][1], feed["typed_month"][1], feed["typed_day"][1]) == (2026, 3, 4)


def test_key_segments_cover_state_tokens_only(config: ReleaseConfig, tokenizer: Tokenizer) -> None:
    context = Context.coerce(["card twice", "refund"])
    batch = collate([tokenize(render(context, YesNo("q")), tokenizer, config)], SPECIAL)
    assert batch.key_segments.tolist() == [[-1, 0, 0, -1, 1]]
    assert batch.feed["key_mask"].tolist() == [[True] * 5]


def test_continuous_only_batches_get_one_inert_candidate_row(
    config: ReleaseConfig, tokenizer: Tokenizer
) -> None:
    tokenized = tokenize(render(Context.coerce("price"), Estimate("q", 0, 1)), tokenizer, config)
    batch = collate([tokenized, tokenized], SPECIAL)
    feed = batch.feed
    assert feed["candidate_ids"].tolist() == [[SPECIAL.cls, SPECIAL.sep]]
    assert feed["candidate_present"].tolist() == [[False], [False]]
    assert feed["candidate_text_mask"].tolist() == [[False, False]]
    assert batch.counts == (0, 0)


def test_limits_are_enforced(config: ReleaseConfig, tokenizer: Tokenizer) -> None:
    too_many = Choice("q", [f"option {index}" for index in range(config.limits.options + 1)])
    with pytest.raises(InputLimitError, match="options is 9"):
        tokenize(render(Context(), too_many), tokenizer, config)
    levels = Rate("q", [f"level {index}" for index in range(config.limits.levels + 1)])
    with pytest.raises(InputLimitError, match="levels is 7"):
        tokenize(render(Context(), levels), tokenizer, config)
    long_text = " ".join(["card"] * (config.limits.context_tokens + 1))
    with pytest.raises(InputLimitError, match="context tokens"):
        tokenize(render(Context.coerce(long_text), YesNo("q")), tokenizer, config)


def test_collate_needs_requests() -> None:
    with pytest.raises(ValueError, match="at least one"):
        collate([], SPECIAL)
