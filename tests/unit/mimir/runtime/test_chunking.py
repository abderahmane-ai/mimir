import numpy as np
import pytest

from mimir.core.errors import InputLimitError
from mimir.runtime.chunking import (
    Piece,
    body_budget,
    candidate_rows,
    chunk_layouts,
    is_crossed,
    plan_chunks,
)
from mimir.runtime.rendering import PASSAGE, TABLE_HEAD, TABLE_ROW
from tests.conftest import SPECIAL


def test_body_budget_leaves_room_for_three_specials() -> None:
    assert body_budget(64, 10) == 51
    with pytest.raises(InputLimitError, match="question tokens"):
        body_budget(8, 5)


def test_segments_pack_into_chunks_with_joins_counted() -> None:
    chunks = plan_chunks([4, 4, 4], [PASSAGE] * 3, [0, 1, 2], budget=9)
    assert chunks == [(Piece(0, 0, 4), Piece(1, 0, 4)), (Piece(2, 0, 4),)]


def test_long_segments_are_split_at_the_budget() -> None:
    chunks = plan_chunks([2, 7], [PASSAGE, PASSAGE], [0, 1], budget=3)
    assert chunks == [(Piece(0, 0, 2),), (Piece(1, 0, 3),), (Piece(1, 3, 6),), (Piece(1, 6, 7),)]


def test_table_rows_repeat_their_header_in_a_new_chunk() -> None:
    kinds = [TABLE_HEAD, TABLE_ROW, TABLE_ROW]
    chunks = plan_chunks([2, 3, 3], kinds, [0, 0, 0], budget=7)
    assert chunks == [
        (Piece(0, 0, 2), Piece(1, 0, 3)),
        (Piece(0, 0, 2), Piece(2, 0, 3)),
    ]


def test_empty_state_gives_one_empty_chunk() -> None:
    assert plan_chunks([], [], [], budget=10) == [()]
    layouts = chunk_layouts(np.array([7, 8]), [], [], [], 16, SPECIAL)
    assert layouts[0].tokens.tolist() == [SPECIAL.cls, 7, 8, SPECIAL.sep]
    assert layouts[0].pieces == ()


def test_chunk_layout_tokens_and_offsets() -> None:
    state = [np.array([10, 11]), np.array([12])]
    layouts = chunk_layouts(np.array([7]), state, [PASSAGE, PASSAGE], [0, 1], 16, SPECIAL)
    assert len(layouts) == 1
    layout = layouts[0]
    assert layout.tokens.tolist() == [
        SPECIAL.cls,
        7,
        SPECIAL.sep,
        10,
        11,
        SPECIAL.newline,
        12,
        SPECIAL.sep,
    ]
    assert [(piece.segment, offset) for piece, offset in layout.pieces] == [(0, 3), (1, 6)]


def test_candidates_cross_behind_the_first_chunk_only_when_everything_fits() -> None:
    assert is_crossed([10], [2, 2], crossing_tokens=100)
    assert not is_crossed([10], [2, 2], crossing_tokens=30)
    assert not is_crossed([10], [], crossing_tokens=100)
    question = np.array([7])
    layouts = chunk_layouts(question, [np.array([10])], [PASSAGE], [0], 16, SPECIAL)
    crossed, start = candidate_rows(question, [np.array([20])], layouts, 100, SPECIAL)
    assert crossed[0].tolist() == [*layouts[0].tokens.tolist(), 20, SPECIAL.sep]
    assert start == len(layouts[0].tokens)
    plain, start = candidate_rows(question, [np.array([20])], layouts, 5, SPECIAL)
    assert plain[0].tolist() == [SPECIAL.cls, 7, SPECIAL.sep, 20, SPECIAL.sep]
    assert start == 3
