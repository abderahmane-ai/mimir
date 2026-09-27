import datetime
import math

from mimir.core.context import Cell, Context, Field, Passage, Table
from mimir.core.decisions import Choice, Verify
from mimir.runtime.rendering import (
    DATE,
    FIELD,
    NUMBER,
    PASSAGE,
    TABLE_HEAD,
    TABLE_ROW,
    Part,
    render,
)


def test_segments_follow_passages_tables_then_fields() -> None:
    context = Context(
        passages=(Passage(title="Ticket", text="Card charged twice"), Passage(text="  ")),
        tables=(
            Table(
                caption="Lines",
                header=("sku", "qty"),
                rows=(
                    (Cell(text="A"), Cell(text="2", number=2.0)),
                    (Cell(text=""), Cell(text=" ")),
                ),
            ),
        ),
        fields=(Field(key="plan", kind="text", value="pro"),),
    )
    rendered = render(context, Choice("q", ["a", "b", "c"]))
    assert [(segment.kind, segment.group, segment.text) for segment in rendered.state] == [
        (PASSAGE, 0, "Ticket\nCard charged twice"),
        (TABLE_HEAD, 0, "Lines\nsku | qty"),
        (TABLE_ROW, 0, "A | 2"),
        (FIELD, 0, "plan: pro"),
    ]
    assert [segment.part for segment in rendered.state] == [
        Part("passage", 0, None),
        Part("table", 0, None),
        Part("table_row", 0, 0),
        Part("field", 0, None),
    ]
    assert rendered.candidates == ("a", "b", "c")
    assert rendered.model_type == "categorical"


def test_blank_rows_are_skipped_but_row_indices_are_kept() -> None:
    table = Table.from_rows([["", ""], ["x", "1"]])
    rendered = render(Context(tables=(table,)), Choice("q", ["a", "b"]))
    assert [segment.part for segment in rendered.state] == [Part("table_row", 0, 1)]


def test_typed_values_are_located_by_character_span() -> None:
    table = Table.from_rows([["total", "$1,200", "2026-03-04"]])
    fields = (
        Field(key="seats", kind="number", value=240.0),
        Field(key="due", kind="datetime", value="2026-03-04T10:30:05"),
        Field(key="since", kind="datetime", value="March 4, 2026"),
    )
    rendered = render(Context(tables=(table,), fields=fields), Choice("q", ["a", "b"]))
    row, seats, due, since = rendered.state
    values = rendered.typed
    assert [(value.segment, value.kind) for value in values] == [
        (0, NUMBER),
        (0, DATE),
        (1, NUMBER),
        (2, DATE),
        (3, DATE),
    ]
    number, date, seat, moment, day = values
    assert row.text[number.char_start : number.char_end] == "$1,200"
    assert number.number == 1200.0
    assert row.text[date.char_start : date.char_end] == "2026-03-04"
    assert (date.year, date.month, date.day, date.second) == (2026, 3, 4, 0)
    assert math.isnan(date.number)
    assert seats.text[seat.char_start : seat.char_end] == "240"
    assert due.text[moment.char_start : moment.char_end] == "2026-03-04T10:30:05"
    assert moment.second == 10 * 3600 + 30 * 60 + 5
    assert since.text[day.char_start : day.char_end] == "March 4, 2026"


def test_list_fields_join_with_commas() -> None:
    rendered = render(
        Context(fields=(Field(key="tags", kind="list", value=("a", "b")),)), Choice("q", ["x", "y"])
    )
    assert rendered.state[0].text == "tags: a, b"
    assert rendered.typed == ()


def test_verify_renders_its_fixed_question() -> None:
    rendered = render(Context(), Verify("It rained."))
    assert rendered.question == "Judge this statement against the evidence: It rained."
    assert rendered.state == ()


def test_cell_dates_use_midnight() -> None:
    cell = Cell(text="4 March 2026", date=datetime.date(2026, 3, 4))
    rendered = render(Context(tables=(Table(rows=((cell,),)),)), Choice("q", ["a", "b"]))
    assert rendered.typed[0].second == 0
