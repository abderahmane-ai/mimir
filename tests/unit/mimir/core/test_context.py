import datetime
import decimal
import math

import pandas
import polars
import pytest
from pydantic import TypeAdapter, ValidationError

from mimir.core.context import (
    Cell,
    Context,
    ContextInput,
    Field,
    JsonState,
    Passage,
    Table,
    field_date,
)
from mimir.core.errors import ContextError


def test_cell_parse_types_numbers_and_dates() -> None:
    assert Cell.parse("$1,200") == Cell(text="$1,200", number=1200.0)
    assert Cell.parse("2026-03-04") == Cell(text="2026-03-04", date=datetime.date(2026, 3, 4))
    assert Cell.parse("hello") == Cell(text="hello")


def test_table_from_rows_keeps_numbers_exact_and_parses_strings() -> None:
    table = Table.from_rows([["$5", 3], [2.5, "x"]], header=["a", "b"], caption="c")
    assert table.rows[0] == (Cell(text="$5", number=5.0), Cell(text="3", number=3.0))
    assert table.rows[1] == (Cell(text="2.5", number=2.5), Cell(text="x"))
    assert table.header == ("a", "b")
    assert table.caption == "c"


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, decimal.Decimal("NaN")])
def test_table_from_rows_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ContextError, match=r"row 0 column 1: .* is not a finite number"):
        Table.from_rows([["ok", value]])


@pytest.mark.parametrize("value", [["a"], {"a": 1}, b"bytes", object()])
def test_table_from_rows_rejects_other_types_with_their_location(value: object) -> None:
    with pytest.raises(ContextError, match=r"row 1 column 0: .* of type .* is not a cell"):
        Table.from_rows([["ok"], [value]])


def test_table_from_rows_converts_every_cell_type() -> None:
    big = 12_345_678_901_234_567
    table = Table.from_rows(
        [
            [
                None,
                True,
                False,
                big,
                decimal.Decimal("1.50"),
                datetime.date(2026, 3, 4),
                datetime.datetime(2026, 3, 4, 0, 0),
                datetime.datetime(2026, 3, 4, 10, 30),
            ]
        ]
    )
    assert table.rows[0] == (
        Cell(text=""),
        Cell(text="true"),
        Cell(text="false"),
        Cell(text=str(big), number=float(big)),
        Cell(text="1.50", number=1.5),
        Cell(text="2026-03-04", date=datetime.date(2026, 3, 4)),
        Cell(text="2026-03-04", date=datetime.date(2026, 3, 4)),
        Cell(text="2026-03-04 10:30:00", date=datetime.date(2026, 3, 4)),
    )


def test_table_from_pandas_takes_columns_drops_the_index_and_blanks_missing_values() -> None:
    frame = pandas.DataFrame(
        {
            "customer": ["Alice Smith", None],
            "spend": [62_400.5, math.nan],
            "orders": pandas.array([3, None], dtype="Int64"),
            "active": [True, False],
            "since": [pandas.Timestamp("2026-03-04"), pandas.NaT],
        },
        index=["a", "b"],
    )
    table = Table.from_dataframe(frame, caption="Customers")
    assert table.header == ("customer", "spend", "orders", "active", "since")
    assert table.caption == "Customers"
    assert table.rows == (
        (
            Cell(text="Alice Smith"),
            Cell(text="62400.5", number=62_400.5),
            Cell(text="3", number=3.0),
            Cell(text="true"),
            Cell(text="2026-03-04", date=datetime.date(2026, 3, 4)),
        ),
        (Cell(text=""), Cell(text=""), Cell(text=""), Cell(text="false"), Cell(text="")),
    )


def test_table_from_polars_matches_the_same_rows() -> None:
    frame = polars.DataFrame(
        {
            "customer": ["Bob Jones", None],
            "spend": ["$4,100.00", None],
            "joined": [datetime.date(2026, 3, 4), None],
            "credit": [decimal.Decimal("1.50"), None],
        }
    )
    table = Table.from_dataframe(frame)
    assert table.header == ("customer", "spend", "joined", "credit")
    assert table.rows == (
        (
            Cell(text="Bob Jones"),
            Cell(text="$4,100.00", number=4100.0),
            Cell(text="2026-03-04", date=datetime.date(2026, 3, 4)),
            Cell(text="1.50", number=1.5),
        ),
        (Cell(text=""), Cell(text=""), Cell(text=""), Cell(text="")),
    )


def test_table_from_an_empty_dataframe_keeps_its_header() -> None:
    table = Table.from_dataframe(pandas.DataFrame({"a": [], "b": []}))
    assert (table.header, table.rows) == (("a", "b"), ())


def test_table_from_dataframe_names_the_wrong_input_type() -> None:
    with pytest.raises(ContextError, match="expected a pandas or polars DataFrame; got dict"):
        Table.from_dataframe({"a": [1]})


def test_table_from_dataframe_reports_the_failing_cell() -> None:
    with pytest.raises(ContextError, match=r"row 0 column 1: .* of type list is not a cell"):
        Table.from_dataframe(polars.DataFrame({"a": [1], "b": [[1, 2]]}))


def test_table_allows_ragged_and_empty_rows() -> None:
    table = Table.from_rows([[], ["a", "b", "c"]], header=["x"])
    assert [len(row) for row in table.rows] == [0, 3]


def test_field_rejects_value_of_the_wrong_kind() -> None:
    with pytest.raises(ValidationError, match="field 'a' of kind number has a str value"):
        Field(key="a", kind="number", value="3")
    with pytest.raises(ValidationError, match="not a date"):
        Field(key="d", kind="datetime", value="yesterday")


def test_field_date_accepts_training_formats_and_iso_datetimes() -> None:
    assert field_date("March 4, 2026") == datetime.datetime(2026, 3, 4)
    assert field_date("2026-03-04T10:30:05") == datetime.datetime(2026, 3, 4, 10, 30, 5)
    assert field_date("soon") is None


def test_from_json_flattens_with_dotted_keys_and_brackets() -> None:
    fields = Field.from_json(
        {
            "customer": {"plan": "enterprise", "seats": 240, "since": "March 4, 2026"},
            "tags": ["a", "b"],
            "lines": [{"qty": 2}, None, True],
            "empty": [],
        }
    )
    assert fields == (
        Field(key="customer.plan", kind="text", value="enterprise"),
        Field(key="customer.seats", kind="number", value=240.0),
        Field(key="customer.since", kind="datetime", value="March 4, 2026"),
        Field(key="tags", kind="list", value=("a", "b")),
        Field(key="lines[0].qty", kind="number", value=2.0),
        Field(key="lines[1]", kind="category", value="null"),
        Field(key="lines[2]", kind="category", value="true"),
        Field(key="empty", kind="list", value=()),
    )


def test_from_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ContextError, match="'x'"):
        Field.from_json({"x": math.inf})


@pytest.mark.parametrize(
    "text",
    [
        "\N{ZERO WIDTH NO-BREAK SPACE}BOM first",
        "non\N{NO-BREAK SPACE}breaking",
        "zero\N{ZERO WIDTH SPACE}width",
        "e\N{COMBINING ACUTE ACCENT} combining",
        "\N{RIGHT-TO-LEFT OVERRIDE}right to left",
        "emoji \U0001f600",
        "null \x00 byte",
        "",
    ],
)
def test_passage_text_is_kept_verbatim(text: str) -> None:
    assert Context.coerce(text).passages == (Passage(text=text),)


def test_coerce_accepts_every_context_like() -> None:
    context = Context(passages=(Passage(text="a"),))
    assert Context.coerce(context) is context
    assert Context.coerce(["a", "b"]).passages == (Passage(text="a"), Passage(text="b"))
    assert Context.coerce({"n": 1}).fields == (Field(key="n", kind="number", value=1.0),)
    assert Context.coerce(JsonState(state=[1])).fields == (
        Field(key="[0]", kind="number", value=1.0),
    )
    assert Context.coerce([]) == Context()


def test_coerce_rejects_other_types() -> None:
    with pytest.raises(ContextError, match="got int"):
        Context.coerce(3)


def test_context_input_distinguishes_context_and_state_objects() -> None:
    adapter = TypeAdapter[ContextInput](ContextInput)
    assert isinstance(adapter.validate_python({"passages": [{"text": "a"}]}), Context)
    assert isinstance(adapter.validate_python({"state": {"a": 1}}), JsonState)
    assert adapter.validate_python(["a"]) == ["a"]
    with pytest.raises(ValidationError):
        adapter.validate_python({"unknown": 1})


def test_models_are_frozen() -> None:
    passage = Passage(text="a")
    attribute = "text"
    with pytest.raises(ValidationError, match="frozen"):
        setattr(passage, attribute, "b")
