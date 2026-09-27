"""Decision inputs: passages, tables and typed fields.

Every entry point accepts a `ContextLike`:

- `str`: one passage;
- `Sequence[str]`: one passage per string;
- `Mapping`: a JSON state, flattened into fields with dotted keys;
- `Context` or `JsonState`.

Numbers and dates in table cells and fields are typed; their original text is kept.
"""

import datetime
import decimal
import math
import numbers
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator
from pydantic import Field as PydanticField

from mimir.core.errors import ContextError
from mimir.core.values import number_text, parse_date, parse_number

if TYPE_CHECKING:
    import pandas
    import polars

FieldKind = Literal["text", "number", "category", "datetime", "list"]
FieldValue = str | float | tuple[str, ...]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Passage(_Frozen):
    """A passage of text with an optional title."""

    text: str
    title: str = ""


class Cell(_Frozen):
    """A table cell with its parsed number and date, if any."""

    text: str
    number: float | None = None
    date: datetime.date | None = None

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Create a cell from raw text, parsing its number and date."""
        found = parse_date(raw)
        return cls(
            text=raw,
            number=parse_number(raw),
            date=None if found is None else datetime.date.fromisoformat(found),
        )


CellValue = str | float | decimal.Decimal | datetime.date | None


def _cell(value: object, row: int, column: int) -> Cell:
    match value:
        case None:
            return Cell(text="")
        case str():
            return Cell.parse(value)
        case bool():
            return Cell(text="true" if value else "false")
        case datetime.datetime() if value.time() == datetime.time():
            return Cell(text=value.date().isoformat(), date=value.date())
        case datetime.datetime():
            return Cell(text=value.isoformat(sep=" "), date=value.date())
        case datetime.date():
            return Cell(text=value.isoformat(), date=value)
        case numbers.Real() | decimal.Decimal():
            number = float(value)
            if not math.isfinite(number):
                message = f"table row {row} column {column}: {value!r} is not a finite number"
                raise ContextError(message)
            exact = isinstance(value, numbers.Integral | decimal.Decimal)
            return Cell(text=str(value) if exact else number_text(number), number=number)
        case _:
            kind = type(value).__name__
            message = f"table row {row} column {column}: {value!r} of type {kind} is not a cell"
            raise ContextError(message)


class Table(_Frozen):
    """A table with an optional caption and header."""

    rows: tuple[tuple[Cell, ...], ...]
    header: tuple[str, ...] = ()
    caption: str = ""

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Sequence[CellValue]],
        *,
        header: Sequence[str] = (),
        caption: str = "",
    ) -> Self:
        """Create a table from raw values.

        Strings are parsed for a number and a date. Numbers keep their value, integers and
        decimals their exact text. Dates are typed; a datetime at midnight is its date.
        Booleans read `true` or `false`, and None is a blank cell.
        """
        return cls(
            rows=tuple(
                tuple(_cell(value, row, column) for column, value in enumerate(values))
                for row, values in enumerate(rows)
            ),
            header=tuple(header),
            caption=caption,
        )

    @classmethod
    def from_dataframe(
        cls, frame: "pandas.DataFrame | polars.DataFrame", *, caption: str = ""
    ) -> Self:
        """Create a table from a pandas or polars DataFrame: its columns are the header, its
        index is dropped, and missing values are blank cells. Cells convert as in `from_rows`.
        """
        if "pandas" in sys.modules:
            import pandas

            if isinstance(frame, pandas.DataFrame):
                present = frame.astype(object).where(frame.notna(), None)
                rows = list(present.itertuples(index=False, name=None))
                return cls.from_rows(
                    rows, header=[str(name) for name in frame.columns], caption=caption
                )
        if "polars" in sys.modules:
            import polars

            if isinstance(frame, polars.DataFrame):
                return cls.from_rows(list(frame.iter_rows()), header=frame.columns, caption=caption)
        message = f"expected a pandas or polars DataFrame; got {type(frame).__name__}"
        raise ContextError(message)


def field_date(value: str) -> datetime.datetime | None:
    """Parse a datetime field value: a date in a `parse_date` format, or ISO 8601."""
    found = parse_date(value)
    if found is not None:
        return datetime.datetime.fromisoformat(found)
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        return None


class Field(_Frozen):
    """A typed field of a structured state, keyed by its dotted path."""

    key: str
    kind: FieldKind
    value: FieldValue

    @model_validator(mode="after")
    def _check_value_matches_kind(self) -> Self:
        value = self.value
        expected = {"number": float, "list": tuple}.get(self.kind, str)
        if not isinstance(value, expected):
            message = f"field {self.key!r} of kind {self.kind} has a {type(value).__name__} value"
            raise ValueError(message)
        if self.kind == "datetime" and isinstance(value, str) and field_date(value) is None:
            message = f"field {self.key!r}: {value!r} is not a date or an ISO 8601 datetime"
            raise ValueError(message)
        return self

    @classmethod
    def from_json(cls, value: JsonValue, prefix: str = "") -> tuple["Field", ...]:
        """Flatten a JSON value into fields, e.g. `{"a": {"b": [1]}}` gives key `a.b[0]`.

        Kinds: a list of strings is one `list` field, booleans and null are `category`,
        numbers are `number`, date strings are `datetime` and other strings are `text`.
        """
        if isinstance(value, Mapping):
            return tuple(
                found
                for name, item in value.items()
                for found in cls.from_json(item, f"{prefix}.{name}" if prefix else str(name))
            )
        if isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                return (cls(key=prefix, kind="list", value=tuple(str(item) for item in value)),)
            return tuple(
                found
                for position, item in enumerate(value)
                for found in cls.from_json(item, f"{prefix}[{position}]")
            )
        return (_scalar_field(prefix, value),)


def _scalar_field(key: str, value: JsonValue) -> Field:
    if isinstance(value, bool):
        return Field(key=key, kind="category", value="true" if value else "false")
    if value is None:
        return Field(key=key, kind="category", value="null")
    if isinstance(value, int | float):
        if not math.isfinite(value):
            message = f"field {key!r}: {value!r} is not a finite number"
            raise ContextError(message)
        return Field(key=key, kind="number", value=float(value))
    if isinstance(value, str):
        return Field(key=key, kind="datetime" if parse_date(value) else "text", value=value)
    message = f"field {key!r}: unsupported JSON value of type {type(value).__name__}"
    raise ContextError(message)


class Context(_Frozen):
    """The input of a decision. May be empty, in which case only the question is read."""

    passages: tuple[Passage, ...] = ()
    tables: tuple[Table, ...] = ()
    fields: tuple[Field, ...] = ()

    @classmethod
    def coerce(cls, value: "ContextLike") -> "Context":
        """Convert a `ContextLike` to a `Context`."""
        if isinstance(value, Context):
            return value
        if isinstance(value, JsonState):
            return cls(fields=Field.from_json(value.state))
        if isinstance(value, str):
            return cls(passages=(Passage(text=value),))
        if isinstance(value, Mapping):
            state: dict[str, JsonValue] = {str(key): item for key, item in value.items()}
            return cls(fields=Field.from_json(state))
        if isinstance(value, Sequence) and all(isinstance(item, str) for item in value):
            return cls(passages=tuple(Passage(text=item) for item in value))
        kind = type(value).__name__
        message = f"expected a Context, a string, a list of strings or a mapping; got {kind}"
        raise ContextError(message)


class JsonState(_Frozen):
    """A structured state in a JSON request body: `{"state": {...}}`."""

    state: dict[str, JsonValue] | list[JsonValue]


ContextLike = Context | JsonState | str | Sequence[str] | Mapping[str, JsonValue]
ContextInput = Annotated[
    str | list[str] | Context | JsonState,
    PydanticField(
        description=(
            "A string (one passage), a list of strings (passages), a Context object, or "
            "{'state': <JSON>} for a structured state."
        )
    ),
]
