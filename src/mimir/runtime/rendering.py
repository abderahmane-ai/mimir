"""Rendering a context into text segments and typed values.

Segment order: passages (title on the first line), then tables (caption and header as one
segment, then one segment per non-blank row with cells joined by ` | `), then fields as
`key: value`. Numbers and dates are located by character span within their segment. The
format matches the training data and is checked by the golden fixtures.
"""

import datetime
import math
from dataclasses import dataclass
from typing import Final

from mimir.core.context import Context, Field, Table, field_date
from mimir.core.decisions import DecisionSpec, ModelType
from mimir.core.results import ContextKind
from mimir.core.values import number_text

# Segment kind codes of the training data; 0 and 5 are the question and candidates.
PASSAGE: Final = 1
TABLE_HEAD: Final = 2
TABLE_ROW: Final = 3
FIELD: Final = 4
CELL_SEPARATOR: Final = " | "
LIST_SEPARATOR: Final = ", "
NUMBER: Final = 0
DATE: Final = 1
SECONDS_PER_HOUR: Final = 3600
SECONDS_PER_MINUTE: Final = 60


@dataclass(frozen=True, slots=True)
class Part:
    """The context part a segment was rendered from."""

    kind: ContextKind
    index: int
    row: int | None


@dataclass(frozen=True, slots=True)
class Segment:
    """A state segment. `group` is the index of its passage, table or field."""

    kind: int
    group: int
    text: str
    part: Part


@dataclass(frozen=True, slots=True)
class TypedValue:
    """A number or date at characters `[char_start, char_end)` of a state segment."""

    segment: int
    char_start: int
    char_end: int
    kind: int
    number: float
    year: int
    month: int
    day: int
    second: int


@dataclass(frozen=True, slots=True)
class Rendered:
    model_type: ModelType
    question: str
    state: tuple[Segment, ...]
    candidates: tuple[str, ...]
    typed: tuple[TypedValue, ...]


def _number(segment: int, start: int, end: int, value: float) -> TypedValue:
    return TypedValue(segment, start, end, NUMBER, value, 0, 0, 0, 0)


def _date(segment: int, start: int, end: int, moment: datetime.date, second: int) -> TypedValue:
    return TypedValue(
        segment, start, end, DATE, math.nan, moment.year, moment.month, moment.day, second
    )


class _Builder:
    def __init__(self) -> None:
        self.state: list[Segment] = []
        self.typed: list[TypedValue] = []

    def add(self, kind: int, group: int, text: str, part: Part) -> int:
        self.state.append(Segment(kind, group, text, part))
        return len(self.state) - 1

    def add_table(self, group: int, table: Table) -> None:
        head = "\n".join(
            part for part in (table.caption, CELL_SEPARATOR.join(table.header)) if part
        )
        if head.strip():
            self.add(TABLE_HEAD, group, head, Part("table", group, None))
        for row_index, row in enumerate(table.rows):
            texts = [cell.text for cell in row]
            if not "".join(texts).strip():
                continue
            text = CELL_SEPARATOR.join(texts)
            segment = self.add(TABLE_ROW, group, text, Part("table_row", group, row_index))
            start = 0
            for cell in row:
                end = start + len(cell.text)
                if cell.number is not None and end > start:
                    self.typed.append(_number(segment, start, end, cell.number))
                if cell.date is not None and end > start:
                    self.typed.append(_date(segment, start, end, cell.date, 0))
                start = end + len(CELL_SEPARATOR)

    def add_field(self, group: int, field: Field) -> None:
        value = field.value
        if isinstance(value, float):
            value_text = number_text(value)
        elif isinstance(value, tuple):
            value_text = LIST_SEPARATOR.join(value)
        else:
            value_text = value
        prefix = f"{field.key}: "
        segment = self.add(FIELD, group, prefix + value_text, Part("field", group, None))
        start, end = len(prefix), len(prefix) + len(value_text)
        if isinstance(value, float):
            self.typed.append(_number(segment, start, end, value))
        elif field.kind == "datetime" and isinstance(value, str):
            moment = field_date(value)
            if moment is not None:
                second = (
                    moment.hour * SECONDS_PER_HOUR
                    + moment.minute * SECONDS_PER_MINUTE
                    + moment.second
                )
                self.typed.append(_date(segment, start, end, moment, second))


def render(context: Context, spec: DecisionSpec) -> Rendered:
    """Render a context for a decision spec."""
    return render_request(context, spec.model_type, spec.model_question, spec.option_texts)


def render_request(
    context: Context, model_type: ModelType, question: str, candidates: tuple[str, ...]
) -> Rendered:
    """Render a context with an explicit model decision type, question and candidate texts."""
    builder = _Builder()
    for group, passage in enumerate(context.passages):
        text = f"{passage.title}\n{passage.text}" if passage.title else passage.text
        if text.strip():
            builder.add(PASSAGE, group, text, Part("passage", group, None))
    for group, table in enumerate(context.tables):
        builder.add_table(group, table)
    for group, field in enumerate(context.fields):
        builder.add_field(group, field)
    return Rendered(
        model_type=model_type,
        question=question,
        state=tuple(builder.state),
        candidates=candidates,
        typed=tuple(builder.typed),
    )
