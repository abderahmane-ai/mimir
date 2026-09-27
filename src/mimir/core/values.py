"""Parsing of numbers and calendar dates from raw strings.

The accepted formats match the model's training data, so a value is typed here exactly when
it was typed during training.
"""

import datetime
import re
from typing import Final

MINUS_SIGN: Final = chr(0x2212)
_GROUPED: Final = r"\d{1,3}(?:,\d{3})+"
_PLAIN: Final = r"\d+"
NUMBER_PATTERN: Final = re.compile(
    rf"^(?P<sign>[-+{MINUS_SIGN}]?)(?P<currency>[$€£]?)(?P<digits>{_GROUPED}|{_PLAIN})"
    r"(?P<fraction>\.\d+)?(?P<percent>%?)$",
    re.ASCII,
)
MONTHS: Final = {
    name: position
    for position, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
ISO_DATE: Final = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$", re.ASCII)
MONTH_DAY_YEAR: Final = re.compile(r"^([a-z]+)\.? (\d{1,2})(?:st|nd|rd|th)? ?, ?(\d{4})$", re.ASCII)
DAY_MONTH_YEAR: Final = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)? ([a-z]+)\.?,? (\d{4})$", re.ASCII)
# Integral numbers below this magnitude are written without a decimal point.
INTEGRAL_TEXT_LIMIT: Final = 1e15


def parse_number(raw: str) -> float | None:
    """Parse a number, or return None.

    Accepts a sign (ASCII or U+2212), one currency symbol, comma-grouped or plain digits, a
    decimal fraction, a trailing percent sign and accounting parentheses for negatives.
    The percent sign does not scale the value: `"12%"` parses as `12.0`.
    """
    candidate = raw.strip()
    negative = False
    if candidate.startswith("(") and candidate.endswith(")"):
        candidate = candidate[1:-1].strip()
        negative = True
    match = NUMBER_PATTERN.match(candidate)
    if match is None:
        return None
    magnitude = float(match["digits"].replace(",", "") + (match["fraction"] or ""))
    if match["sign"] in {"-", MINUS_SIGN}:
        negative = not negative
    return -magnitude if negative else magnitude


def _calendar_date(year: int, month: int, day: int) -> str | None:
    try:
        return datetime.date(year, month, day).isoformat()
    except ValueError:
        return None


def parse_date(raw: str) -> str | None:
    """Parse a calendar date to ISO 8601 (`YYYY-MM-DD`), or return None.

    Accepts `2026-03-04`, `March 4, 2026`, `Mar. 4th, 2026` and `4 March 2026`, ignoring case.
    Returns None for dates that do not exist, such as February 30.
    """
    candidate = " ".join(raw.strip().lower().split())
    if match := ISO_DATE.match(candidate):
        return _calendar_date(int(match[1]), int(match[2]), int(match[3]))
    if (match := MONTH_DAY_YEAR.match(candidate)) and match[1] in MONTHS:
        return _calendar_date(int(match[3]), MONTHS[match[1]], int(match[2]))
    if (match := DAY_MONTH_YEAR.match(candidate)) and match[2] in MONTHS:
        return _calendar_date(int(match[3]), MONTHS[match[2]], int(match[1]))
    return None


def number_text(number: float) -> str:
    """Format a number as it appears in rendered text; integral values have no fraction."""
    if number.is_integer() and abs(number) < INTEGRAL_TEXT_LIMIT:
        return str(int(number))
    return repr(number)
