import pytest

from mimir.core.values import number_text, parse_date, parse_number


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", 42.0),
        ("  42  ", 42.0),
        ("-3.5", -3.5),
        ("\N{MINUS SIGN}3.5", -3.5),
        ("+7", 7.0),
        ("$1,200", 1200.0),
        ("€1,234,567.89", 1234567.89),
        ("12%", 12.0),
        ("(45)", -45.0),
        ("(-45)", 45.0),
        ("0.001", 0.001),
    ],
)
def test_parse_number_reads_supported_forms(raw: str, expected: float) -> None:
    assert parse_number(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "abc",
        "1,20",
        "12,3456",
        "1.2.3",
        "$$5",
        "5$",
        "1e5",
        "\N{ARABIC-INDIC DIGIT THREE}",
        "\N{FULLWIDTH DIGIT ONE}\N{FULLWIDTH DIGIT TWO}",
        "12 %",
        "nan",
    ],
)
def test_parse_number_rejects_other_text(raw: str) -> None:
    assert parse_number(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-03-04", "2026-03-04"),
        ("March 4, 2026", "2026-03-04"),
        ("mar. 4th, 2026", "2026-03-04"),
        ("Sept 30, 2025", "2025-09-30"),
        ("4 March 2026", "2026-03-04"),
        ("4th   march,  2026", "2026-03-04"),
        ("MAY 1, 2024", "2024-05-01"),
    ],
)
def test_parse_date_reads_supported_forms(raw: str, expected: str) -> None:
    assert parse_date(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "2026-02-30",
        "2026-13-01",
        "Smarch 4, 2026",
        "04/03/2026",
        "2026",
        "",
        "March 2026",
        "Mar. 4th 2026",
    ],
)
def test_parse_date_rejects_invalid_or_unsupported(raw: str) -> None:
    assert parse_date(raw) is None


@pytest.mark.parametrize(
    ("number", "expected"),
    [(3.0, "3"), (-0.0, "0"), (2.5, "2.5"), (1e15, "1000000000000000.0"), (1e20, "1e+20")],
)
def test_number_text(number: float, expected: str) -> None:
    assert number_text(number) == expected
