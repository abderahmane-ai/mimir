from pathlib import Path

import pytest

from mimir.cli.toolfile import MAX_FILE_BYTES, ToolFileError, read_tools
from mimir.core.decisions import Choice, YesNo

VALID = """\
tools:
  - name: route_ticket
    description: Route a support ticket to the team that owns it.
    decision:
      type: choice
      question: Which team should handle this ticket?
      options:
        billing: "Billing: payments, refunds and invoices"
        security: "Security: account access and fraud"
  - name: is_late
    description: Check whether the invoice is late.
    decision: {type: yes_no, question: "Is the invoice late?"}
    risk: 0.05
    alpha: 0.2
"""


def _write(tmp_path: Path, text: str | bytes) -> Path:
    path = tmp_path / "tools.yaml"
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def test_a_tools_file_reads_in_order(tmp_path: Path) -> None:
    definitions = read_tools(_write(tmp_path, VALID))
    route, late = definitions.tools
    assert route.name == "route_ticket"
    assert route.decision == Choice(
        "Which team should handle this ticket?",
        {
            "billing": "Billing: payments, refunds and invoices",
            "security": "Security: account access and fraud",
        },
    )
    assert (route.risk, route.alpha) == (0.05, None)
    assert (late.decision, late.risk, late.alpha) == (YesNo("Is the invoice late?"), 0.05, 0.2)


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_line_endings_and_a_byte_order_mark_are_read(tmp_path: Path, newline: str) -> None:
    text = "﻿" + VALID.replace("\n", newline)
    assert len(read_tools(_write(tmp_path, text)).tools) == 2


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "a: &x [1, 2]\nb: [*x, *x, *x]\n",
            "aliases are not allowed in a tools file",
        ),
        (
            VALID.replace("    risk: 0.05\n", "    risk: 0.05\n    risk: 0.02\n"),
            r"keys \['risk'\] are repeated",
        ),
        ("tools: !!python/object/apply:os.system [echo]\n", "could not determine a constructor"),
        ("tools: [\n", "expected the node content"),
        ("- one\n- two\n", "Input should be a valid dictionary"),
        ("", "Input should be a valid dictionary"),
        (
            VALID.replace("type: yes_no", "type: guess"),
            r"tools\.1\.decision\n  Input tag 'guess' found",
        ),
        (VALID.replace("name: is_late", "name: is late"), "tools.1.name"),
    ],
)
def test_unsafe_or_invalid_files_are_refused_by_name(
    tmp_path: Path, text: str, message: str
) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(ToolFileError, match=message) as raised:
        read_tools(path)
    assert str(raised.value).startswith(f"{path}: ")


def test_a_billion_laughs_file_is_refused_before_it_expands(tmp_path: Path) -> None:
    levels = ['a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]']
    levels += [
        f"{chr(98 + level)}: &{chr(98 + level)} [{', '.join(['*' + chr(97 + level)] * 9)}]"
        for level in range(8)
    ]
    with pytest.raises(ToolFileError, match="aliases are not allowed"):
        read_tools(_write(tmp_path, "\n".join(levels) + "\n"))


def test_unreadable_files_name_the_path(tmp_path: Path) -> None:
    with pytest.raises(ToolFileError, match="No such file"):
        read_tools(tmp_path / "missing.yaml")
    with pytest.raises(ToolFileError, match="utf-8"):
        read_tools(_write(tmp_path, b"tools: \xff\xfe\n"))
    with pytest.raises(ToolFileError, match=f"a tools file holds at most {MAX_FILE_BYTES}"):
        read_tools(_write(tmp_path, "#" * (MAX_FILE_BYTES + 1)))
