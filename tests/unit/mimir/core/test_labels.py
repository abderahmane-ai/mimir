import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mimir.core.decisions import Choice, Estimate, MultiChoice, Rank, Rate, Verify, YesNo
from mimir.core.labels import LabelError, LabelledDecision, is_correct, read_labelled
from tests.conftest import result_for


@pytest.mark.parametrize(
    ("decision", "label"),
    [
        (Choice("q", ["a", "b"]), "a"),
        (Choice("q", ["a", "b"]), None),
        (MultiChoice("q", ["a", "b"]), []),
        (MultiChoice("q", ["a", "b"]), ["b", "a"]),
        (YesNo("q"), True),
        (Verify("c"), "not_enough_information"),
        (Rank("q", ["a", "b"]), "b"),
        (Rank("q", ["a", "b"]), ["a", "b"]),
        (Rate("q", ["l", "m", "h"]), "m"),
        (Estimate("q", 0, 10), 10),
    ],
)
def test_valid_labels(decision: object, label: object) -> None:
    item = LabelledDecision.model_validate({"context": "x", "decision": decision, "label": label})
    assert item.label == label


@pytest.mark.parametrize(
    ("decision", "label"),
    [
        (Choice("q", ["a", "b"]), "c"),
        (Choice("q", ["a", "b"]), True),
        (MultiChoice("q", ["a", "b"]), ["c"]),
        (YesNo("q"), "yes"),
        (Verify("c"), "maybe"),
        (Rank("q", ["a", "b"]), []),
        (Rate("q", ["l", "m", "h"]), None),
        (Estimate("q", 0, 10), 11),
        (Estimate("q", 0, 10), True),
    ],
)
def test_labels_that_do_not_fit_the_spec_are_rejected(decision: object, label: object) -> None:
    with pytest.raises(ValidationError, match="does not fit"):
        LabelledDecision.model_validate({"context": "x", "decision": decision, "label": label})


def test_is_correct_per_type() -> None:
    choice = result_for(Choice("q", ["a", "b"]))
    assert is_correct(choice, "a")
    assert not is_correct(choice, None)
    assert is_correct(result_for(MultiChoice("q", ["a", "b"])), ["a"])
    assert not is_correct(result_for(MultiChoice("q", ["a", "b"])), ["a", "b"])
    rank = result_for(Rank("q", ["a", "b"]))
    assert is_correct(rank, "a")
    assert is_correct(rank, ["b", "a"])
    assert not is_correct(rank, "b")
    estimate = result_for(Estimate("q", 0, 10))
    assert is_correct(estimate, 3.0)
    assert not is_correct(estimate.model_copy(update={"interval": ()}), 3.0)
    assert not is_correct(estimate.model_copy(update={"interval": (4.0, 5.0)}), 3.0)


def _line(label: object) -> str:
    return json.dumps(
        {"context": "x", "decision": {"type": "yes_no", "question": "q"}, "label": label}
    )


def test_read_labelled_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(_line(True) + "\n\n" + _line(False) + "\r\n", encoding="utf-8")
    assert [item.label for item in read_labelled(path)] == [True, False]


def test_read_labelled_names_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(_line(True) + "\n" + _line("maybe") + "\n", encoding="utf-8")
    with pytest.raises(LabelError, match=r"labels\.jsonl:2"):
        read_labelled(path)


@pytest.mark.parametrize("content", ["", "\n\n", "{not json}\n"])
def test_read_labelled_rejects_empty_and_malformed_files(tmp_path: Path, content: str) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(LabelError):
        read_labelled(path)
