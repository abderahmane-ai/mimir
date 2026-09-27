import pytest
from pydantic import TypeAdapter, ValidationError

from mimir.core.decisions import (
    VERDICT_TEXTS,
    VERDICTS,
    Choice,
    DecisionSpec,
    Estimate,
    MultiChoice,
    Rank,
    Rate,
    Verify,
    YesNo,
)

SPECS = TypeAdapter[DecisionSpec](DecisionSpec)


def test_choice_from_texts_uses_texts_as_ids() -> None:
    spec = Choice("Which team?", ["billing", "security", "shipping"])
    assert spec.options == {"billing": "billing", "security": "security", "shipping": "shipping"}
    assert spec.option_ids == ("billing", "security", "shipping")
    assert spec.model_type == "categorical"


def test_choice_with_two_options_is_binary() -> None:
    assert Choice("q", {"a": "Alpha", "b": "Beta"}).model_type == "binary"


def test_mapping_options_keep_ids_and_texts_in_order() -> None:
    spec = Rank("q", {"z": "last", "a": "first", "m": "middle"})
    assert spec.option_ids == ("z", "a", "m")
    assert spec.option_texts == ("last", "first", "middle")


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: Choice("q", ["a", "a"]), "duplicate options"),
        (lambda: Choice("q", ["a"]), "at least 2"),
        (lambda: Choice("q", []), "at least 2"),
        (lambda: Choice("q", {"a": "same", "b": "same"}), "duplicate texts"),
        (lambda: Choice("q", {" ": "x", "b": "y"}), "blank ids"),
        (lambda: Choice("q", {"a": " ", "b": "y"}), "blank texts"),
        (lambda: Choice(" \t", ["a", "b"]), "blank"),
        (lambda: MultiChoice("q", ["a"]), "at least 2"),
        (lambda: Rank("q", ["a"]), "at least 2"),
        (lambda: Rate("q", ["low", "high"]), "at least 3"),
        (lambda: Estimate("q", 1.0, 1.0), "below high"),
        (lambda: Estimate("q", 2.0, 1.0), "below high"),
        (lambda: Estimate("q", 0.0, float("inf")), "finite|below"),
        (lambda: Verify(""), "at least 1 character"),
    ],
)
def test_invalid_specs_raise_naming_the_problem(build: object, match: str) -> None:
    assert callable(build)
    with pytest.raises(ValidationError, match=match):
        build()


def test_yes_no_and_verify_use_fixed_phrasings() -> None:
    yes_no = YesNo("Is the invoice late?")
    assert (yes_no.model_type, yes_no.option_ids, yes_no.option_texts) == (
        "binary",
        ("no", "yes"),
        ("no", "yes"),
    )
    verify = Verify("The invoice is paid.")
    assert (
        verify.model_question == "Judge this statement against the evidence: The invoice is paid."
    )
    assert verify.option_ids == VERDICTS
    assert verify.option_texts == VERDICT_TEXTS
    assert verify.model_type == "categorical"


@pytest.mark.parametrize(
    "spec",
    [
        Choice("q", ["a", "b", "c"]),
        MultiChoice("q", {"x": "X", "y": "Y"}),
        YesNo("q"),
        Verify("claim"),
        Rank("q", ["a", "b"]),
        Rate("q", ["low", "medium", "high"]),
        Estimate("q", 0.0, 10.0, "kg"),
    ],
)
def test_specs_round_trip_through_json(spec: DecisionSpec) -> None:
    assert SPECS.validate_json(SPECS.dump_json(spec)) == spec


def test_union_rejects_unknown_type_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SPECS.validate_python({"type": "guess", "question": "q"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        SPECS.validate_python({"type": "yes_no", "question": "q", "options": ["a", "b"]})


def test_estimate_unit_and_types() -> None:
    spec = Estimate("How heavy?", 0, 100, unit="kg")
    assert (spec.low, spec.high, spec.unit, spec.model_type) == (0.0, 100.0, "kg", "continuous")
    assert spec.option_ids == ()
