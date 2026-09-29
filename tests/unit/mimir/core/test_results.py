import pytest
from pydantic import TypeAdapter

from mimir.core.decisions import (
    SPEC_TYPES,
    Choice,
    DecisionSpec,
    Estimate,
    MultiChoice,
    Rank,
    Rate,
    Verify,
    YesNo,
)
from mimir.core.results import (
    RESULT_FOR_SPEC,
    DecisionResult,
    OptionSet,
    Status,
)
from tests.conftest import result_for

RESULTS = TypeAdapter[DecisionResult](DecisionResult)
SPECS: list[DecisionSpec] = [
    Choice("q", ["a", "b", "c"]),
    MultiChoice("q", ["a", "b"]),
    YesNo("q"),
    Verify("claim"),
    Rank("q", ["a", "b"]),
    Rate("q", ["l", "m", "h"]),
    Estimate("q", 0.0, 1.0),
]


def test_every_spec_type_has_exactly_one_result_type() -> None:
    spec_types = {spec_type.model_fields["type"].default for spec_type in SPEC_TYPES}
    assert set(RESULT_FOR_SPEC) == spec_types
    for name, result_type in RESULT_FOR_SPEC.items():
        assert result_type.model_fields["type"].default == name


@pytest.mark.parametrize("spec", SPECS)
def test_results_round_trip_through_the_union(spec: DecisionSpec) -> None:
    result = result_for(spec)
    parsed = RESULTS.validate_json(RESULTS.dump_json(result))
    assert parsed == result
    assert type(parsed) is RESULT_FOR_SPEC[spec.type]
    assert parsed.schema_version == 1


def test_status_serialises_as_lowercase_strings() -> None:
    assert [status.value for status in Status] == ["decided", "abstained", "deferred"]
    result = result_for(YesNo("q"), Status.DEFERRED)
    assert result.model_dump(mode="json")["status"] == "deferred"
    assert result.model_dump(mode="json")["actionable"] is False
    assert "deferral" not in result.model_dump(mode="json")


def test_empty_estimate_interval_is_distinct_from_none() -> None:
    result = result_for(Estimate("q", 0.0, 1.0))
    empty = result.model_copy(update={"interval": ()})
    assert RESULTS.validate_json(RESULTS.dump_json(empty)).model_dump()["interval"] == ()
    missing = result.model_copy(update={"interval": None})
    assert RESULTS.validate_json(RESULTS.dump_json(missing)).model_dump()["interval"] is None


def test_option_set_records_abstain_separately() -> None:
    assert OptionSet(options=("a",), abstain=True).model_dump() == {
        "options": ("a",),
        "abstain": True,
    }
