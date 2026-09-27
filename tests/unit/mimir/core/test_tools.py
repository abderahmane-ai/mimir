import asyncio

import pytest

from mimir.core.decisions import Choice, Rate
from mimir.core.results import ChoiceResult
from mimir.core.tools import ToolArguments
from tests.conftest import RecordingDecider


def test_tool_call_uses_the_bound_spec_risk_and_alpha() -> None:
    decider = RecordingDecider()
    spec = Choice("Which team?", ["billing", "security"])
    tool = decider.tool("route_ticket", spec, "Route a ticket.", risk=0.02, alpha=0.3)
    result = tool("My card was charged twice")
    assert isinstance(result, ChoiceResult)
    call = decider.calls[0]
    assert call.requests[0][1] == spec
    assert (call.risk, call.alpha) == (0.02, 0.3)
    asyncio.run(tool.acall("again"))
    assert len(decider.calls) == 2


@pytest.mark.parametrize("name", ["", "has space", "x" * 65, "dots.are.bad", "ünïcode"])
def test_tool_names_must_be_portable(name: str) -> None:
    with pytest.raises(ValueError, match="must match"):
        RecordingDecider().tool(name, Choice("q", ["a", "b"]), "desc")


def test_tool_description_must_not_be_blank() -> None:
    with pytest.raises(ValueError, match="blank description"):
        RecordingDecider().tool("ok", Choice("q", ["a", "b"]), "  ")


def test_tool_schemas() -> None:
    tool = RecordingDecider().tool("rate_urgency", Rate("q", ["l", "m", "h"]), "Rate urgency.")
    assert tool.input_schema == ToolArguments.model_json_schema()
    assert tool.input_schema["required"] == ["context"]
    assert tool.input_schema["additionalProperties"] is False
    output = tool.output_schema
    assert output["title"] == "RateResult"
    properties = output["properties"]
    assert isinstance(properties, dict)
    assert "prediction_set" in properties
