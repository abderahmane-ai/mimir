import asyncio

import pytest
from pydantic import ValidationError

from mimir.core.decisions import Choice, Rate, YesNo
from mimir.core.results import ChoiceResult
from mimir.core.tools import GUIDANCE, ToolArguments, ToolDefinition, ToolDefinitions
from tests.conftest import RecordingDecider

ROUTE = {
    "name": "route_ticket",
    "description": "Route a ticket.",
    "decision": {"type": "choice", "question": "Which team?", "options": ["billing", "security"]},
}


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


@pytest.mark.parametrize(
    "name", ["", "has space", "x" * 65, "dots.are.bad", "ünïcode", "trailing\n"]
)
def test_tool_names_must_be_portable(name: str) -> None:
    with pytest.raises(ValueError, match="must match"):
        RecordingDecider().tool(name, Choice("q", ["a", "b"]), "desc")


def test_tool_description_must_not_be_blank() -> None:
    with pytest.raises(ValueError, match="blank description"):
        RecordingDecider().tool("ok", Choice("q", ["a", "b"]), "  ")


def test_call_with_validates_the_arguments_then_decides() -> None:
    decider = RecordingDecider()
    tool = decider.tool("route_ticket", Choice("Which team?", ["billing", "security"]), "Route.")
    assert isinstance(tool.call_with({"context": "charged twice"}), ChoiceResult)
    state = {"context": {"state": {"customer": {"plan": "enterprise"}}}}
    assert isinstance(asyncio.run(tool.acall_with(state)), ChoiceResult)
    first, second = (call.requests[0][0] for call in decider.calls)
    assert first.passages[0].text == "charged twice"
    assert second.fields[0].key == "customer.plan"
    for arguments in ({}, {"context": 3}, {"context": "x", "extra": 1}):
        with pytest.raises(ValidationError):
            tool.call_with(arguments)
    assert len(decider.calls) == 2


def test_agent_description_ends_with_the_status_guidance() -> None:
    tool = RecordingDecider().tool("is_late", YesNo("Is it late?"), "Check lateness.")
    assert tool.agent_description == f"Check lateness.\n\n{GUIDANCE}"
    assert "`deferred`" in GUIDANCE


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


def test_definitions_bind_in_order_with_their_risk_and_alpha() -> None:
    definitions = ToolDefinitions.model_validate(
        {
            "tools": [
                ROUTE,
                {
                    "name": "is_late",
                    "description": "Check lateness.",
                    "decision": {"type": "yes_no", "question": "Is it late?"},
                    "risk": 0.05,
                    "alpha": 0.2,
                },
            ]
        }
    )
    decider = RecordingDecider()
    tools = definitions.bind(decider)
    assert [tool.name for tool in tools] == ["route_ticket", "is_late"]
    assert tools[0].spec == Choice("Which team?", ["billing", "security"])
    assert (tools[0].risk, tools[0].alpha) == (0.01, None)
    assert (tools[1].spec, tools[1].risk, tools[1].alpha) == (YesNo("Is it late?"), 0.05, 0.2)
    assert all(tool.decider is decider for tool in tools)


def test_definitions_reject_repeated_names_empty_lists_and_unknown_keys() -> None:
    with pytest.raises(ValidationError, match=r"\['route_ticket'\] are declared more than once"):
        ToolDefinitions.model_validate({"tools": [ROUTE, ROUTE]})
    with pytest.raises(ValidationError, match="at least 1 item"):
        ToolDefinitions.model_validate({"tools": []})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ToolDefinition.model_validate({**ROUTE, "question": "stray"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "has space", "should match pattern"),
        ("name", "trailing\n", "should match pattern"),
        ("description", "   ", "at least 1 character"),
        ("alpha", 1.0, "less than 1"),
        ("risk", float("nan"), "finite number"),
    ],
)
def test_definition_fields_are_validated(field: str, value: object, message: str) -> None:
    with pytest.raises(ValidationError, match=message) as raised:
        ToolDefinition.model_validate({**ROUTE, field: value})
    assert raised.value.errors()[0]["loc"] == (field,)
