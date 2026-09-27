import asyncio
import json
from typing import Final

import pytest
from agents import Agent, FunctionTool, RunConfig, Runner, function_tool
from agents.testing import ScriptedModel, assistant_message
from agents.testing import function_call as scripted_call

from mimir.core.decisions import Choice
from mimir.core.results import ChoiceResult, Status
from mimir.integrations.openai_agents import as_function_tool, guard
from tests.conftest import RecordingDecider, YesNoDecider

TEXT: Final = "my card was charged twice"
RULE: Final = "Refunds above 500 dollars need a manager."
RUN: Final = RunConfig(tracing_disabled=True)


def _outputs(model: ScriptedModel, call: int) -> list[str]:
    items = model.calls[call].input
    assert isinstance(items, list)
    return [str(item["output"]) for item in items if item.get("type") == "function_call_output"]


def _refund_tool(refunds: list[float]) -> FunctionTool:
    @function_tool
    def issue_refund(amount: float) -> str:
        """Refund the customer."""
        refunds.append(amount)
        return f"refunded {amount}"

    return issue_refund


def test_the_agent_loop_calls_the_decision_tool_and_reads_its_result() -> None:
    decider = RecordingDecider()
    tool = decider.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    model = ScriptedModel(
        [
            [scripted_call("route_ticket", {"context": TEXT}, call_id="call-1")],
            [assistant_message("billing")],
        ]
    )
    agent = Agent(name="support", model=model, tools=[as_function_tool(tool)])
    result = asyncio.run(Runner.run(agent, "Route this ticket.", run_config=RUN))
    assert result.final_output == "billing"
    [output] = _outputs(model, 1)
    parsed = ChoiceResult.model_validate_json(output)
    assert (parsed.answer, parsed.status) == ("billing", Status.DECIDED)
    assert decider.calls[0].requests[0][0].passages[0].text == TEXT
    model.assert_complete()


def test_the_function_tool_carries_the_decision_tool_contract() -> None:
    tool = RecordingDecider().tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    converted = as_function_tool(tool)
    assert converted.name == "route_ticket"
    assert converted.description == tool.agent_description
    assert converted.params_json_schema == tool.input_schema
    assert converted.strict_json_schema is False
    assert converted.output_json_schema is None


@pytest.mark.parametrize("arguments", ["{}", '{"context": 3}', "not json"])
def test_invalid_arguments_go_back_to_the_model(arguments: str) -> None:
    decider = RecordingDecider()
    tool = decider.tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    model = ScriptedModel(
        [[scripted_call("route_ticket", arguments, call_id="c")], [assistant_message("done")]]
    )
    agent = Agent(name="support", model=model, tools=[as_function_tool(tool)])
    asyncio.run(Runner.run(agent, "Route.", run_config=RUN))
    [output] = _outputs(model, 1)
    assert output.startswith("Invalid arguments for route_ticket: ")
    assert decider.calls == []


def _guarded_run(decider: YesNoDecider) -> tuple[list[float], ScriptedModel, Agent[None]]:
    refunds: list[float] = []
    check = decider.tool_call_check(RULE, tools=["issue_refund"])
    model = ScriptedModel(
        [
            [scripted_call("issue_refund", {"amount": 900.0}, call_id="call-1")],
            [assistant_message("done")],
        ]
    )
    agent = Agent(name="refunds", model=model, tools=[guard(_refund_tool(refunds), check)])
    return refunds, model, agent


def test_an_allowed_call_runs_without_approval() -> None:
    decider = YesNoDecider(answer=True)
    refunds, model, agent = _guarded_run(decider)
    result = asyncio.run(Runner.run(agent, "Refund.", run_config=RUN))
    assert (refunds, result.interruptions) == ([900.0], [])
    assert _outputs(model, 1) == ["refunded 900.0"]
    [(context, _)] = decider.calls[0].requests
    assert [(field.key, field.value) for field in context.fields] == [
        ("tool", "issue_refund"),
        ("arguments.amount", 900.0),
    ]


def test_a_denied_call_is_rejected_with_the_reason() -> None:
    refunds, model, agent = _guarded_run(YesNoDecider(answer=False))
    result = asyncio.run(Runner.run(agent, "Refund.", run_config=RUN))
    assert (refunds, result.interruptions, result.final_output) == ([], [], "done")
    assert _outputs(model, 1) == [
        "The call to issue_refund is not allowed under the rules (confidence 0.90)."
    ]


def test_an_escalated_call_waits_for_a_person_then_runs() -> None:
    refunds, _, agent = _guarded_run(YesNoDecider(answer=True, status=Status.DEFERRED))

    async def main() -> str:
        paused = await Runner.run(agent, "Refund.", run_config=RUN)
        [pending] = paused.interruptions
        assert pending.name == "issue_refund"
        assert json.loads(pending.arguments or "{}") == {"amount": 900.0}
        assert refunds == []
        state = paused.to_state()
        state.approve(pending)
        resumed = await Runner.run(agent, state, run_config=RUN)
        return str(resumed.final_output)

    assert asyncio.run(main()) == "done"
    assert refunds == [900.0]


def test_guard_keeps_the_original_tool_unchanged() -> None:
    original = _refund_tool([])
    guarded = guard(original, YesNoDecider(answer=True).tool_call_check(RULE))
    assert original.needs_approval is False
    assert original.tool_input_guardrails is None
    assert callable(guarded.needs_approval)
    assert [item.get_name() for item in guarded.tool_input_guardrails or []] == ["tool_call_check"]
