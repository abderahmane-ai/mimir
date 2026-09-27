import asyncio
from typing import Final

import pytest
from agents import FunctionTool, RunConfig, Runner, function_tool
from agents.testing import ScriptedModel, assistant_message
from agents.testing import function_call as scripted_call

from examples.openai_agents import agent, guarded_agent
from mimir.core.results import Status
from tests.conftest import RecordingDecider, YesNoDecider

RUN: Final = RunConfig(tracing_disabled=True)


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")


def test_the_support_agent_routes_the_ticket() -> None:
    decider = RecordingDecider()
    model = ScriptedModel(
        [
            [scripted_call("route_ticket", {"context": "charged twice"}, call_id="c")],
            [assistant_message("Billing will answer.")],
        ]
    )
    result = asyncio.run(Runner.run(agent.build_agent(decider, model), "Help.", run_config=RUN))
    assert result.final_output == "Billing will answer."
    [(context, spec)] = decider.calls[0].requests
    assert context.passages[0].text == "charged twice"
    assert spec.option_ids == tuple(agent.TEAMS)


def _refund_tool(refunds: list[float]) -> FunctionTool:
    @function_tool
    def issue_refund(order: str, amount: float) -> str:
        """Refund `amount` dollars on `order`."""
        refunds.append(amount)
        return f"Refunded {amount:.2f} dollars on order {order}."

    return issue_refund


def _refund_model() -> ScriptedModel:
    call = scripted_call("issue_refund", {"order": "4412", "amount": 900.0}, call_id="c")
    return ScriptedModel([[call], [assistant_message("done")]])


def test_the_refunds_agent_escalates_to_the_console(monkeypatch: pytest.MonkeyPatch) -> None:
    refunds: list[float] = []
    decider = YesNoDecider(answer=True, status=Status.DEFERRED)
    built = guarded_agent.build_agent(decider, _refund_tool(refunds), _refund_model())
    asked: list[str] = []

    def approve(prompt: str) -> str:
        asked.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", approve)
    result = asyncio.run(guarded_agent.run_with_approvals(built, "Refund 900 on 4412."))
    assert (refunds, result.final_output) == ([900.0], "done")
    assert asked == ['Approve issue_refund {"order":"4412","amount":900.0}? [y/N] ']
    [(context, _)] = decider.calls[0].requests
    assert tuple(passage.text for passage in context.passages) == guarded_agent.RULES


def test_the_refunds_agent_runs_certified_refunds_without_asking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refunds: list[float] = []
    built = guarded_agent.build_agent(
        YesNoDecider(answer=True), _refund_tool(refunds), _refund_model()
    )

    def refuse(prompt: str) -> str:
        pytest.fail(f"asked {prompt!r} for a certified refund")

    monkeypatch.setattr("builtins.input", refuse)
    result = asyncio.run(guarded_agent.run_with_approvals(built, "Refund 900 on 4412."))
    assert (refunds, result.final_output) == ([900.0], "done")
