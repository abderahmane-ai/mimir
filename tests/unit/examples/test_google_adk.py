import asyncio

import pytest
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from examples.google_adk import agent, guarded_agent
from mimir.core.results import Status
from tests.conftest import RecordingDecider, YesNoDecider
from tests.unit.mimir.integrations.test_google_adk import ScriptedLlm, _call


def test_the_support_agent_routes_the_ticket() -> None:
    decider = RecordingDecider()
    llm = ScriptedLlm(
        model="scripted",
        replies=[_call("route_ticket", {"context": "charged twice"}), types.Part(text="Billing.")],
    )
    assert asyncio.run(agent.ask(agent.build_agent(decider, llm), "Help.")) == "Billing."
    assert decider.calls[0].requests[0][1].option_ids == tuple(agent.TEAMS)


@pytest.mark.parametrize(("answer", "issued"), [("y", [900.0]), ("n", [])])
def test_the_refunds_agent_asks_the_console_to_confirm(
    monkeypatch: pytest.MonkeyPatch, answer: str, issued: list[float]
) -> None:
    refunds: list[float] = []

    def issue_refund(order: str, amount: float) -> str:
        """Refund `amount` dollars on `order`."""
        refunds.append(amount)
        return f"Refunded {amount:.2f} dollars on order {order}."

    llm = ScriptedLlm(
        model="scripted",
        replies=[
            _call("issue_refund", {"order": "4412", "amount": 900.0}),
            types.Part(text="Done."),
        ],
    )
    decider = YesNoDecider(answer=True, status=Status.DEFERRED)
    built = guarded_agent.build_agent(decider, FunctionTool(issue_refund), llm)
    asked: list[str] = []

    def reply(prompt: str) -> str:
        asked.append(prompt)
        return answer

    monkeypatch.setattr("builtins.input", reply)
    reply_text = asyncio.run(guarded_agent.run_with_approvals(built, "Refund 900 on 4412."))
    assert refunds == issued
    assert reply_text == "Done."
    [prompt] = asked
    assert "needs a person's approval" in prompt
