import pytest
from crewai.tools import BaseTool, tool

from examples.crewai import agent, guarded_agent
from mimir.core.results import ChoiceResult, Status
from tests.conftest import RecordingDecider, YesNoDecider
from tests.unit.mimir.integrations.test_crewai import FINAL, ScriptedLLM, _action


@pytest.fixture(autouse=True)
def _no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")


def test_the_support_crew_routes_the_ticket() -> None:
    decider = RecordingDecider()
    llm = ScriptedLLM(model="scripted", replies=[_action("route_ticket", {"context": "x"}), FINAL])
    assert str(agent.build_crew(decider, llm).kickoff(inputs={"ticket": "x"})) == "done"
    assert ChoiceResult.model_validate_json(llm.observation(1)).answer == "billing"
    assert decider.calls[0].requests[0][1].option_ids == tuple(agent.TEAMS)


@pytest.mark.parametrize(("answer", "issued"), [("y", [900.0]), ("n", [])])
def test_the_refunds_crew_asks_the_console_about_escalations(
    monkeypatch: pytest.MonkeyPatch, answer: str, issued: list[float]
) -> None:
    refunds: list[float] = []

    @tool("issue_refund")
    def issue_refund(order: str, amount: float) -> str:
        """Refund `amount` dollars on `order`."""
        refunds.append(amount)
        return f"Refunded {amount:.2f} dollars on order {order}."

    refund: BaseTool = issue_refund
    call = _action("issue_refund", {"order": "4412", "amount": 900.0})
    crew = guarded_agent.build_crew(refund, ScriptedLLM(model="scripted", replies=[call, FINAL]))
    monkeypatch.setattr("builtins.input", lambda: answer)
    decider = YesNoDecider(answer=True, status=Status.DEFERRED)
    assert guarded_agent.run_checked(crew, decider, "Refund 900 on 4412.") == "done"
    assert refunds == issued
