import pytest
from agent_framework import FunctionTool

from examples.agent_framework import agent, guarded_agent
from mimir.core.results import ChoiceResult, Status
from tests.conftest import RecordingDecider, YesNoDecider
from tests.unit.mimir.integrations.test_agent_framework import ScriptedChatClient, _run


def test_the_support_agent_routes_the_ticket() -> None:
    decider = RecordingDecider()
    client = ScriptedChatClient("route_ticket", {"context": "charged twice"})
    response = _run(agent.build_agent(decider, client))
    assert ChoiceResult.model_validate_json(response.text).answer == "billing"
    assert decider.calls[0].requests[0][1].option_ids == tuple(agent.TEAMS)


@pytest.mark.parametrize(
    ("answer", "status", "issued"),
    [(True, Status.DECIDED, [900.0]), (True, Status.DEFERRED, []), (False, Status.DECIDED, [])],
)
def test_the_refunds_agent_runs_only_certified_refunds(
    answer: bool, status: Status, issued: list[float]
) -> None:
    refunds: list[float] = []

    def issue_refund(order: str, amount: float) -> str:
        refunds.append(amount)
        return f"Refunded {amount:.2f} dollars on order {order}."

    tool = FunctionTool(name="issue_refund", description="Refund.", func=issue_refund)
    client = ScriptedChatClient("issue_refund", {"order": "4412", "amount": 900.0})
    built = guarded_agent.build_agent(YesNoDecider(answer=answer, status=status), client, tool)
    _run(built)
    assert refunds == issued
