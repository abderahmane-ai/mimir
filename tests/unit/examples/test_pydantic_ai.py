import asyncio

import pytest
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from examples.pydantic_ai import agent, guarded_agent
from mimir.core.results import ChoiceResult, Status
from tests.conftest import RecordingDecider, YesNoDecider
from tests.unit.mimir.integrations.test_pydantic_ai import _returns, _script


def test_the_support_agent_routes_the_ticket() -> None:
    decider = RecordingDecider()
    model = FunctionModel(_script("route_ticket", {"context": "charged twice"}))
    result = asyncio.run(agent.build_agent(decider, model).run("Help."))
    [returned] = _returns(result.all_messages())
    assert isinstance(returned, ToolReturnPart)
    assert isinstance(returned.content, ChoiceResult)
    assert decider.calls[0].requests[0][1].option_ids == tuple(agent.TEAMS)


@pytest.mark.parametrize(("answer", "issued"), [("y", [900.0]), ("n", [])])
def test_the_refunds_agent_escalates_to_the_console(
    monkeypatch: pytest.MonkeyPatch, answer: str, issued: list[float]
) -> None:
    refunds: list[float] = []

    def issue_refund(order: str, amount: float) -> str:
        """Refund `amount` dollars on `order`."""
        refunds.append(amount)
        return f"Refunded {amount:.2f} dollars on order {order}."

    decider = YesNoDecider(answer=True, status=Status.DEFERRED)
    model = FunctionModel(_script("issue_refund", {"order": "4412", "amount": 900.0}))
    built = guarded_agent.build_agent(decider, FunctionToolset([issue_refund]), model)
    asked: list[str] = []

    def reply(prompt: str) -> str:
        asked.append(prompt)
        return answer

    monkeypatch.setattr("builtins.input", reply)
    asyncio.run(guarded_agent.run_with_approvals(built, "Refund 900 on 4412."))
    assert refunds == issued
    [prompt] = asked
    assert "needs a person's approval" in prompt
    assert prompt.endswith("Approve issue_refund {'order': '4412', 'amount': 900.0}? [y/N] ")
