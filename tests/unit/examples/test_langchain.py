from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from examples.langchain import agent, guarded_agent
from mimir.core.results import ChoiceResult, Status
from tests.conftest import RecordingDecider, YesNoDecider


class ScriptedChatModel(GenericFakeChatModel):
    """A fake chat model that accepts tools and answers with its scripted messages."""

    def bind_tools(self, *_: object, **__: object) -> "ScriptedChatModel":
        return self


def _script(name: str, args: dict[str, Any]) -> ScriptedChatModel:
    call = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}])
    replies: Iterator[AIMessage | str] = iter([call, AIMessage(content="done")])
    return ScriptedChatModel(messages=replies)


def test_the_support_agent_routes_the_ticket() -> None:
    decider = RecordingDecider()
    built = agent.build_agent(decider, _script("route_ticket", {"context": "charged twice"}))
    state = built.invoke({"messages": [{"role": "user", "content": "Help."}]})
    [message] = [item for item in state["messages"] if isinstance(item, ToolMessage)]
    assert isinstance(message.artifact, ChoiceResult)
    assert decider.calls[0].requests[0][1].option_ids == tuple(agent.TEAMS)


def _refund_tool(refunds: list[float]) -> BaseTool:
    @tool
    def issue_refund(order: str, amount: float) -> str:
        """Refund `amount` dollars on `order`."""
        refunds.append(amount)
        return f"Refunded {amount:.2f} dollars on order {order}."

    return issue_refund


@pytest.mark.parametrize(("answer", "issued"), [("y", [900.0]), ("n", [])])
def test_the_refunds_agent_escalates_to_the_console(
    monkeypatch: pytest.MonkeyPatch, answer: str, issued: list[float]
) -> None:
    refunds: list[float] = []
    decider = YesNoDecider(answer=True, status=Status.DEFERRED)
    model = _script("issue_refund", {"order": "4412", "amount": 900.0})
    built = guarded_agent.build_agent(decider, _refund_tool(refunds), model)
    asked: list[str] = []

    def reply(prompt: str) -> str:
        asked.append(prompt)
        return answer

    monkeypatch.setattr("builtins.input", reply)
    state = guarded_agent.run_with_approvals(built, "Refund 900 on 4412.", thread="t")
    assert (refunds, state["messages"][-1].content) == (issued, "done")
    [prompt] = asked
    assert prompt.endswith("Approve issue_refund {'order': '4412', 'amount': 900.0}? [y/N] ")
    assert "needs a person's approval" in prompt
