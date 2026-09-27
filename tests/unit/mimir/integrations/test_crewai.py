import json
from collections.abc import Iterator
from typing import Any, Final

import pytest
from crewai import Agent, Crew, Task
from crewai.hooks import (
    ToolCallHookContext,
    register_before_tool_call_hook,
    unregister_before_tool_call_hook,
)
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool, tool
from pydantic import PrivateAttr

from mimir.core.checks import CheckOutcome, Permission
from mimir.core.decisions import Choice
from mimir.core.results import ChoiceResult, Status
from mimir.integrations.crewai import ToolCallHook, as_crewai_tool, tool_call_hook
from tests.conftest import RecordingDecider, YesNoDecider

TEXT: Final = "my card was charged twice"
RULE: Final = "Refunds above 500 dollars need a manager."


@pytest.fixture(autouse=True)
def _no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")


class ScriptedLLM(BaseLLM):
    """An LLM answering with its scripted replies, in CrewAI's ReAct text format."""

    replies: list[str]
    _prompts: list[str] = PrivateAttr(default_factory=list)

    def call(self, messages: object, *_: object, **__: object) -> str:
        self._prompts.append(json.dumps(messages, default=str))
        return self.replies[len(self._prompts) - 1]

    @property
    def prompts(self) -> list[str]:
        return self._prompts

    def observation(self, call: int) -> str:
        """The tool result the model read before reply `call`."""
        messages = json.loads(self._prompts[call])
        [shown] = [
            str(message["content"]).split("Observation: ", 1)[1]
            for message in messages
            if message["role"] == "assistant" and "Observation: " in str(message["content"])
        ]
        return shown


def _action(name: str, arguments: dict[str, Any]) -> str:
    return f"Thought: I should call {name}.\nAction: {name}\nAction Input: {json.dumps(arguments)}"


FINAL: Final = "Thought: I now know the final answer\nFinal Answer: done"


def _run(llm: ScriptedLLM, tools: list[BaseTool]) -> str:
    agent = Agent(
        role="Support", goal="Resolve tickets.", backstory="A support agent.", llm=llm, tools=tools
    )
    task = Task(description="Handle the ticket.", expected_output="done", agent=agent)
    return str(Crew(agents=[agent], tasks=[task]).kickoff())


def test_the_agent_loop_calls_the_decision_tool_and_reads_its_result() -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("which team", ["billing", "security"]), "Route.")
    llm = ScriptedLLM(model="scripted", replies=[_action("route_ticket", {"context": TEXT}), FINAL])
    assert _run(llm, [as_crewai_tool(route)]) == "done"
    assert decider.calls[0].requests[0][0].passages[0].text == TEXT
    observed = ChoiceResult.model_validate_json(llm.observation(1))
    assert (observed.answer, observed.status) == ("billing", Status.DECIDED)


def test_the_tool_carries_the_decision_tool_contract() -> None:
    route = RecordingDecider().tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    converted = as_crewai_tool(route)
    assert converted.name == "route_ticket"
    assert route.agent_description in converted.description
    assert converted.result_schema is ChoiceResult
    result = converted.run(context=TEXT)
    assert isinstance(result, ChoiceResult)
    assert result.status is Status.DECIDED


def test_invalid_arguments_go_back_to_the_model() -> None:
    decider = RecordingDecider()
    route = decider.tool("route_ticket", Choice("q", ["a", "b"]), "Route.")
    llm = ScriptedLLM(model="scripted", replies=[_action("route_ticket", {"context": 3}), FINAL])
    assert _run(llm, [as_crewai_tool(route)]) == "done"
    assert "validation error" in llm.observation(1).lower()
    assert decider.calls == []


@pytest.fixture
def registered() -> Iterator[list[ToolCallHook]]:
    hooks: list[ToolCallHook] = []
    yield hooks
    for hook in hooks:
        unregister_before_tool_call_hook(hook)


def _refunds(refunds: list[float]) -> BaseTool:
    @tool("issue_refund")
    def issue_refund(amount: float) -> str:
        """Refund the customer."""
        refunds.append(amount)
        return f"refunded {amount}"

    return issue_refund


def _checked_run(
    hook: ToolCallHook, registered: list[ToolCallHook], refunds: list[float]
) -> ScriptedLLM:
    register_before_tool_call_hook(hook)
    registered.append(hook)
    llm = ScriptedLLM(model="scripted", replies=[_action("issue_refund", {"amount": 900.0}), FINAL])
    assert _run(llm, [_refunds(refunds)]) == "done"
    return llm


def test_an_allowed_call_runs(registered: list[ToolCallHook]) -> None:
    refunds: list[float] = []
    decider = YesNoDecider(answer=True)
    check = decider.tool_call_check(RULE, tools=["issue_refund"])
    _checked_run(tool_call_hook(check), registered, refunds)
    assert refunds == [900.0]
    [(context, _)] = decider.calls[0].requests
    assert [(field.key, field.value) for field in context.fields] == [
        ("tool", "issue_refund"),
        ("arguments.amount", 900.0),
    ]


def test_a_denied_call_is_blocked(registered: list[ToolCallHook]) -> None:
    refunds: list[float] = []
    check = YesNoDecider(answer=False).tool_call_check(RULE, tools=["issue_refund"])
    llm = _checked_run(tool_call_hook(check), registered, refunds)
    assert refunds == []
    assert llm.observation(1).startswith("Tool execution blocked by hook. Tool: issue_refund")


@pytest.mark.parametrize("approved", [True, False])
def test_an_escalated_call_goes_to_the_approver(
    registered: list[ToolCallHook], approved: bool
) -> None:
    refunds: list[float] = []
    seen: list[CheckOutcome] = []

    def approve(_: ToolCallHookContext, outcome: CheckOutcome) -> bool:
        seen.append(outcome)
        return approved

    decider = YesNoDecider(answer=True, status=Status.DEFERRED)
    check = decider.tool_call_check(RULE, tools=["issue_refund"])
    _checked_run(tool_call_hook(check, approve=approve), registered, refunds)
    assert refunds == ([900.0] if approved else [])
    [outcome] = seen
    assert (outcome.tool, outcome.permission) == ("issue_refund", Permission.ESCALATE)


def test_an_escalated_call_without_an_approver_is_blocked(registered: list[ToolCallHook]) -> None:
    refunds: list[float] = []
    check = YesNoDecider(answer=None, status=Status.ABSTAINED).tool_call_check(RULE)
    _checked_run(tool_call_hook(check), registered, refunds)
    assert refunds == []
