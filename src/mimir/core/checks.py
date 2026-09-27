"""Tool-call checks: whether an agent's pending tool call may run under written rules.

The model reads each rule as a passage and the call as fields (`tool`, `arguments.<name>`),
and answers a yes/no question about it. A certified yes allows the call, a certified no denies
it, and a deferral or an abstention escalates it to a person.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from pydantic import JsonValue

from mimir.core.context import Context, Field, Passage
from mimir.core.decisions import YesNo
from mimir.core.results import Status, YesNoResult
from mimir.core.wire import DEFAULT_RISK

if TYPE_CHECKING:
    from mimir.core.decider import Decider

DEFAULT_QUESTION: Final = "Is this tool call allowed under the rules?"
RULE_TITLE: Final = "Rule"


class Permission(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """The permission for one pending call, and the decision it was read from. `result` is
    None for a tool the check does not apply to."""

    tool: str
    permission: Permission
    result: YesNoResult | None

    @property
    def reason(self) -> str:
        """One sentence for the agent or the approver, naming the tool and the confidence."""
        if self.result is None:
            return f"The call to {self.tool} is not covered by the check."
        confidence = self.result.confidence
        measured = "" if confidence is None else f" (confidence {confidence:.2f})"
        match self.permission:
            case Permission.ALLOW:
                return f"The call to {self.tool} is allowed under the rules{measured}."
            case Permission.DENY:
                return f"The call to {self.tool} is not allowed under the rules{measured}."
            case Permission.ESCALATE:
                cause = (
                    "the check abstained"
                    if self.result.deferral is None
                    else f"the check was deferred: {self.result.deferral.reason}"
                )
                return f"The call to {self.tool} needs a person's approval; {cause}."


@dataclass(frozen=True, slots=True)
class ToolCallCheck:
    """A yes/no decision on pending tool calls, made against `rules`.

    `tools` names the tools it applies to; None applies it to every tool. Calls to other tools
    are allowed without a decision.
    """

    decider: "Decider" = field(repr=False)
    rules: tuple[str, ...]
    question: str = DEFAULT_QUESTION
    tools: frozenset[str] | None = None
    risk: float = DEFAULT_RISK

    def __post_init__(self) -> None:
        if not self.rules:
            message = "a tool-call check needs at least one rule to decide against"
            raise ValueError(message)
        blank = [index for index, rule in enumerate(self.rules) if not rule.strip()]
        if blank:
            message = f"rules {blank} are blank"
            raise ValueError(message)
        YesNo(self.question)

    def applies_to(self, tool: str) -> bool:
        """Whether calls to `tool` are decided rather than allowed outright."""
        return self.tools is None or tool in self.tools

    def context(self, tool: str, arguments: Mapping[str, JsonValue]) -> Context:
        """The context the model reads for one pending call."""
        return Context(
            passages=tuple(Passage(title=RULE_TITLE, text=rule) for rule in self.rules),
            fields=Field.from_json({"tool": tool, "arguments": dict(arguments)}),
        )

    def __call__(self, tool: str, arguments: Mapping[str, JsonValue]) -> CheckOutcome:
        if not self.applies_to(tool):
            return _unchecked(tool)
        spec = YesNo(self.question)
        result = self.decider.decide(self.context(tool, arguments), spec, risk=self.risk)
        return _outcome(tool, result)

    async def acall(self, tool: str, arguments: Mapping[str, JsonValue]) -> CheckOutcome:
        if not self.applies_to(tool):
            return _unchecked(tool)
        spec = YesNo(self.question)
        result = await self.decider.adecide(self.context(tool, arguments), spec, risk=self.risk)
        return _outcome(tool, result)


def _unchecked(tool: str) -> CheckOutcome:
    return CheckOutcome(tool=tool, permission=Permission.ALLOW, result=None)


def _outcome(tool: str, result: YesNoResult) -> CheckOutcome:
    if result.status is Status.DECIDED and result.answer is not None:
        permission = Permission.ALLOW if result.answer else Permission.DENY
    else:
        permission = Permission.ESCALATE
    return CheckOutcome(tool=tool, permission=permission, result=result)
