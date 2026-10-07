"""The `Decider` base class, implemented by `Mimir` (local) and `MimirClient` (HTTP).

Subclasses implement `_run` and `info`. The public methods, shortcuts and async variants are
defined here.
"""

import abc
import asyncio
from collections.abc import Iterable, Mapping, Sequence
from typing import overload

from mimir.core.checks import DEFAULT_QUESTION, ToolCallCheck
from mimir.core.context import Context, ContextLike
from mimir.core.decisions import (
    DEFAULT_CHOICE_QUESTION,
    Choice,
    DecisionSpec,
    Estimate,
    MultiChoice,
    Rank,
    Rate,
    Verify,
    YesNo,
)
from mimir.core.errors import RiskLevelError
from mimir.core.results import (
    ChoiceResult,
    DecisionResult,
    EstimateResult,
    MultiChoiceResult,
    RankResult,
    RateResult,
    Status,
    VerifyResult,
    YesNoResult,
)
from mimir.core.tools import DecisionTool
from mimir.core.wire import DEFAULT_RISK, Mode, ModelInfo, check_mode

Items = Sequence[tuple[ContextLike, DecisionSpec]]
Requests = Sequence[tuple[Context, DecisionSpec]]
ChoiceOptions = Sequence[str] | Mapping[str, str]


def _question_and_options(
    question: str | ChoiceOptions | None, options: ChoiceOptions | None
) -> tuple[str, ChoiceOptions]:
    """Read `choose`'s second argument as the question, or as the options when it is not text."""
    if question is None or isinstance(question, str):
        if options is None:
            message = f"choose needs options; got question={question!r} and options=None"
            raise TypeError(message)
        return (DEFAULT_CHOICE_QUESTION if question is None else question), options
    if options is not None:
        message = f"choose got options twice: {question!r} as the second argument and {options!r}"
        raise TypeError(message)
    return DEFAULT_CHOICE_QUESTION, question


def _picked(result: ChoiceResult) -> str | None:
    return result.answer if result.status == Status.DECIDED else None


def _certified_risk(risk: float | None) -> float:
    if risk is None:
        message = "risk None: pass a certified risk level, or call decide_uncertified"
        raise RiskLevelError(message)
    return risk


class Decider(abc.ABC):
    """Abstract base for decision engines."""

    @abc.abstractmethod
    def _run(
        self,
        requests: Requests,
        *,
        mode: Mode,
        min_confidence: float | None,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        """Return one result per request, in order. `risk=None` skips the policy."""

    async def _arun(
        self,
        requests: Requests,
        *,
        mode: Mode,
        min_confidence: float | None,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        return await asyncio.to_thread(
            self._run,
            requests,
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
            batch_size=batch_size,
        )

    @abc.abstractmethod
    def info(self) -> ModelInfo:
        """Return the loaded model, runtime and certified risk levels."""

    @overload
    def decide(
        self,
        context: ContextLike,
        spec: Choice,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> ChoiceResult: ...
    @overload
    def decide(
        self,
        context: ContextLike,
        spec: MultiChoice,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> MultiChoiceResult: ...
    @overload
    def decide(
        self,
        context: ContextLike,
        spec: YesNo,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> YesNoResult: ...
    @overload
    def decide(
        self,
        context: ContextLike,
        spec: Verify,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> VerifyResult: ...
    @overload
    def decide(
        self,
        context: ContextLike,
        spec: Rank,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> RankResult: ...
    @overload
    def decide(
        self,
        context: ContextLike,
        spec: Rate,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> RateResult: ...
    @overload
    def decide(
        self,
        context: ContextLike,
        spec: Estimate,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> EstimateResult: ...
    def decide(
        self,
        context: ContextLike,
        spec: DecisionSpec,
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> DecisionResult:
        """Make a decision.

        `standard` answers every request. `threshold` defers answers below
        `min_confidence`. `certified` defers answers below the policy's threshold at `risk`.
        `risk` must be one of `info().risk_levels`; it annotates the certificate. `alpha` is
        the miscoverage of the conformal prediction set; None uses the release default.
        """
        check_mode(mode, min_confidence)
        requests = [(Context.coerce(context), spec)]
        found = self._run(
            requests,
            mode=mode,
            min_confidence=min_confidence,
            risk=_certified_risk(risk),
            alpha=alpha,
            batch_size=None,
        )
        return found[0]

    def decide_many(
        self,
        items: Items,
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
        batch_size: int | None = None,
    ) -> list[DecisionResult]:
        """Make decisions for many `(context, spec)` pairs, returned in order.

        Requests are sorted by length and packed into batches up to the release's token budget.
        `batch_size` additionally caps the number of requests per batch.
        """
        check_mode(mode, min_confidence)
        requests = [(Context.coerce(context), spec) for context, spec in items]
        return self._run(
            requests,
            mode=mode,
            min_confidence=min_confidence,
            risk=_certified_risk(risk),
            alpha=alpha,
            batch_size=batch_size,
        )

    @overload
    def decide_uncertified(self, context: ContextLike, spec: Choice) -> ChoiceResult: ...
    @overload
    def decide_uncertified(self, context: ContextLike, spec: MultiChoice) -> MultiChoiceResult: ...
    @overload
    def decide_uncertified(self, context: ContextLike, spec: YesNo) -> YesNoResult: ...
    @overload
    def decide_uncertified(self, context: ContextLike, spec: Verify) -> VerifyResult: ...
    @overload
    def decide_uncertified(self, context: ContextLike, spec: Rank) -> RankResult: ...
    @overload
    def decide_uncertified(self, context: ContextLike, spec: Rate) -> RateResult: ...
    @overload
    def decide_uncertified(self, context: ContextLike, spec: Estimate) -> EstimateResult: ...
    def decide_uncertified(self, context: ContextLike, spec: DecisionSpec) -> DecisionResult:
        """Return the raw model answer, without calibration or certificate."""
        requests = [(Context.coerce(context), spec)]
        return self._run(
            requests,
            mode=Mode.STANDARD,
            min_confidence=None,
            risk=None,
            alpha=None,
            batch_size=None,
        )[0]

    def decide_uncertified_many(
        self, items: Items, *, batch_size: int | None = None
    ) -> list[DecisionResult]:
        """Return raw model answers for many `(context, spec)` pairs, batched as `decide_many`."""
        requests = [(Context.coerce(context), spec) for context, spec in items]
        return self._run(
            requests,
            mode=Mode.STANDARD,
            min_confidence=None,
            risk=None,
            alpha=None,
            batch_size=batch_size,
        )

    @overload
    async def adecide(
        self,
        context: ContextLike,
        spec: Choice,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> ChoiceResult: ...
    @overload
    async def adecide(
        self,
        context: ContextLike,
        spec: MultiChoice,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> MultiChoiceResult: ...
    @overload
    async def adecide(
        self,
        context: ContextLike,
        spec: YesNo,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> YesNoResult: ...
    @overload
    async def adecide(
        self,
        context: ContextLike,
        spec: Verify,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> VerifyResult: ...
    @overload
    async def adecide(
        self,
        context: ContextLike,
        spec: Rank,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> RankResult: ...
    @overload
    async def adecide(
        self,
        context: ContextLike,
        spec: Rate,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> RateResult: ...
    @overload
    async def adecide(
        self,
        context: ContextLike,
        spec: Estimate,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> EstimateResult: ...
    async def adecide(
        self,
        context: ContextLike,
        spec: DecisionSpec,
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> DecisionResult:
        """Async version of `decide`."""
        check_mode(mode, min_confidence)
        requests = [(Context.coerce(context), spec)]
        found = await self._arun(
            requests,
            mode=mode,
            min_confidence=min_confidence,
            risk=_certified_risk(risk),
            alpha=alpha,
            batch_size=None,
        )
        return found[0]

    async def adecide_many(
        self,
        items: Items,
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
        batch_size: int | None = None,
    ) -> list[DecisionResult]:
        """Async version of `decide_many`."""
        check_mode(mode, min_confidence)
        requests = [(Context.coerce(context), spec) for context, spec in items]
        return await self._arun(
            requests,
            mode=mode,
            min_confidence=min_confidence,
            risk=_certified_risk(risk),
            alpha=alpha,
            batch_size=batch_size,
        )

    @overload
    async def adecide_uncertified(self, context: ContextLike, spec: Choice) -> ChoiceResult: ...
    @overload
    async def adecide_uncertified(
        self, context: ContextLike, spec: MultiChoice
    ) -> MultiChoiceResult: ...
    @overload
    async def adecide_uncertified(self, context: ContextLike, spec: YesNo) -> YesNoResult: ...
    @overload
    async def adecide_uncertified(self, context: ContextLike, spec: Verify) -> VerifyResult: ...
    @overload
    async def adecide_uncertified(self, context: ContextLike, spec: Rank) -> RankResult: ...
    @overload
    async def adecide_uncertified(self, context: ContextLike, spec: Rate) -> RateResult: ...
    @overload
    async def adecide_uncertified(self, context: ContextLike, spec: Estimate) -> EstimateResult: ...
    async def adecide_uncertified(self, context: ContextLike, spec: DecisionSpec) -> DecisionResult:
        """Async version of `decide_uncertified`."""
        requests = [(Context.coerce(context), spec)]
        found = await self._arun(
            requests,
            mode=Mode.STANDARD,
            min_confidence=None,
            risk=None,
            alpha=None,
            batch_size=None,
        )
        return found[0]

    async def adecide_uncertified_many(
        self, items: Items, *, batch_size: int | None = None
    ) -> list[DecisionResult]:
        """Async version of `decide_uncertified_many`."""
        requests = [(Context.coerce(context), spec) for context, spec in items]
        return await self._arun(
            requests,
            mode=Mode.STANDARD,
            min_confidence=None,
            risk=None,
            alpha=None,
            batch_size=batch_size,
        )

    @overload
    def choose(
        self,
        context: ContextLike,
        question: str,
        options: ChoiceOptions,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> ChoiceResult: ...
    @overload
    def choose(
        self,
        context: ContextLike,
        options: ChoiceOptions,
        /,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> ChoiceResult: ...
    @overload
    def choose(
        self,
        context: ContextLike,
        *,
        options: ChoiceOptions,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> ChoiceResult: ...
    def choose(
        self,
        context: ContextLike,
        question: str | ChoiceOptions | None = None,
        options: ChoiceOptions | None = None,
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> ChoiceResult:
        """Pick one option, or none. Without a question, `choose(context, options)` asks
        `DEFAULT_CHOICE_QUESTION`."""
        text, found = _question_and_options(question, options)
        return self.decide(
            context,
            Choice(text, found),
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
        )

    def pick(
        self,
        context: ContextLike,
        options: ChoiceOptions,
        *,
        question: str | None = None,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
    ) -> str | None:
        """Return the chosen option id, or None when the result is not `DECIDED`."""
        return _picked(
            self.choose(
                context,
                DEFAULT_CHOICE_QUESTION if question is None else question,
                options,
                mode=mode,
                min_confidence=min_confidence,
                risk=risk,
            )
        )

    def yes_no(
        self,
        context: ContextLike,
        question: str,
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
    ) -> YesNoResult:
        return self.decide(
            context, YesNo(question), mode=mode, min_confidence=min_confidence, risk=risk
        )

    def verify(
        self,
        context: ContextLike,
        claim: str,
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
    ) -> VerifyResult:
        return self.decide(
            context, Verify(claim), mode=mode, min_confidence=min_confidence, risk=risk
        )

    def rank(
        self,
        context: ContextLike,
        question: str,
        candidates: Sequence[str] | Mapping[str, str],
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
    ) -> RankResult:
        return self.decide(
            context,
            Rank(question, candidates),
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
        )

    def rate(
        self,
        context: ContextLike,
        question: str,
        levels: Sequence[str] | Mapping[str, str],
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> RateResult:
        return self.decide(
            context,
            Rate(question, levels),
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
        )

    def estimate(
        self,
        context: ContextLike,
        question: str,
        low: float,
        high: float,
        *,
        unit: str | None = None,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> EstimateResult:
        return self.decide(
            context,
            Estimate(question, low, high, unit),
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
        )

    @overload
    async def achoose(
        self,
        context: ContextLike,
        question: str,
        options: ChoiceOptions,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> ChoiceResult: ...
    @overload
    async def achoose(
        self,
        context: ContextLike,
        options: ChoiceOptions,
        /,
        *,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> ChoiceResult: ...
    @overload
    async def achoose(
        self,
        context: ContextLike,
        *,
        options: ChoiceOptions,
        mode: Mode = ...,
        min_confidence: float | None = ...,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> ChoiceResult: ...
    async def achoose(
        self,
        context: ContextLike,
        question: str | ChoiceOptions | None = None,
        options: ChoiceOptions | None = None,
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> ChoiceResult:
        """Async version of `choose`."""
        text, found = _question_and_options(question, options)
        return await self.adecide(
            context,
            Choice(text, found),
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
        )

    async def apick(
        self,
        context: ContextLike,
        options: ChoiceOptions,
        *,
        question: str | None = None,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
    ) -> str | None:
        """Async version of `pick`."""
        return _picked(
            await self.achoose(
                context,
                DEFAULT_CHOICE_QUESTION if question is None else question,
                options,
                mode=mode,
                min_confidence=min_confidence,
                risk=risk,
            )
        )

    async def ayes_no(
        self,
        context: ContextLike,
        question: str,
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
    ) -> YesNoResult:
        return await self.adecide(
            context, YesNo(question), mode=mode, min_confidence=min_confidence, risk=risk
        )

    async def averify(
        self,
        context: ContextLike,
        claim: str,
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
    ) -> VerifyResult:
        return await self.adecide(
            context, Verify(claim), mode=mode, min_confidence=min_confidence, risk=risk
        )

    async def arank(
        self,
        context: ContextLike,
        question: str,
        candidates: Sequence[str] | Mapping[str, str],
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
    ) -> RankResult:
        return await self.adecide(
            context,
            Rank(question, candidates),
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
        )

    async def arate(
        self,
        context: ContextLike,
        question: str,
        levels: Sequence[str] | Mapping[str, str],
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> RateResult:
        return await self.adecide(
            context,
            Rate(question, levels),
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
        )

    async def aestimate(
        self,
        context: ContextLike,
        question: str,
        low: float,
        high: float,
        *,
        unit: str | None = None,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> EstimateResult:
        return await self.adecide(
            context,
            Estimate(question, low, high, unit),
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
        )

    def tool(
        self,
        name: str,
        spec: DecisionSpec,
        description: str,
        *,
        mode: Mode = Mode.STANDARD,
        min_confidence: float | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> DecisionTool:
        """Create a `DecisionTool` that applies `spec` to any context it is called with."""
        check_mode(mode, min_confidence)
        return DecisionTool(
            name=name,
            spec=spec,
            description=description,
            decider=self,
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
            alpha=alpha,
        )

    def tool_call_check(
        self,
        rules: str | Sequence[str],
        *,
        question: str = DEFAULT_QUESTION,
        tools: Iterable[str] | None = None,
        mode: Mode = Mode.THRESHOLD,
        min_confidence: float | None = 0.5,
        risk: float = DEFAULT_RISK,
    ) -> ToolCallCheck:
        """Create a `ToolCallCheck` deciding pending calls of `tools` (every tool if None)
        against `rules`."""
        check_mode(mode, min_confidence)
        return ToolCallCheck(
            decider=self,
            rules=(rules,) if isinstance(rules, str) else tuple(rules),
            question=question,
            tools=None if tools is None else frozenset(tools),
            mode=mode,
            min_confidence=min_confidence,
            risk=risk,
        )
