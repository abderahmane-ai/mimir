"""The `Decider` base class, implemented by `Mimir` (local) and `MimirClient` (HTTP).

Subclasses implement `_run` and `info`. The public methods, shortcuts and async variants are
defined here.
"""

import abc
import asyncio
from collections.abc import Mapping, Sequence
from typing import overload

from mimir.core.context import Context, ContextLike
from mimir.core.decisions import (
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
    VerifyResult,
    YesNoResult,
)
from mimir.core.tools import DecisionTool
from mimir.core.wire import DEFAULT_RISK, ModelInfo

Items = Sequence[tuple[ContextLike, DecisionSpec]]
Requests = Sequence[tuple[Context, DecisionSpec]]


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
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        """Return one result per request, in order. `risk=None` skips the policy."""

    async def _arun(
        self,
        requests: Requests,
        *,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        return await asyncio.to_thread(
            self._run, requests, risk=risk, alpha=alpha, batch_size=batch_size
        )

    @abc.abstractmethod
    def info(self) -> ModelInfo:
        """Return the loaded model, runtime and certified risk levels."""

    @overload
    def decide(
        self, context: ContextLike, spec: Choice, *, risk: float = ..., alpha: float | None = ...
    ) -> ChoiceResult: ...
    @overload
    def decide(
        self,
        context: ContextLike,
        spec: MultiChoice,
        *,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> MultiChoiceResult: ...
    @overload
    def decide(
        self, context: ContextLike, spec: YesNo, *, risk: float = ..., alpha: float | None = ...
    ) -> YesNoResult: ...
    @overload
    def decide(
        self, context: ContextLike, spec: Verify, *, risk: float = ..., alpha: float | None = ...
    ) -> VerifyResult: ...
    @overload
    def decide(
        self, context: ContextLike, spec: Rank, *, risk: float = ..., alpha: float | None = ...
    ) -> RankResult: ...
    @overload
    def decide(
        self, context: ContextLike, spec: Rate, *, risk: float = ..., alpha: float | None = ...
    ) -> RateResult: ...
    @overload
    def decide(
        self, context: ContextLike, spec: Estimate, *, risk: float = ..., alpha: float | None = ...
    ) -> EstimateResult: ...
    def decide(
        self,
        context: ContextLike,
        spec: DecisionSpec,
        *,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> DecisionResult:
        """Make a certified decision.

        `risk` must be one of `info().risk_levels`. `alpha` is the miscoverage of the conformal
        prediction set; None uses the release default.
        """
        requests = [(Context.coerce(context), spec)]
        found = self._run(requests, risk=_certified_risk(risk), alpha=alpha, batch_size=None)
        return found[0]

    def decide_many(
        self,
        items: Items,
        *,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
        batch_size: int | None = None,
    ) -> list[DecisionResult]:
        """Make certified decisions for many `(context, spec)` pairs, returned in order.

        Requests are sorted by length and packed into batches up to the release's token budget.
        `batch_size` additionally caps the number of requests per batch.
        """
        requests = [(Context.coerce(context), spec) for context, spec in items]
        return self._run(requests, risk=_certified_risk(risk), alpha=alpha, batch_size=batch_size)

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
        """Return the raw model answer, without calibration, gating or certificate."""
        requests = [(Context.coerce(context), spec)]
        return self._run(requests, risk=None, alpha=None, batch_size=None)[0]

    def decide_uncertified_many(
        self, items: Items, *, batch_size: int | None = None
    ) -> list[DecisionResult]:
        """Return raw model answers for many `(context, spec)` pairs, batched as `decide_many`."""
        requests = [(Context.coerce(context), spec) for context, spec in items]
        return self._run(requests, risk=None, alpha=None, batch_size=batch_size)

    @overload
    async def adecide(
        self, context: ContextLike, spec: Choice, *, risk: float = ..., alpha: float | None = ...
    ) -> ChoiceResult: ...
    @overload
    async def adecide(
        self,
        context: ContextLike,
        spec: MultiChoice,
        *,
        risk: float = ...,
        alpha: float | None = ...,
    ) -> MultiChoiceResult: ...
    @overload
    async def adecide(
        self, context: ContextLike, spec: YesNo, *, risk: float = ..., alpha: float | None = ...
    ) -> YesNoResult: ...
    @overload
    async def adecide(
        self, context: ContextLike, spec: Verify, *, risk: float = ..., alpha: float | None = ...
    ) -> VerifyResult: ...
    @overload
    async def adecide(
        self, context: ContextLike, spec: Rank, *, risk: float = ..., alpha: float | None = ...
    ) -> RankResult: ...
    @overload
    async def adecide(
        self, context: ContextLike, spec: Rate, *, risk: float = ..., alpha: float | None = ...
    ) -> RateResult: ...
    @overload
    async def adecide(
        self, context: ContextLike, spec: Estimate, *, risk: float = ..., alpha: float | None = ...
    ) -> EstimateResult: ...
    async def adecide(
        self,
        context: ContextLike,
        spec: DecisionSpec,
        *,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> DecisionResult:
        """Async version of `decide`."""
        requests = [(Context.coerce(context), spec)]
        found = await self._arun(requests, risk=_certified_risk(risk), alpha=alpha, batch_size=None)
        return found[0]

    async def adecide_many(
        self,
        items: Items,
        *,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
        batch_size: int | None = None,
    ) -> list[DecisionResult]:
        """Async version of `decide_many`."""
        requests = [(Context.coerce(context), spec) for context, spec in items]
        return await self._arun(
            requests, risk=_certified_risk(risk), alpha=alpha, batch_size=batch_size
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
        found = await self._arun(requests, risk=None, alpha=None, batch_size=None)
        return found[0]

    async def adecide_uncertified_many(
        self, items: Items, *, batch_size: int | None = None
    ) -> list[DecisionResult]:
        """Async version of `decide_uncertified_many`."""
        requests = [(Context.coerce(context), spec) for context, spec in items]
        return await self._arun(requests, risk=None, alpha=None, batch_size=batch_size)

    def choose(
        self,
        context: ContextLike,
        question: str,
        options: Sequence[str] | Mapping[str, str],
        *,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> ChoiceResult:
        return self.decide(context, Choice(question, options), risk=risk, alpha=alpha)

    def yes_no(
        self, context: ContextLike, question: str, *, risk: float = DEFAULT_RISK
    ) -> YesNoResult:
        return self.decide(context, YesNo(question), risk=risk)

    def verify(
        self, context: ContextLike, claim: str, *, risk: float = DEFAULT_RISK
    ) -> VerifyResult:
        return self.decide(context, Verify(claim), risk=risk)

    def rank(
        self,
        context: ContextLike,
        question: str,
        candidates: Sequence[str] | Mapping[str, str],
        *,
        risk: float = DEFAULT_RISK,
    ) -> RankResult:
        return self.decide(context, Rank(question, candidates), risk=risk)

    def rate(
        self,
        context: ContextLike,
        question: str,
        levels: Sequence[str] | Mapping[str, str],
        *,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> RateResult:
        return self.decide(context, Rate(question, levels), risk=risk, alpha=alpha)

    def estimate(
        self,
        context: ContextLike,
        question: str,
        low: float,
        high: float,
        *,
        unit: str | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> EstimateResult:
        return self.decide(context, Estimate(question, low, high, unit), risk=risk, alpha=alpha)

    async def achoose(
        self,
        context: ContextLike,
        question: str,
        options: Sequence[str] | Mapping[str, str],
        *,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> ChoiceResult:
        return await self.adecide(context, Choice(question, options), risk=risk, alpha=alpha)

    async def ayes_no(
        self, context: ContextLike, question: str, *, risk: float = DEFAULT_RISK
    ) -> YesNoResult:
        return await self.adecide(context, YesNo(question), risk=risk)

    async def averify(
        self, context: ContextLike, claim: str, *, risk: float = DEFAULT_RISK
    ) -> VerifyResult:
        return await self.adecide(context, Verify(claim), risk=risk)

    async def arank(
        self,
        context: ContextLike,
        question: str,
        candidates: Sequence[str] | Mapping[str, str],
        *,
        risk: float = DEFAULT_RISK,
    ) -> RankResult:
        return await self.adecide(context, Rank(question, candidates), risk=risk)

    async def arate(
        self,
        context: ContextLike,
        question: str,
        levels: Sequence[str] | Mapping[str, str],
        *,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> RateResult:
        return await self.adecide(context, Rate(question, levels), risk=risk, alpha=alpha)

    async def aestimate(
        self,
        context: ContextLike,
        question: str,
        low: float,
        high: float,
        *,
        unit: str | None = None,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> EstimateResult:
        return await self.adecide(
            context, Estimate(question, low, high, unit), risk=risk, alpha=alpha
        )

    def tool(
        self,
        name: str,
        spec: DecisionSpec,
        description: str,
        *,
        risk: float = DEFAULT_RISK,
        alpha: float | None = None,
    ) -> DecisionTool:
        """Create a `DecisionTool` that applies `spec` to any context it is called with."""
        return DecisionTool(
            name=name, spec=spec, description=description, decider=self, risk=risk, alpha=alpha
        )
