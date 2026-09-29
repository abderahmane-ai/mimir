"""`Mimir`, the local decision engine on ONNX Runtime."""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tokenizers import Tokenizer

from mimir.core.context import Context, ContextLike
from mimir.core.decider import Decider, Items
from mimir.core.decisions import DecisionSpec
from mimir.core.errors import (
    EquivalenceError,
    PolicyError,
    RiskLevelError,
    UncertifiedRuntimeError,
)
from mimir.core.results import DecisionResult
from mimir.core.wire import InputLimits, ModelInfo, RuntimeInfo
from mimir.policy.assessment import Assessment, assess
from mimir.policy.binding import LoadedRuntime, bind
from mimir.policy.distributions import Readout, decide, identity, probabilities
from mimir.policy.document import Policy, read_policy
from mimir.runtime.answer import Outcome, Provenance, build_result, relevant_context
from mimir.runtime.artifact import DEFAULT_MODEL, Snapshot, load_snapshot
from mimir.runtime.batching import pack
from mimir.runtime.equivalence import check_equivalence, equivalence_key, is_cached, record_pass
from mimir.runtime.graph import check_graph, weights_path
from mimir.runtime.hardware import hardware_name
from mimir.runtime.layout import Tokenized, collate, tokenize
from mimir.runtime.readout import OUTPUTS, split_outputs
from mimir.runtime.rendering import render
from mimir.runtime.session import (
    PROVIDERS,
    Device,
    GraphSession,
    check_single_runtime,
    open_session,
    options_sha256,
    resolve_device,
    runtime_version,
)
from mimir.runtime.signature import ManifestVerifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """A result and the policy assessment behind it (None for uncertified results)."""

    result: DecisionResult
    assessment: Assessment | None


@dataclass(frozen=True, slots=True)
class LoadedPolicy:
    """A policy and how it applies to the loaded runtime."""

    policy: Policy
    certification: Literal["certified", "equivalent"]


def _relative(snapshot: Snapshot, path: Path) -> str:
    return path.relative_to(snapshot.root).as_posix()


def loaded_runtime(snapshot: Snapshot, device: Device) -> LoadedRuntime:
    """Describe the loaded graph and runtime for policy binding."""
    graph = snapshot.path(snapshot.config.variants[snapshot.variant].graph)
    weights = weights_path(graph)
    provider = PROVIDERS[device]
    return LoadedRuntime(
        variant=snapshot.variant,
        graph_sha256=snapshot.digests[_relative(snapshot, graph)],
        weights_sha256=snapshot.digests[_relative(snapshot, weights)],
        opset=snapshot.config.graph.opset.get("ai.onnx", 0),
        onnxruntime=runtime_version(),
        provider=provider,
        options_sha256=options_sha256(provider),
        hardware=hardware_name(device),
    )


def load_policy(
    snapshot: Snapshot, session: GraphSession, runtime: LoadedRuntime, custom: Path | None
) -> LoadedPolicy | None:
    """Load the custom or release policy and bind it to the runtime.

    Runs the equivalence check (once per configuration, then cached) when the policy does not
    list this hardware. Returns None when the release has no policy for the variant.
    """
    if custom is not None:
        policy = read_policy(custom, custom.with_suffix(".npz"))
    else:
        stem = snapshot.config.variants[snapshot.variant].policy
        document, arrays = f"{stem}.json", f"{stem}.npz"
        if not (snapshot.has(document) and snapshot.has(arrays)):
            return None
        policy = read_policy(snapshot.path(document), snapshot.path(arrays))
    if bind(policy.document.fingerprint, runtime) == "certified":
        return LoadedPolicy(policy, "certified")
    key = equivalence_key(runtime, policy)
    if not is_cached(key):
        records = check_equivalence(session, snapshot, policy, snapshot.config)
        record_pass(key, runtime, records)
    return LoadedPolicy(policy, "equivalent")


def _load_session(
    model: str,
    *,
    revision: str | None,
    cache_dir: Path | None,
    variant: str | None,
    device: Device,
    verifier: ManifestVerifier | None,
    allow_unsigned: bool,
    offline: bool,
) -> tuple[Snapshot, GraphSession]:
    """Load and verify one variant for a device, and open its session."""
    snapshot = load_snapshot(
        model,
        revision=revision,
        cache_dir=cache_dir,
        variant=variant,
        device=device,
        verifier=verifier,
        allow_unsigned=allow_unsigned,
        offline=offline,
    )
    graph = snapshot.path(snapshot.config.variants[snapshot.variant].graph)
    authenticated = {snapshot.path(relative).resolve() for relative in snapshot.digests}
    check_graph(graph, snapshot.config.graph, authenticated)
    return snapshot, open_session(graph, device)


def _prepare(
    model: str,
    *,
    revision: str | None,
    cache_dir: Path | None,
    variant: str | None,
    device: Device,
    verifier: ManifestVerifier | None,
    allow_unsigned: bool,
    offline: bool,
    custom: Path | None,
) -> tuple[Snapshot, GraphSession, LoadedRuntime, LoadedPolicy | None]:
    """Load one variant, open its session, and bind its policy."""
    snapshot, session = _load_session(
        model,
        revision=revision,
        cache_dir=cache_dir,
        variant=variant,
        device=device,
        verifier=verifier,
        allow_unsigned=allow_unsigned,
        offline=offline,
    )
    runtime = loaded_runtime(snapshot, device)
    loaded = load_policy(snapshot, session, runtime, custom)
    return snapshot, session, runtime, loaded


class Mimir(Decider):
    """Local MIMIR engine. Create one with `Mimir.from_pretrained`.

    Safe to share across threads.
    """

    def __init__(
        self,
        *,
        snapshot: Snapshot,
        session: GraphSession,
        tokenizer: Tokenizer,
        device: Device,
        runtime: LoadedRuntime,
        policy: LoadedPolicy | None,
    ) -> None:
        self._snapshot = snapshot
        self._session = session
        self._tokenizer = tokenizer
        self._device = device
        self._runtime = runtime
        self._policy = policy

    @classmethod
    def from_pretrained(
        cls,
        model: str = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        device: str = "auto",
        variant: str | None = None,
        policy: str | Path | None = None,
        cache_dir: str | Path | None = None,
        allow_unsigned: bool = False,
        verifier: ManifestVerifier | None = None,
        offline: bool = False,
    ) -> "Mimir":
        """Download, verify and load a release.

        Args:
            model: Hub repository id or local directory.
            revision: Hub revision; defaults to the revision pinned by this package version.
            device: `auto`, `cpu` or `cuda`. `auto` serves the best configuration that can
                decide: the device's variant when it carries a policy, else the CPU release.
            variant: Graph variant; defaults to the variant listed for the device.
            policy: Path to a custom policy JSON from `mimir calibrate` (its `.npz` alongside).
            cache_dir: Hub cache directory.
            allow_unsigned: Load a local directory that has no manifest signature.
            verifier: Manifest signature verifier; defaults to Sigstore with the release identity.
            offline: Load only from `cache_dir`, with no network access.

        Raises:
            ArtifactError: download, signature, integrity or graph contract failure.
            UncertifiedRuntimeError: the runtime does not match the policy's certification.
            EquivalenceError: the equivalence check failed on unlisted hardware.
        """
        check_single_runtime()
        cache = None if cache_dir is None else Path(cache_dir)
        custom = None if policy is None else Path(policy)
        resolved = resolve_device(device)
        try:
            snapshot, session, runtime, loaded = _prepare(
                model,
                revision=revision,
                cache_dir=cache,
                variant=variant,
                device=resolved,
                verifier=verifier,
                allow_unsigned=allow_unsigned,
                offline=offline,
                custom=custom,
            )
        except (UncertifiedRuntimeError, EquivalenceError):
            if device != "auto" or resolved != "cuda":
                raise
            logger.warning(
                "device 'auto': the CUDA release cannot serve this machine; loading the "
                "CPU release instead"
            )
            resolved = "cpu"
            snapshot, session, runtime, loaded = _prepare(
                model,
                revision=revision,
                cache_dir=cache,
                variant=variant,
                device=resolved,
                verifier=verifier,
                allow_unsigned=allow_unsigned,
                offline=offline,
                custom=custom,
            )
        else:
            if (
                loaded is None
                and custom is None
                and variant is None
                and device == "auto"
                and resolved == "cuda"
            ):
                logger.warning(
                    "device 'auto': the CUDA variant ships without a policy; loading the "
                    "certified CPU release instead"
                )
                resolved = "cpu"
                snapshot, session, runtime, loaded = _prepare(
                    model,
                    revision=revision,
                    cache_dir=cache,
                    variant=variant,
                    device=resolved,
                    verifier=verifier,
                    allow_unsigned=allow_unsigned,
                    offline=offline,
                    custom=custom,
                )
        tokenizer = Tokenizer.from_file(str(snapshot.path(snapshot.config.tokenizer)))
        return cls(
            snapshot=snapshot,
            session=session,
            tokenizer=tokenizer,
            device=resolved,
            runtime=runtime,
            policy=loaded,
        )

    def info(self) -> ModelInfo:
        config = self._snapshot.config
        runtime = self._runtime
        loaded = self._policy
        return ModelInfo(
            model=self._snapshot.model,
            revision=self._snapshot.revision,
            variant=self._snapshot.variant,
            device=self._device,
            runtime=RuntimeInfo(
                onnxruntime=runtime.onnxruntime,
                provider=runtime.provider,
                options_sha256=runtime.options_sha256,
                hardware=runtime.hardware,
            ),
            certification="none" if loaded is None else loaded.certification,
            policy=None if loaded is None else loaded.policy.document.origin,
            risk_levels=() if loaded is None else loaded.policy.risk_levels,
            decision_types=config.decision_types,
            limits=InputLimits(
                options=config.limits.options,
                levels=config.limits.levels,
                context_tokens=config.limits.context_tokens,
            ),
        )

    def _check_request(self, risk: float | None, alpha: float | None) -> float:
        config = self._snapshot.config
        chosen = config.default_alpha if alpha is None else alpha
        if not 0 < chosen < 1:
            message = f"alpha must be in (0, 1); got {chosen}"
            raise ValueError(message)
        if risk is None:
            return chosen
        if self._policy is None:
            message = (
                f"{self._snapshot.model}@{self._snapshot.revision} has no policy for variant "
                f"{self._snapshot.variant}; use decide_uncertified or pass policy="
            )
            raise PolicyError(message)
        levels = self._policy.policy.risk_levels
        if risk not in levels:
            message = f"risk {risk} is not certified; choose one of {list(levels)}"
            raise RiskLevelError(message)
        return chosen

    def _provenance(self) -> Provenance | None:
        if self._policy is None:
            return None
        document = self._policy.policy.document
        return Provenance(
            model=self._snapshot.model,
            revision=self._snapshot.revision,
            variant=self._snapshot.variant,
            confidence=document.confidence,
            origin=document.origin,
        )

    def _outcome(
        self, tokenized: Tokenized, readout: Readout, risk: float | None, alpha: float
    ) -> Outcome:
        if risk is None or self._policy is None:
            probs = probabilities(readout, identity(tokenized.rendered.model_type))
            return Outcome(probs, decide(tokenized.rendered.model_type, probs), None)
        assessment = assess(readout, self._policy.policy, risk, alpha)
        return Outcome(assessment.probabilities, assessment.decision, assessment)

    def _evaluate(
        self,
        requests: Sequence[tuple[Context, DecisionSpec]],
        *,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[Evaluation]:
        started = time.perf_counter()
        chosen_alpha = self._check_request(risk, alpha)
        config = self._snapshot.config
        tokenized = [
            tokenize(render(context, spec), self._tokenizer, config) for context, spec in requests
        ]
        provenance = self._provenance()
        found: dict[int, Evaluation] = {}
        for indices in pack(tokenized, config.layout.crossing_tokens, batch_size):
            members = [tokenized[index] for index in indices]
            batch = collate(members, config.layout.special_tokens)
            outputs = dict(zip(OUTPUTS, self._session.run(list(OUTPUTS), batch.feed), strict=True))
            split = split_outputs(
                outputs,
                batch,
                [member.rendered.model_type for member in members],
                [len(member.rendered.state) for member in members],
            )
            latency_ms = (time.perf_counter() - started) * 1000
            for index, member, output in zip(indices, members, split, strict=True):
                outcome = self._outcome(member, output.readout, risk, chosen_alpha)
                result = build_result(
                    requests[index][1],
                    outcome,
                    relevant_context(member.rendered.state, output.relevance),
                    risk,
                    provenance,
                    latency_ms,
                )
                found[index] = Evaluation(result, outcome.assessment)
        return [found[index] for index in range(len(requests))]

    def _run(
        self,
        requests: Sequence[tuple[Context, DecisionSpec]],
        *,
        risk: float | None,
        alpha: float | None,
        batch_size: int | None,
    ) -> list[DecisionResult]:
        evaluations = self._evaluate(requests, risk=risk, alpha=alpha, batch_size=batch_size)
        return [evaluation.result for evaluation in evaluations]

    def assess_many(
        self, items: Items, *, alpha: float | None = None, batch_size: int | None = None
    ) -> list[Evaluation]:
        """Return results with their policy assessments, as `mimir calibrate` needs.

        Assessments are computed at the loaded policy's first risk level; their calibrated
        scores, decisions and gate outcomes do not depend on the risk level.
        """
        levels = () if self._policy is None else self._policy.policy.risk_levels
        if not levels:
            message = "assess_many needs a loaded policy with at least one risk level"
            raise PolicyError(message)
        requests = [(Context.coerce(context), spec) for context, spec in items]
        risk = levels[0]
        return self._evaluate(requests, risk=risk, alpha=alpha, batch_size=batch_size)

    def count_tokens(self, context: ContextLike, spec: DecisionSpec) -> int:
        """Return the encoder tokens a request uses, before batch padding."""
        rendered = render(Context.coerce(context), spec)
        return tokenize(rendered, self._tokenizer, self._snapshot.config).encoded_tokens

    @property
    def policy(self) -> Policy | None:
        """The loaded policy, or None."""
        return None if self._policy is None else self._policy.policy

    @property
    def runtime(self) -> LoadedRuntime:
        """The loaded graph and runtime, as policy fingerprints describe them."""
        return self._runtime

    @property
    def snapshot(self) -> Snapshot:
        """The verified release."""
        return self._snapshot
