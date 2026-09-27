"""The `mimir` command line.

Commands print JSON to stdout. Errors print `error: <message>` to stderr and exit with status 1.
"""

import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Annotated, Final, Literal

import typer
from pydantic import JsonValue, ValidationError

from mimir.core.context import Context, Field
from mimir.core.decider import Decider
from mimir.core.decisions import Choice, DecisionSpec, MultiChoice, Rank, Rate, Verify, YesNo
from mimir.core.errors import MimirError
from mimir.core.labels import LabelError, read_labelled
from mimir.core.schema import json_schemas
from mimir.core.wire import DEFAULT_RISK, DecideRequest
from mimir.evaluation.bench import bench as bench_results
from mimir.extras import engine_class

DEFAULT_MODEL: Final = "vathosai/mimir-1"
SpecType = Literal["choice", "multi_choice", "yes_no", "verify", "rank", "rate"]

app = typer.Typer(
    name="mimir",
    help="Typed, calibrated and certified decisions with MIMIR.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

Model = Annotated[str, typer.Option(help="Hub repository id or local release directory.")]
Revision = Annotated[str | None, typer.Option(help="Hub revision; defaults to the pinned one.")]
DeviceOption = Annotated[str, typer.Option(help="auto, cpu or cuda.")]
VariantOption = Annotated[str | None, typer.Option(help="Graph variant, e.g. fp32 or fp16.")]
PolicyOption = Annotated[Path | None, typer.Option(help="Custom policy JSON from `calibrate`.")]
Unsigned = Annotated[
    bool, typer.Option(help="Accept a local release directory without a manifest signature.")
]
Server = Annotated[
    str | None, typer.Option(help="Use a MIMIR server at this URL instead of a local model.")
]


def _emit(value: JsonValue) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


def _decider(
    *,
    model: str,
    revision: str | None,
    device: str,
    variant: str | None,
    policy: Path | None,
    allow_unsigned: bool,
    server: str | None,
) -> Decider:
    if server is not None:
        from mimir.client.http import MimirClient

        return MimirClient(server)
    return engine_class("mimir").from_pretrained(
        model,
        revision=revision,
        device=device,
        variant=variant,
        policy=policy,
        allow_unsigned=allow_unsigned,
    )


def _spec(kind: SpecType, question: str, options: list[str]) -> DecisionSpec:
    match kind:
        case "choice":
            return Choice(question, options)
        case "multi_choice":
            return MultiChoice(question, options)
        case "yes_no":
            return YesNo(question)
        case "verify":
            return Verify(question)
        case "rank":
            return Rank(question, options)
        case "rate":
            return Rate(question, options)


def _read_json(path: Path | None) -> JsonValue:
    raw = sys.stdin.read() if path is None else path.read_text(encoding="utf-8")
    loaded: JsonValue = json.loads(raw)
    return loaded


@app.command()
def decide(
    question: Annotated[str | None, typer.Option(help="The question; omit to read JSON.")] = None,
    option: Annotated[list[str] | None, typer.Option(help="An option; repeat for each.")] = None,
    kind: Annotated[SpecType, typer.Option("--type", help="Decision spec type.")] = "choice",
    text: Annotated[str | None, typer.Option(help="Context passage.")] = None,
    state: Annotated[
        Path | None, typer.Option(help="JSON file read as a structured state.")
    ] = None,
    request: Annotated[
        Path | None, typer.Option(help="DecideRequest JSON file; stdin when omitted.")
    ] = None,
    risk: Annotated[float, typer.Option(help="Certified risk level.")] = DEFAULT_RISK,
    uncertified: Annotated[bool, typer.Option(help="Skip the policy.")] = False,
    model: Model = DEFAULT_MODEL,
    revision: Revision = None,
    device: DeviceOption = "auto",
    variant: VariantOption = None,
    policy: PolicyOption = None,
    allow_unsigned: Unsigned = False,
    server: Server = None,
) -> None:
    """Make one decision from flags, or from a DecideRequest JSON on stdin."""
    try:
        if question is not None:
            if state is not None:
                context = Context(fields=tuple(Field.from_json(_read_json(state))))
            else:
                context = Context.coerce(text or "")
            spec = _spec(kind, question, option or [])
            alpha = None
        else:
            parsed = DecideRequest.model_validate(_read_json(request))
            context, spec, risk, alpha = (
                Context.coerce(parsed.context),
                parsed.decision,
                parsed.risk,
                parsed.alpha,
            )
        decider = _decider(
            model=model,
            revision=revision,
            device=device,
            variant=variant,
            policy=policy,
            allow_unsigned=allow_unsigned,
            server=server,
        )
        result = (
            decider.decide_uncertified(context, spec)
            if uncertified
            else decider.decide(context, spec, risk=risk, alpha=alpha)
        )
    except (MimirError, ValidationError, ValueError, OSError) as error:
        raise _fail(str(error)) from error
    _emit(result.model_dump(mode="json"))


@app.command()
def schema(
    name: Annotated[
        str | None, typer.Argument(help="One schema by name; all when omitted.")
    ] = None,
) -> None:
    """Print the JSON Schemas of every spec, result and request."""
    schemas = json_schemas()
    if name is None:
        _emit(schemas)
        return
    if name not in schemas:
        message = f"no schema named {name!r}; choose one of {sorted(schemas)}"
        raise _fail(message)
    _emit(schemas[name])


@app.command()
def download(
    model: Model = DEFAULT_MODEL,
    revision: Revision = None,
    device: DeviceOption = "auto",
    variant: VariantOption = None,
    cache_dir: Annotated[Path | None, typer.Option(help="Hub cache directory.")] = None,
) -> None:
    """Download and verify a release for offline use."""
    try:
        from mimir.runtime.artifact import load_snapshot
        from mimir.runtime.session import resolve_device

        snapshot = load_snapshot(
            model,
            revision=revision,
            cache_dir=cache_dir,
            variant=variant,
            device=resolve_device(device),
        )
    except (MimirError, ModuleNotFoundError) as error:
        raise _fail(str(error)) from error
    _emit(
        {
            "model": snapshot.model,
            "revision": snapshot.revision,
            "variant": snapshot.variant,
            "path": str(snapshot.root),
            "signed": snapshot.is_signed,
            "files": dict(snapshot.digests),
        }
    )


def _environment() -> dict[str, JsonValue]:
    found: dict[str, JsonValue] = {
        "mimirai": metadata.version("mimirai"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        from mimir.runtime.equivalence import cache_directory
        from mimir.runtime.hardware import hardware_name
        from mimir.runtime.session import installed_runtimes, resolve_device, runtime_version
    except ModuleNotFoundError as error:
        found["local_engine"] = f"not installed ({error.name}): pip install 'mimirai[local]'"
        return found
    import onnxruntime

    runtimes = installed_runtimes()
    device = resolve_device("auto")
    found |= {
        "onnxruntime": runtime_version(),
        "runtime_distributions": list(runtimes),
        "runtime_conflict": len(runtimes) > 1,
        "providers": list(onnxruntime.get_available_providers()),
        "device": device,
        "hardware": hardware_name(device),
        "cache": str(cache_directory()),
    }
    return found


@app.command()
def doctor(
    verify: Annotated[bool, typer.Option(help="Load the model and run its checks.")] = False,
    model: Model = DEFAULT_MODEL,
    revision: Revision = None,
    device: DeviceOption = "auto",
    variant: VariantOption = None,
    allow_unsigned: Unsigned = False,
) -> None:
    """Report the environment; with --verify, load the model and report its certification."""
    report = _environment()
    if verify:
        try:
            engine = engine_class("mimir doctor --verify").from_pretrained(
                model,
                revision=revision,
                device=device,
                variant=variant,
                allow_unsigned=allow_unsigned,
            )
        except MimirError as error:
            raise _fail(str(error)) from error
        report["model"] = engine.info().model_dump(mode="json")
    _emit(report)


@app.command()
def bench(
    path: Annotated[Path, typer.Argument(help="Labelled JSONL file.")],
    risk: Annotated[float, typer.Option(help="Certified risk level.")] = DEFAULT_RISK,
    batch_size: Annotated[int | None, typer.Option(help="Requests per batch.")] = None,
    model: Model = DEFAULT_MODEL,
    revision: Revision = None,
    device: DeviceOption = "auto",
    variant: VariantOption = None,
    policy: PolicyOption = None,
    allow_unsigned: Unsigned = False,
    server: Server = None,
) -> None:
    """Report accuracy, coverage and realised risk on labelled decisions."""
    try:
        labelled = read_labelled(path)
        decider = _decider(
            model=model,
            revision=revision,
            device=device,
            variant=variant,
            policy=policy,
            allow_unsigned=allow_unsigned,
            server=server,
        )
        results = decider.decide_many(
            [(item.context, item.decision) for item in labelled], risk=risk, batch_size=batch_size
        )
    except (MimirError, LabelError, OSError) as error:
        raise _fail(str(error)) from error
    report = bench_results(results, [item.label for item in labelled])
    _emit({"risk": risk, **report.model_dump(mode="json")})


@app.command()
def calibrate(
    path: Annotated[Path, typer.Argument(help="Labelled JSONL file.")],
    out: Annotated[Path, typer.Option(help="Policy JSON to write; the .npz goes alongside.")],
    risk: Annotated[float, typer.Option(help="Risk level to certify.")] = DEFAULT_RISK,
    confidence: Annotated[float, typer.Option(help="Confidence of the certificate.")] = 0.95,
    batch_size: Annotated[int | None, typer.Option(help="Requests per batch.")] = None,
    model: Model = DEFAULT_MODEL,
    revision: Revision = None,
    device: DeviceOption = "auto",
    variant: VariantOption = None,
    allow_unsigned: Unsigned = False,
) -> None:
    """Certify thresholds on your labelled decisions and write a custom policy."""
    try:
        from mimir.policy.document import write_policy
        from mimir.runtime.calibrate import calibrate as certify

        labelled = read_labelled(path)
        engine = engine_class("mimir calibrate").from_pretrained(
            model,
            revision=revision,
            device=device,
            variant=variant,
            allow_unsigned=allow_unsigned,
        )
        calibration = certify(
            engine, labelled, risk=risk, confidence=confidence, batch_size=batch_size
        )
        write_policy(calibration.policy, out, out.with_suffix(".npz"))
    except (MimirError, LabelError, OSError, ValueError) as error:
        raise _fail(str(error)) from error
    _emit(
        {
            "policy": str(out),
            "risk": risk,
            "confidence": confidence,
            "types": {
                item.model_type: {
                    "records": item.records,
                    "passed_gate": item.passed_gate,
                    "needed": item.needed,
                    **item.certified.model_dump(mode="json"),
                }
                for item in calibration.types
            },
        }
    )


def main() -> None:
    app()
