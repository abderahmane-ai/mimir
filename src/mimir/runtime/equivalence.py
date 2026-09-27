"""Runtime equivalence check for hardware a policy does not list.

The release ships graph inputs for a fixed set of requests (`equivalence/inputs.npz`) and, per
variant, the certified decisions on them (`equivalence/expected.npz`). The check runs the graph
on those inputs and requires every decision, and whether it is taken at each certified risk
level, to match. Passing checks are cached per graph, policy, runtime and hardware.

`expected.npz` arrays, per variant and request id:

- `<variant>/<id>/chosen`: int64 chosen option indices (empty for abstain);
- `<variant>/<id>/taken`: bool, one entry per risk level in `config.risk_levels` order.
"""

import datetime
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from mimir.core.errors import EquivalenceError, IntegrityError
from mimir.policy.assessment import assess
from mimir.policy.binding import LoadedRuntime
from mimir.policy.document import Policy, read_arrays
from mimir.runtime.artifact import Snapshot
from mimir.runtime.readout import OUTPUTS, as_float64, readout_at
from mimir.runtime.release import ReleaseConfig
from mimir.runtime.session import GraphSession

INPUTS_FILE: Final = "equivalence/inputs.npz"
EXPECTED_FILE: Final = "equivalence/expected.npz"
CACHE_ENVIRONMENT: Final = "MIMIR_CACHE"

Array = npt.NDArray[np.generic]


def cache_directory() -> Path:
    """Return `$MIMIR_CACHE`, else `$XDG_CACHE_HOME/mimir`, else `~/.cache/mimir`."""
    configured = os.environ.get(CACHE_ENVIRONMENT)
    if configured:
        return Path(configured)
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "mimir"


def equivalence_key(runtime: LoadedRuntime, policy: Policy) -> str:
    """Return the cache key for one graph, policy, runtime and hardware."""
    identity = {
        "variant": runtime.variant,
        "graph_sha256": runtime.graph_sha256,
        "weights_sha256": runtime.weights_sha256,
        "onnxruntime": runtime.onnxruntime,
        "provider": runtime.provider,
        "options_sha256": runtime.options_sha256,
        "hardware": runtime.hardware,
        "policy": policy.document.model_dump(mode="json"),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _cache_file(key: str) -> Path:
    return cache_directory() / "equivalence" / f"{key}.json"


def is_cached(key: str) -> bool:
    return _cache_file(key).is_file()


def record_pass(key: str, runtime: LoadedRuntime, records: int) -> None:
    path = _cache_file(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "hardware": runtime.hardware,
        "provider": runtime.provider,
        "onnxruntime": runtime.onnxruntime,
        "records": records,
        "checked_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    part.replace(path)


def _request_ids(arrays: Mapping[str, Array]) -> list[str]:
    return sorted({name.split("/", 1)[0] for name in arrays})


def _inputs_of(arrays: Mapping[str, Array], request: str, names: Sequence[str]) -> dict[str, Array]:
    return {name: arrays[f"{request}/inputs/{name}"] for name in names}


def check_equivalence(
    session: GraphSession, snapshot: Snapshot, policy: Policy, config: ReleaseConfig
) -> int:
    """Run the equivalence set and return the number of requests checked.

    Raises:
        IntegrityError: the release has no equivalence set for this variant.
        EquivalenceError: a decision or taken flag differs; the message lists the first ones.
    """
    if not (snapshot.has(INPUTS_FILE) and snapshot.has(EXPECTED_FILE)):
        message = (
            f"{snapshot.model}@{snapshot.revision} has no equivalence set, so hardware its "
            "policy does not list cannot be accepted; create a policy with `mimir calibrate`"
        )
        raise IntegrityError(message)
    inputs = read_arrays(snapshot.path(INPUTS_FILE))
    expected = read_arrays(snapshot.path(EXPECTED_FILE))
    names = [spec.name for spec in config.graph.inputs]
    variant = snapshot.variant
    differences: list[str] = []
    requests = _request_ids(inputs)
    for request in requests:
        feed = _inputs_of(inputs, request, names)
        outputs = as_float64(dict(zip(OUTPUTS, session.run(list(OUTPUTS), feed), strict=True)))
        count = int(feed["candidate_present"][0].sum())
        kind = config.decision_types[int(feed["decision_type"][0])]
        readout = readout_at(outputs, 0, kind, count)
        assessments = [
            assess(readout, policy, risk, config.default_alpha) for risk in config.risk_levels
        ]
        found_chosen = assessments[0].decision.chosen
        found_taken = [assessment.taken for assessment in assessments]
        chosen = tuple(int(index) for index in expected[f"{variant}/{request}/chosen"])
        taken = [bool(flag) for flag in expected[f"{variant}/{request}/taken"]]
        if found_chosen != chosen or found_taken != taken:
            differences.append(
                f"{request}: chosen {found_chosen} vs {chosen}, taken {found_taken} vs {taken}"
            )
    if differences:
        message = (
            f"{len(differences)} of {len(requests)} equivalence decisions differ on this "
            f"runtime; create a policy with `mimir calibrate`: {differences[:5]}"
        )
        raise EquivalenceError(message)
    return len(requests)
