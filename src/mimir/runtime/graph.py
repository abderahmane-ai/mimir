"""Graph contract checks, run on the graph structure before a session is created.

A graph is rejected if it uses an opset or operator outside the contract, defines local
functions, names an external-data location that is not a plain file name beside it or
points at a file not in the manifest, or has different inputs or outputs. All failures
are reported together.
"""

from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Final

import numpy as np
import onnx
from google.protobuf.message import DecodeError

from mimir.core.errors import GraphContractError
from mimir.runtime.release import GraphContract, TensorSpec

DEFAULT_DOMAIN: Final = "ai.onnx"
EXTERNAL_LOCATION: Final = "location"


def _domain(name: str) -> str:
    return name or DEFAULT_DOMAIN


def _specs(values: Iterable[onnx.ValueInfoProto]) -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for value in values:
        tensor = value.type.tensor_type
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)).name
        specs.append(TensorSpec(name=value.name, dtype=dtype, rank=len(tensor.shape.dim)))
    return tuple(specs)


def _operators(graph: onnx.GraphProto) -> set[str]:
    found: set[str] = set()
    for node in graph.node:
        found.add(f"{_domain(node.domain)}::{node.op_type}")
        for attribute in node.attribute:
            for inner in (attribute.g, *attribute.graphs):
                found |= _operators(inner)
    return found


def external_locations(model: onnx.ModelProto) -> set[str]:
    """Return the external-data locations referenced by the graph's initializers."""
    locations: set[str] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            entries = {entry.key: entry.value for entry in tensor.external_data}
            locations.add(entries.get(EXTERNAL_LOCATION, ""))
    return locations


def contract_failures(
    model: onnx.ModelProto, contract: GraphContract, graph: Path, authenticated: Collection[Path]
) -> list[str]:
    """Return all contract violations of `model`, stored beside `graph`.

    `authenticated` contains the resolved paths verified against the manifest. The external
    data must be named by a plain file name and resolve to one of them: the Hub cache
    materialises release files as symlinks into a shared, sharded blob store, so no
    directory comparison survives every layout.
    """
    failures: list[str] = []
    opset = {_domain(entry.domain): int(entry.version) for entry in model.opset_import}
    if opset != contract.opset:
        failures.append(f"opset {opset}, the contract allows {contract.opset}")
    foreign = sorted(_operators(model.graph) - set(contract.operators))
    if foreign:
        failures.append(f"operators outside the contract: {foreign}")
    if model.functions:
        names = sorted(function.name for function in model.functions)
        failures.append(f"local functions {names}; the contract allows none")
    root = graph.parent
    for location in sorted(external_locations(model)):
        if not location or Path(location).name != location:
            failures.append(f"external data at {location!r} is outside {root}")
        elif (root / location).resolve() not in authenticated:
            failures.append(f"external data at {location!r} is not authenticated by the manifest")
    for kind, found, expected in (
        ("inputs", _specs(model.graph.input), contract.inputs),
        ("outputs", _specs(model.graph.output), contract.outputs),
    ):
        if found != expected:
            failures.append(f"{kind} {list(found)}, the contract declares {list(expected)}")
    return failures


def read_structure(path: Path) -> onnx.ModelProto:
    """Load a graph without its external data."""
    try:
        return onnx.load(str(path), load_external_data=False)
    except (OSError, DecodeError) as error:
        message = f"{path}: not a readable ONNX graph: {error}"
        raise GraphContractError(message) from error


def weights_path(path: Path) -> Path:
    """Return the graph's external-data file, or the graph itself if its weights are inline."""
    locations = sorted(external_locations(read_structure(path)))
    if len(locations) > 1:
        message = f"{path}: weights are split over {len(locations)} files; expected one"
        raise GraphContractError(message)
    return path.parent / locations[0] if locations else path


def check_graph(path: Path, contract: GraphContract, authenticated: Collection[Path]) -> None:
    """Raise `GraphContractError` listing every violation of the graph at `path`."""
    failures = contract_failures(read_structure(path), contract, path, authenticated)
    if failures:
        message = f"{path}: " + "; ".join(failures)
        raise GraphContractError(message)
