from pathlib import Path

import onnx
import pytest
from onnx import helper

from mimir.core.errors import GraphContractError
from mimir.runtime.graph import (
    check_graph,
    contract_failures,
    external_locations,
    read_structure,
    weights_path,
)
from mimir.runtime.release import TensorSpec
from tests.conftest import build_graph, graph_contract


def _authenticated(graph: Path) -> set[Path]:
    return {graph.resolve(), (graph.parent / "model.onnx_data").resolve()}


def test_the_built_graph_keeps_its_contract(tmp_path: Path) -> None:
    graph = tmp_path / "model.onnx"
    build_graph(graph)
    check_graph(graph, graph_contract(graph), _authenticated(graph))
    assert external_locations(read_structure(graph)) == {"model.onnx_data"}
    assert weights_path(graph) == tmp_path / "model.onnx_data"


def test_every_violation_is_reported_together(tmp_path: Path) -> None:
    graph = tmp_path / "model.onnx"
    build_graph(graph)
    contract = graph_contract(graph)
    narrowed = contract.model_copy(
        update={
            "opset": {"ai.onnx": 19},
            "operators": tuple(op for op in contract.operators if op != "ai.onnx::Div"),
            "outputs": (
                *contract.outputs[:-1],
                TensorSpec(name="workspace", dtype="float16", rank=2),
            ),
        }
    )
    failures = contract_failures(read_structure(graph), narrowed, graph, _authenticated(graph))
    assert len(failures) == 3
    assert "opset {'ai.onnx': 20}" in failures[0]
    assert "['ai.onnx::Div']" in failures[1]
    assert failures[2].startswith("outputs")


def test_external_data_must_stay_in_the_directory_and_be_authenticated(tmp_path: Path) -> None:
    graph = tmp_path / "inner" / "model.onnx"
    build_graph(graph)
    model = read_structure(graph)
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = "../outside.bin"
    failures = contract_failures(model, graph_contract(graph), graph, {graph.resolve()})
    assert failures == [f"external data at '../outside.bin' is outside {graph.parent}"]
    local = tmp_path / "model.onnx"
    build_graph(local)
    failures = contract_failures(read_structure(local), graph_contract(local), local, set())
    assert failures == ["external data at 'model.onnx_data' is not authenticated by the manifest"]


def test_external_data_survives_the_hub_cache_symlinks(tmp_path: Path) -> None:
    graph_blob = tmp_path / "blobs" / "aa" / "graph"
    data_blob = tmp_path / "blobs" / "c6" / "data"
    graph_blob.parent.mkdir(parents=True)
    data_blob.parent.mkdir(parents=True)
    staged = tmp_path / "staged"
    staged.mkdir()
    build_graph(staged / "model.onnx")
    (staged / "model.onnx").rename(graph_blob)
    (staged / "model.onnx_data").rename(data_blob)
    snapshot = tmp_path / "snapshots" / "abc" / "onnx"
    snapshot.mkdir(parents=True)
    linked = snapshot / "model.onnx"
    linked.symlink_to(graph_blob)
    data = snapshot / "model.onnx_data"
    data.symlink_to(data_blob)
    authenticated = {linked.resolve(), data.resolve()}
    failures = contract_failures(
        read_structure(linked), graph_contract(graph_blob), linked, authenticated
    )
    assert failures == []
    escape = tmp_path / "escape.bin"
    escape.write_bytes(b"not weights")
    data.unlink()
    data.symlink_to(escape)
    failures = contract_failures(
        read_structure(linked), graph_contract(graph_blob), linked, authenticated
    )
    assert failures == ["external data at 'model.onnx_data' is not authenticated by the manifest"]


def test_local_functions_are_refused(tmp_path: Path) -> None:
    graph = tmp_path / "model.onnx"
    build_graph(graph)
    model = read_structure(graph)
    function = helper.make_function(
        "custom", "Identity2", ["x"], ["y"], [], [helper.make_opsetid("", 20)]
    )
    model.functions.append(function)
    failures = contract_failures(model, graph_contract(graph), graph, _authenticated(graph))
    assert failures == ["local functions ['Identity2']; the contract allows none"]


def test_unreadable_graphs_raise(tmp_path: Path) -> None:
    graph = tmp_path / "model.onnx"
    graph.write_bytes(b"\x00\x01 not a protobuf")
    with pytest.raises(GraphContractError, match="not a readable ONNX graph"):
        read_structure(graph)
    with pytest.raises(GraphContractError, match="not a readable ONNX graph"):
        read_structure(tmp_path / "missing.onnx")


def test_check_graph_raises_with_the_path(tmp_path: Path) -> None:
    graph = tmp_path / "model.onnx"
    build_graph(graph)
    contract = graph_contract(graph).model_copy(update={"operators": ()})
    with pytest.raises(GraphContractError, match=r"model\.onnx: operators outside"):
        check_graph(graph, contract, _authenticated(graph))


def test_weights_split_over_files_are_refused(tmp_path: Path) -> None:
    graph = tmp_path / "model.onnx"
    build_graph(graph)
    model = read_structure(graph)
    tensor = next(item for item in model.graph.initializer if item.name == "quarter")
    onnx.external_data_helper.set_external_data(tensor, "second.bin")
    onnx.save_model(model, str(graph))
    with pytest.raises(GraphContractError, match="split over 2 files"):
        weights_path(graph)
