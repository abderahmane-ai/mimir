import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mimir.cli.app import app
from mimir.runtime.session import torch_version

RUNNER = CliRunner()


def run(*arguments: str, stdin: str | None = None) -> tuple[int, str, str]:
    result = RUNNER.invoke(app, list(arguments), input=stdin)
    return result.exit_code, result.stdout, result.stderr


@pytest.fixture(autouse=True)
def _scripted_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the CLI's engines on scripted outputs instead of release weights."""
    import mimir.runtime.engine as engine_module
    import mimir.runtime.session as session_module
    from tests.conftest import fake_run

    monkeypatch.setattr(engine_module, "load_model", lambda *_: None)
    monkeypatch.setattr(session_module, "run", fake_run)


def test_schema_prints_every_schema_or_one() -> None:
    code, out, _ = run("schema")
    assert code == 0
    assert "DecisionResult" in json.loads(out)
    code, out, _ = run("schema", "YesNo")
    assert json.loads(out)["title"] == "YesNo"
    code, _, err = run("schema", "Nope")
    assert code == 1
    assert "no schema named 'Nope'" in err


def test_doctor_reports_the_environment() -> None:
    code, out, _ = run("doctor")
    report = json.loads(out)
    assert code == 0
    assert report["torch"] == torch_version()
    assert report["cuda_available"] in (True, False)
    assert report["device"] in {"cpu", "cuda"}


def test_doctor_verify_loads_the_model(release: Path) -> None:
    code, out, _ = run(
        "doctor", "--verify", "--model", str(release), "--allow-unsigned", "--device", "cpu"
    )
    assert code == 0
    assert json.loads(out)["model"]["certification"] == "certified"


def test_decide_from_flags(release: Path) -> None:
    code, out, err = run(
        "decide",
        "--model",
        str(release),
        "--allow-unsigned",
        "--device",
        "cpu",
        "--question",
        "which team",
        "--option",
        "billing",
        "--option",
        "security",
        "--text",
        "my card was charged twice",
    )
    assert code == 0, err
    result = json.loads(out)
    assert (result["type"], result["answer"]) == ("choice", "billing")


def test_decide_from_a_request_on_stdin(release: Path) -> None:
    request = {
        "context": {"state": {"plan": "pro"}},
        "decision": {"type": "yes_no", "question": "is it late"},
    }
    code, out, err = run(
        "decide",
        "--model",
        str(release),
        "--allow-unsigned",
        "--device",
        "cpu",
        stdin=json.dumps(request),
    )
    assert code == 0, err
    assert json.loads(out)["type"] == "yes_no"


def test_decide_uncertified_and_errors(release: Path) -> None:
    code, out, _ = run(
        "decide",
        "--model",
        str(release),
        "--allow-unsigned",
        "--device",
        "cpu",
        "--uncertified",
        "--question",
        "is it late",
        "--type",
        "yes_no",
    )
    assert code == 0
    assert json.loads(out)["certificate"] is None
    code, _, err = run("decide", "--model", str(release), "--allow-unsigned", stdin="{bad json")
    assert code == 1
    assert err.startswith("error:")
    code, _, err = run("decide", "--model", str(release), "--question", "q", "--option", "a")
    assert code == 1
    assert "at least 2" in err


def test_download_verifies_a_local_release(release: Path) -> None:
    code, _, err = run("download", "--model", str(release), "--device", "cpu")
    assert code == 1
    assert "manifest.json.sigstore is missing" in err


def _labels(path: Path, count: int) -> Path:
    lines = [
        json.dumps(
            {
                "context": f"card {index}",
                "decision": {
                    "type": "multi_choice",
                    "question": "which apply",
                    "options": ["refund", "order"],
                },
                "label": ["refund", "order"],
            }
        )
        for index in range(count)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_bench_reports_per_type(release: Path, tmp_path: Path) -> None:
    labels = _labels(tmp_path / "labels.jsonl", 10)
    code, out, err = run(
        "bench", str(labels), "--model", str(release), "--allow-unsigned", "--device", "cpu"
    )
    assert code == 0, err
    report = json.loads(out)
    assert report["records"] == 10
    assert report["types"]["multi_choice"]["accuracy"]["value"] == 1.0


def test_calibrate_writes_a_custom_policy(release: Path, tmp_path: Path) -> None:
    labels = _labels(tmp_path / "labels.jsonl", 400)
    out_path = tmp_path / "policy.json"
    code, out, err = run(
        "calibrate",
        str(labels),
        "--out",
        str(out_path),
        "--model",
        str(release),
        "--allow-unsigned",
        "--device",
        "cpu",
    )
    assert code == 0, err
    report = json.loads(out)
    assert report["types"]["multilabel"]["threshold"] is not None
    assert out_path.is_file()
    assert out_path.with_suffix(".npz").is_file()
    code, out, err = run(
        "decide",
        "--model",
        str(release),
        "--allow-unsigned",
        "--device",
        "cpu",
        "--policy",
        str(out_path),
        "--question",
        "which apply",
        "--type",
        "multi_choice",
        "--option",
        "refund",
        "--option",
        "order",
    )
    assert code == 0, err
    assert json.loads(out)["certificate"]["origin"] == "custom"


def test_bench_rejects_a_bad_labels_file(release: Path, tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{}\n", encoding="utf-8")
    code, _, err = run("bench", str(labels), "--model", str(release), "--allow-unsigned")
    assert code == 1
    assert "labels.jsonl:1" in err


def test_serve_refuses_an_open_public_address(
    release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MIMIR_API_KEYS", raising=False)
    code, _, err = run("serve", "--host", "0.0.0.0", "--model", str(release), "--allow-unsigned")
    assert code == 1
    assert "error: --host 0.0.0.0 accepts connections from other machines" in err


def test_serve_takes_generic_tools_only_with_mcp() -> None:
    code, _, err = run("serve", "--generic-tools")
    assert code == 1
    assert "error: --generic-tools adds MCP tools; pass --mcp as well" in err


def test_serve_and_mcp_report_a_bad_tools_file(tmp_path: Path) -> None:
    tools = tmp_path / "tools.yaml"
    tools.write_text("tools: []\n", encoding="utf-8")
    for command in ("serve", "mcp"):
        code, _, err = run(command, "--tools", str(tools))
        assert code == 1
        assert f"error: {tools}: 1 validation error for ToolDefinitions" in err


def test_mcp_needs_tools_or_generic_tools(release: Path) -> None:
    code, _, err = run("mcp", "--model", str(release), "--allow-unsigned")
    assert code == 1
    assert "error: an MCP server needs configured tools (--tools FILE) or --generic-tools" in err
