import importlib.util
import sys

import pytest

from mimir.core.errors import MissingExtraError
from mimir.extras import LOCAL_MODULES, MCP_MODULES, SERVER_MODULES, engine_class, require_extra
from mimir.runtime.engine import Mimir


def test_engine_class_imports_the_engine() -> None:
    assert engine_class("test") is Mimir


def test_a_missing_module_outside_the_extras_is_not_disguised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mimir.runtime.engine", None)
    with pytest.raises(ModuleNotFoundError) as caught:
        engine_class("test")
    assert not isinstance(caught.value, MissingExtraError)


def test_local_modules_are_the_local_extra() -> None:
    assert {
        "torch",
        "transformers",
        "safetensors",
        "numpy",
        "tokenizers",
        "sigstore",
        "huggingface_hub",
    } == (LOCAL_MODULES)


def test_server_and_mcp_modules_are_their_extras() -> None:
    assert {"fastapi", "prometheus_client", "uvicorn", "yaml"} == SERVER_MODULES
    assert {"mcp", "uvicorn", "yaml"} == MCP_MODULES


def test_require_extra_names_the_extra_and_the_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_extra("mimir serve", "server", SERVER_MODULES)
    found = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == "uvicorn" else found(name)
    )
    with pytest.raises(MissingExtraError, match=r"mimir mcp requires the 'mcp' extra \(uvicorn"):
        require_extra("mimir mcp", "mcp", MCP_MODULES)
