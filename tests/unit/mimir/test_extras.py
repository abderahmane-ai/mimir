import sys

import pytest

from mimir.core.errors import MissingExtraError
from mimir.extras import LOCAL_MODULES, engine_class
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
    assert {"onnxruntime", "numpy", "tokenizers", "sigstore", "huggingface_hub", "onnx"} == (
        LOCAL_MODULES
    )
