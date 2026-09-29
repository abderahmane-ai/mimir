import subprocess
import sys
import textwrap
from importlib import metadata

import mimir

# Runs in a fresh interpreter where the `local` extra's modules cannot be imported, as in a base
# install: the contract, the client, the compat translation and the bench must all load.
WITHOUT_EXTRAS = textwrap.dedent(
    """
    import importlib.abc
    import sys

    BLOCKED = {"huggingface_hub", "numpy", "onnx", "onnxruntime", "sigstore", "tokenizers"}

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, Block())

    import mimir
    from mimir import Choice, Context, ChoiceResult, MimirClient
    import mimir.core.schema
    import mimir.compat.systemone.v1
    import mimir.evaluation.bench

    mimir.core.schema.json_schemas()
    assert not BLOCKED & set(name.split(".")[0] for name in sys.modules)
    try:
        mimir.Mimir
    except mimir.core.errors.MissingExtraError as error:
        print(error)
    else:
        raise SystemExit("Mimir imported without the local extra")
    """
)


def test_the_contract_and_client_import_without_extras() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT_EXTRAS], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert "pip install 'mimir-decisions[local]'" in completed.stdout


def test_public_names_resolve() -> None:
    for name in mimir.__all__:
        assert getattr(mimir, name) is not None
    assert mimir.__version__ == metadata.version("mimir-decisions")
