import json
import re
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).parents[2]
SERVER_NAME: Final = "io.github.mythologic/mimir"
LABEL: Final = re.compile(r'io\.modelcontextprotocol\.server\.name="([^"]+)"')


def _server() -> dict[str, object]:
    loaded: dict[str, object] = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    return loaded


def test_the_registry_entry_matches_the_package_version() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    assert version == metadata.version("mimirai")
    server = _server()
    assert (server["name"], server["version"]) == (SERVER_NAME, version)
    packages = server["packages"]
    assert isinstance(packages, list)
    pypi, oci = packages
    assert (pypi["identifier"], pypi["version"]) == ("mimirai", version)
    assert pypi["runtimeArguments"][0]["value"] == f"mimirai[local,mcp]=={version}"
    assert oci["identifier"] == f"ghcr.io/mythologic/mimir:{version}-cpu"


def test_every_ownership_proof_names_the_registry_entry() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"<!-- mcp-name: {SERVER_NAME} -->" in readme
    for variant in ("cpu", "cuda"):
        dockerfile = (ROOT / "docker" / f"{variant}.Dockerfile").read_text(encoding="utf-8")
        assert LABEL.findall(dockerfile) == [SERVER_NAME]


def test_the_registry_command_exists() -> None:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    assert scripts == {"mimir": "mimir.cli.app:main", "mimirai": "mimir.cli.app:main"}
