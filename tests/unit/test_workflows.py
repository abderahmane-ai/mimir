import re
from pathlib import Path
from typing import Final

import pytest
import yaml

from mimir.runtime.signature import RELEASE_IDENTITY, RELEASE_ISSUER

ROOT: Final = Path(__file__).parents[2]
WORKFLOWS: Final = ROOT / ".github" / "workflows"
REPOSITORY: Final = "https://github.com/vathosai/mimir"
MAIN_ONLY: Final = "github.ref == 'refs/heads/main'"
PINNED: Final = re.compile(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}|\$/\.github/workflows/[\w-]+\.yml")

Mapping = dict[object, object]


def _mapping(value: object) -> Mapping:
    assert isinstance(value, dict)
    return value


def _workflow(name: str) -> Mapping:
    return _mapping(yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8")))


def _jobs(workflow: Mapping) -> dict[str, Mapping]:
    return {str(name): _mapping(job) for name, job in _mapping(workflow["jobs"]).items()}


def _triggers(workflow: Mapping) -> set[object]:
    # PyYAML reads the key `on` as the boolean true (YAML 1.1).
    return set(_mapping(workflow[True]))


def _needs(job: Mapping) -> set[str]:
    needs = job.get("needs", [])
    if isinstance(needs, str):
        return {needs}
    assert isinstance(needs, list)
    return {str(need) for need in needs}


def _ancestors(jobs: dict[str, Mapping], name: str) -> set[str]:
    direct = _needs(jobs[name])
    return direct.union(*(_ancestors(jobs, parent) for parent in direct))


def _steps(job: Mapping) -> list[Mapping]:
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    return [_mapping(step) for step in steps]


def test_the_signing_workflow_is_the_identity_the_package_pins() -> None:
    assert f"{REPOSITORY}/.github/workflows/sign-model.yml@refs/heads/main" == RELEASE_IDENTITY
    assert RELEASE_ISSUER == "https://token.actions.githubusercontent.com"
    workflow = _workflow("sign-model.yml")
    assert _triggers(workflow) == {"workflow_dispatch"}
    sign = _jobs(workflow)["sign"]
    assert sign["if"] == MAIN_ONLY
    assert sign["permissions"] == {"contents": "read", "id-token": "write"}


def test_releases_run_only_from_main_and_images_name_that_identity() -> None:
    identity = f"{REPOSITORY}/.github/workflows/release.yml@refs/heads/main"
    containers = (ROOT / "docs" / "surfaces" / "containers.md").read_text(encoding="utf-8")
    assert f"--certificate-identity {identity}" in containers
    workflow = _workflow("release.yml")
    assert _triggers(workflow) == {"workflow_dispatch"}
    roots = [job.get("if") for job in _jobs(workflow).values() if not _needs(job)]
    assert roots
    assert set(roots) == {MAIN_ONLY}


def test_the_model_tag_precedes_every_publication() -> None:
    jobs = _jobs(_workflow("release.yml"))
    for published in ("pypi", "images", "registry", "docs", "github"):
        assert {"gate", "build", "model"} <= _ancestors(jobs, published), published
    assert {"pypi", "images"} <= _ancestors(jobs, "registry")
    assert set(jobs) == _ancestors(jobs, "github") | {"github"}


def test_pypi_is_published_by_trusted_publishing() -> None:
    pypi = _jobs(_workflow("release.yml"))["pypi"]
    assert pypi["permissions"] == {"id-token": "write"}
    publish = [step for step in _steps(pypi) if "pypi-publish" in str(step.get("uses"))]
    assert len(publish) == 1
    assert "with" not in publish[0]


@pytest.mark.parametrize("name", ["check.yml", "release.yml", "sign-model.yml"])
def test_every_action_is_pinned_to_a_commit(name: str) -> None:
    jobs = _jobs(_workflow(name)).values()
    uses = [str(item["uses"]) for job in jobs for item in [job, *_steps(job)] if "uses" in item]
    assert uses
    assert [use for use in uses if PINNED.fullmatch(use) is None] == []
