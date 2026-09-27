import re
import tomllib
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).parents[2]
DOCS: Final = ROOT / "docs"
SNIPPET: Final = re.compile(r'^--8<-- "([^"]+)"$', re.MULTILINE)
REFERENCE: Final = re.compile(r"^::: (\S+)$", re.MULTILINE)
IMAGE_TAG: Final = re.compile(r"ghcr\.io/vathosai/mimir:([0-9][^\s-]*)-(?:cpu|cuda)")


def _nav_pages(entries: list[object]) -> list[str]:
    pages: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            pages.append(entry)
        elif isinstance(entry, dict):
            for value in entry.values():
                pages.extend(_nav_pages(value) if isinstance(value, list) else [value])
    return pages


def _pages() -> dict[str, str]:
    return {
        path.relative_to(DOCS).as_posix(): path.read_text(encoding="utf-8")
        for path in DOCS.rglob("*.md")
    }


def test_every_page_is_in_the_navigation_once() -> None:
    config = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))
    nav = _nav_pages(config["project"]["nav"])
    assert len(nav) == len(set(nav))
    assert set(nav) == set(_pages())


def test_the_framework_pages_include_every_example_once() -> None:
    included = [path for text in _pages().values() for path in SNIPPET.findall(text)]
    scripts = {
        path.relative_to(ROOT).as_posix()
        for pattern in ("*/*.py", "*/*.ts")
        for path in (ROOT / "examples").glob(pattern)
        if not path.name.endswith(".test.ts")
    }
    assert len(included) == len(set(included))
    assert set(included) == scripts


def test_the_reference_documents_every_integration() -> None:
    text = (DOCS / "reference" / "integrations.md").read_text(encoding="utf-8")
    modules = {
        f"mimir.integrations.{path.stem}"
        for path in (ROOT / "src" / "mimir" / "integrations").glob("*.py")
        if path.stem != "__init__"
    }
    assert set(REFERENCE.findall(text)) == modules


def test_every_image_tag_names_the_package_version() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    texts = [*_pages().values(), (ROOT / "README.md").read_text(encoding="utf-8")]
    tags = [tag for text in texts for tag in IMAGE_TAG.findall(text)]
    assert tags
    assert set(tags) == {version}
