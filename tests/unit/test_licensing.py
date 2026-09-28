from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).parents[2]
LICENSING_FILES: Final = (
    "LICENSE",
    "MODEL-LICENSE.md",
    "COMMERCIAL-LICENSING.md",
    "TRADEMARKS.md",
)
RETIRED_FILES: Final = ("MIMIR-COMMUNITY-LICENSE.md", "MIMIR-COMMERCIAL-LICENSE.md")
STALE_TERMS: Final = (
    "500,000",
    "500k",
    "two years",
    "two-year",
    "community license",
    "community licence",
    "commercial license",
    "commercial licence",
    "apache-2.0 after",
    "mimir-community-license",
    "mimir-commercial-license",
)
SKIPPED_DIRECTORIES: Final = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "artifacts",
        "dist",
        "node_modules",
    }
)
SKIPPED_PREFIXES: Final = (".venv",)
SKIPPED_NAMES: Final = frozenset({"CLAUDE.md", "openapi.json", "package-lock.json", "uv.lock"})
TEXT_SUFFIXES: Final = frozenset(
    {".cfg", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".ts", ".txt", ".yaml", ".yml"}
)


def _texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in sorted(ROOT.rglob("*")):
        parts = path.relative_to(ROOT).parts
        if path.is_dir() or any(part in SKIPPED_DIRECTORIES for part in parts):
            continue
        if any(part.startswith(SKIPPED_PREFIXES) for part in parts):
            continue
        if path.name in SKIPPED_NAMES or path == Path(__file__):
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {"LICENSE", "Makefile"}:
            continue
        texts[path.relative_to(ROOT).as_posix()] = path.read_text(encoding="utf-8", errors="ignore")
    return texts


def test_the_licensing_files_are_the_published_set() -> None:
    for name in LICENSING_FILES:
        assert (ROOT / name).is_file(), f"{name} is missing"
    for name in RETIRED_FILES:
        assert not (ROOT / name).exists(), f"{name} must not be published"


def test_the_revenue_threshold_is_defined_and_used() -> None:
    license_text = (ROOT / "MODEL-LICENSE.md").read_text(encoding="utf-8")
    assert "**“Revenue Threshold”** means US$1,000,000 in annual gross revenue." in license_text
    assert license_text.count("Revenue Threshold") >= 3


def test_the_readme_states_both_licenses_and_links_them() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Licensing" in readme
    assert "Apache-2.0" in readme
    assert "MODEL-LICENSE.md" in readme
    assert "COMMERCIAL-LICENSING.md" in readme


def test_no_stale_licensing_terms_survive() -> None:
    offenders = [
        f"{name}: {term}"
        for name, text in _texts().items()
        for term in STALE_TERMS
        if term in text.lower()
    ]
    assert not offenders, f"stale licensing terms: {offenders}"
