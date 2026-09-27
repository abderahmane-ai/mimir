"""Downloading and verifying a release.

A release is a Hugging Face Hub repository at a pinned revision, or a local directory with the
same layout. `load_snapshot` verifies, in order: the manifest signature, the supported package
versions, and the SHA-256 of each file the chosen variant uses. `config.json` is parsed only
after its digest is verified. Offline loading (`offline=True` or `HF_HUB_OFFLINE`) reads only
the local cache.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Final, TypeVar

from huggingface_hub import constants, snapshot_download
from huggingface_hub.errors import EntryNotFoundError
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import BaseModel, ValidationError

from mimir.core.errors import ArtifactError, FormatVersionError, IntegrityError, SignatureError
from mimir.runtime.release import (
    CONFIG_FILE,
    FORMAT_VERSION,
    MANIFEST_FILE,
    PACKAGE,
    SIGNATURE_FILE,
    Manifest,
    ReleaseConfig,
    is_known_type_order,
)
from mimir.runtime.signature import ManifestVerifier, SigstoreVerifier

DEFAULT_MODEL: Final = "Mythologic/MIMIR-1"
DEFAULT_REVISION: Final = "v1.0"
LOCAL_REVISION: Final = "local"
EQUIVALENCE_DIR: Final = "equivalence"

Model = TypeVar("Model", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A verified release. `digests` maps each verified file, relative to `root`, to its
    SHA-256."""

    root: Path
    model: str
    revision: str
    config: ReleaseConfig
    variant: str
    digests: Mapping[str, str]
    is_signed: bool

    def path(self, relative: str) -> Path:
        return self.root / relative

    def has(self, relative: str) -> bool:
        return relative in self.digests


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _download(
    model: str, revision: str, cache_dir: Path | None, patterns: list[str], *, offline: bool
) -> Path:
    try:
        found = snapshot_download(
            model,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=patterns,
            local_files_only=offline,
            library_name=PACKAGE,
            library_version=metadata.version(PACKAGE),
        )
    except (OSError, EntryNotFoundError) as error:
        where = f" (offline, cache {cache_dir or constants.HF_HUB_CACHE})" if offline else ""
        message = f"{model}@{revision}: cannot fetch {patterns}{where}: {error}"
        raise ArtifactError(message) from error
    return Path(found)


def _parse(kind: type[Model], path: Path) -> Model:
    try:
        return kind.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        message = f"{path}: {error}"
        raise ArtifactError(message) from error


def _check_loadable(manifest: Manifest, where: Path) -> None:
    if manifest.format_version != FORMAT_VERSION:
        message = f"{where}: manifest format {manifest.format_version}; this package reads 1"
        raise FormatVersionError(message)
    accepted = manifest.loadable_by.get(PACKAGE)
    version = metadata.version(PACKAGE)
    try:
        is_loadable = accepted is not None and SpecifierSet(accepted).contains(
            version, prereleases=True
        )
    except InvalidSpecifier as error:
        message = f"{where}: loadable_by {accepted!r} is not a version specifier"
        raise FormatVersionError(message) from error
    if not is_loadable:
        message = f"{where}: this release requires {PACKAGE} {accepted}; {version} is installed"
        raise FormatVersionError(message)


def _verify_files(root: Path, manifest: Manifest, files: list[str]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative in files:
        expected = manifest.files.get(relative)
        if expected is None:
            message = f"{root}: {relative} is not in the manifest"
            raise IntegrityError(message)
        path = root / relative
        if not path.is_file():
            message = f"{root}: {relative} is in the manifest but missing"
            raise IntegrityError(message)
        found = file_sha256(path)
        if found != expected:
            message = f"{path}: SHA-256 {found}, the manifest has {expected}"
            raise IntegrityError(message)
        verified[relative] = found
    return verified


def variant_files(manifest: Manifest, config: ReleaseConfig, variant: str) -> list[str]:
    """Return the files a variant uses: config, tokenizer, graph and external data, policy and
    equivalence set."""
    chosen = config.variants[variant]
    graph = PurePosixPath(chosen.graph)
    files = {CONFIG_FILE, config.tokenizer, chosen.graph}
    for relative in manifest.files:
        path = PurePosixPath(relative)
        if path.parent == graph.parent and path.name.startswith(graph.name):
            files.add(relative)
        if relative in {f"{chosen.policy}.json", f"{chosen.policy}.npz"}:
            files.add(relative)
        if path.parts[0] == EQUIVALENCE_DIR:
            files.add(relative)
    return sorted(files)


def choose_variant(config: ReleaseConfig, variant: str | None, device: str) -> str:
    """Return `variant` if given and present, else the first variant listing `device`."""
    if variant is not None:
        if variant not in config.variants:
            message = f"variant {variant!r} is not among {sorted(config.variants)}"
            raise ArtifactError(message)
        return variant
    for name, entry in config.variants.items():
        if device in entry.devices:
            return name
    message = f"no variant lists device {device!r}: {sorted(config.variants)}"
    raise ArtifactError(message)


def load_snapshot(
    model: str,
    *,
    revision: str | None,
    cache_dir: Path | None,
    variant: str | None,
    device: str,
    verifier: ManifestVerifier | None = None,
    allow_unsigned: bool = False,
    offline: bool = False,
) -> Snapshot:
    """Download (if needed) and verify a release for one variant.

    Args:
        model: A Hub repository id or a local directory. Local releases report revision `local`.
        revision: Hub revision; defaults to `DEFAULT_REVISION`.
        allow_unsigned: Accept a local directory without a manifest signature. Hub releases
            always require one.
        offline: Read only from `cache_dir` and verify the signature with the cached trust
            root, with no network access. Also on when `HF_HUB_OFFLINE` is set.
    """
    is_offline = offline or constants.HF_HUB_OFFLINE
    local = Path(model)
    is_local = local.is_dir()
    if is_local:
        root, resolved = local, LOCAL_REVISION
    else:
        pinned = revision or DEFAULT_REVISION
        first = [CONFIG_FILE, MANIFEST_FILE, SIGNATURE_FILE]
        root = _download(model, pinned, cache_dir, first, offline=is_offline)
        resolved = root.name
    manifest_path = root / MANIFEST_FILE
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        message = f"{root}: no {MANIFEST_FILE}: {error}"
        raise IntegrityError(message) from error
    bundle_path = root / SIGNATURE_FILE
    is_signed = bundle_path.is_file()
    if is_signed:
        chosen_verifier = verifier or SigstoreVerifier(offline=is_offline)
        chosen_verifier.verify(manifest_bytes, bundle_path.read_bytes(), str(manifest_path))
    elif not (is_local and allow_unsigned):
        message = f"{root}: {SIGNATURE_FILE} is missing"
        raise SignatureError(message)
    manifest = _parse(Manifest, manifest_path)
    _check_loadable(manifest, manifest_path)
    _verify_files(root, manifest, [CONFIG_FILE])
    config = _parse(ReleaseConfig, root / CONFIG_FILE)
    if config.format_version != FORMAT_VERSION:
        message = f"{root / CONFIG_FILE}: format {config.format_version}; this package reads 1"
        raise FormatVersionError(message)
    if not is_known_type_order(config):
        message = f"{root / CONFIG_FILE}: decision types {list(config.decision_types)} differ"
        raise FormatVersionError(message)
    chosen = choose_variant(config, variant, device)
    needed = variant_files(manifest, config, chosen)
    if not is_local:
        root = _download(model, resolved, cache_dir, needed, offline=is_offline)
    digests = _verify_files(root, manifest, needed)
    return Snapshot(
        root=root,
        model=str(local.resolve()) if is_local else model,
        revision=resolved,
        config=config,
        variant=chosen,
        digests=digests,
        is_signed=is_signed,
    )
