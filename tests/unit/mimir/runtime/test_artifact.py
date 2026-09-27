import json
import shutil
import socket
from pathlib import Path

import pytest

from mimir.core.errors import ArtifactError, FormatVersionError, IntegrityError, SignatureError
from mimir.runtime.artifact import (
    Snapshot,
    choose_variant,
    file_sha256,
    load_snapshot,
    variant_files,
)
from mimir.runtime.release import Manifest, ReleaseConfig
from mimir.runtime.signature import ManifestVerifier
from tests.conftest import write_manifest


class AcceptingVerifier:
    def __init__(self) -> None:
        self.seen: list[tuple[bytes, bytes]] = []

    def verify(self, manifest: bytes, bundle: bytes, where: str) -> None:
        assert where.endswith("manifest.json")
        self.seen.append((manifest, bundle))


class RefusingVerifier:
    def verify(self, manifest: bytes, bundle: bytes, where: str) -> None:
        message = f"{where}: bad signature over {len(manifest)} and {len(bundle)} bytes"
        raise SignatureError(message)


def load(
    root: Path, *, verifier: ManifestVerifier | None = None, allow_unsigned: bool = False
) -> Snapshot:
    return load_snapshot(
        str(root),
        revision=None,
        cache_dir=None,
        variant=None,
        device="cpu",
        verifier=verifier,
        allow_unsigned=allow_unsigned,
    )


def test_unsigned_local_release_loads_only_when_allowed(release: Path) -> None:
    with pytest.raises(SignatureError, match=r"manifest\.json\.sigstore is missing"):
        load(release)
    snapshot = load(release, allow_unsigned=True)
    assert (snapshot.revision, snapshot.variant, snapshot.is_signed) == ("local", "fp32", False)
    assert snapshot.model == str(release.resolve())
    assert set(snapshot.digests) == {
        "config.json",
        "tokenizer.json",
        "onnx/model.onnx",
        "onnx/model.onnx_data",
        "policy/fp32.json",
        "policy/fp32.npz",
    }
    assert snapshot.digests["onnx/model.onnx"] == file_sha256(release / "onnx" / "model.onnx")


def test_signed_release_is_verified_with_the_manifest_bytes(release: Path) -> None:
    (release / "manifest.json.sigstore").write_bytes(b"bundle")
    verifier = AcceptingVerifier()
    snapshot = load(release, verifier=verifier)
    assert snapshot.is_signed
    assert verifier.seen == [((release / "manifest.json").read_bytes(), b"bundle")]


def test_a_bad_signature_stops_loading(release: Path) -> None:
    (release / "manifest.json.sigstore").write_bytes(b"bundle")
    with pytest.raises(SignatureError, match="bad signature"):
        load(release, verifier=RefusingVerifier(), allow_unsigned=True)


@pytest.mark.parametrize(
    "relative", ["onnx/model.onnx_data", "onnx/model.onnx", "tokenizer.json", "policy/fp32.npz"]
)
def test_a_tampered_file_is_refused(release: Path, relative: str) -> None:
    path = release / relative
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(IntegrityError, match=f"{relative}: SHA-256"):
        load(release, allow_unsigned=True)


def test_a_tampered_config_is_refused_before_it_is_parsed(release: Path) -> None:
    (release / "config.json").write_text("not json", encoding="utf-8")
    with pytest.raises(IntegrityError, match=r"config\.json: SHA-256"):
        load(release, allow_unsigned=True)


def test_a_missing_file_is_refused(release: Path) -> None:
    (release / "tokenizer.json").unlink()
    with pytest.raises(IntegrityError, match=r"tokenizer\.json is in the manifest but missing"):
        load(release, allow_unsigned=True)


def test_a_file_missing_from_the_manifest_is_refused(release: Path) -> None:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    del manifest["files"]["tokenizer.json"]
    (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IntegrityError, match=r"tokenizer\.json is not in the manifest"):
        load(release, allow_unsigned=True)


def test_a_release_for_other_package_versions_is_refused(release: Path) -> None:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    manifest["loadable_by"] = {"mimirai": ">=2,<3"}
    (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FormatVersionError, match=r"requires mimirai >=2,<3; 1\.0\.0 is installed"):
        load(release, allow_unsigned=True)
    manifest["loadable_by"] = {"mimirai": "not a specifier"}
    (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FormatVersionError, match="not a version specifier"):
        load(release, allow_unsigned=True)


def test_an_unknown_config_format_is_refused(release: Path) -> None:
    config = json.loads((release / "config.json").read_text(encoding="utf-8"))
    config["format_version"] = 2
    (release / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_manifest(release)
    with pytest.raises(FormatVersionError, match="format 2"):
        load(release, allow_unsigned=True)


def test_a_missing_manifest_is_refused(release: Path) -> None:
    (release / "manifest.json").unlink()
    with pytest.raises(IntegrityError, match=r"no manifest\.json"):
        load(release, allow_unsigned=True)


def test_variant_selection(release: Path) -> None:
    config = ReleaseConfig.model_validate_json((release / "config.json").read_bytes())
    assert choose_variant(config, None, "cpu") == "fp32"
    assert choose_variant(config, "fp32", "cuda") == "fp32"
    with pytest.raises(ArtifactError, match="no variant lists device 'cuda'"):
        choose_variant(config, None, "cuda")
    with pytest.raises(ArtifactError, match="'int8' is not among"):
        choose_variant(config, "int8", "cpu")


def test_variant_files_select_only_what_the_variant_reads(release: Path) -> None:
    config = ReleaseConfig.model_validate_json((release / "config.json").read_bytes())
    files = {
        "config.json": "",
        "tokenizer.json": "",
        "onnx/model.onnx": "",
        "onnx/model.onnx_data": "",
        "onnx/model_fp16.onnx": "",
        "policy/fp32.json": "",
        "policy/fp16.json": "",
        "equivalence/inputs.npz": "",
        "README.md": "",
    }
    manifest = Manifest(format_version=1, files=files, loadable_by={})
    assert variant_files(manifest, config, "fp32") == [
        "config.json",
        "equivalence/inputs.npz",
        "onnx/model.onnx",
        "onnx/model.onnx_data",
        "policy/fp32.json",
        "tokenizer.json",
    ]


def _cached_hub_release(release: Path, cache: Path, revision: str) -> str:
    """Lay `release` out as the Hub cache holds `mythologic/mimir-1` at `revision`."""
    commit = "0123456789abcdef0123456789abcdef01234567"
    repository = cache / "models--mythologic--mimir-1"
    shutil.copytree(release, repository / "snapshots" / commit)
    (repository / "snapshots" / commit / "manifest.json.sigstore").write_bytes(b"bundle")
    (repository / "refs").mkdir()
    (repository / "refs" / revision).write_text(commit, encoding="utf-8")
    return commit


def test_offline_loads_a_cached_release_without_the_network(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = _cached_hub_release(release, tmp_path / "hub", "v1.0")

    def refuse(*_: object) -> None:
        message = "network access in an offline load"
        raise AssertionError(message)

    monkeypatch.setattr(socket.socket, "connect", refuse)
    verifier = AcceptingVerifier()
    snapshot = load_snapshot(
        "mythologic/mimir-1",
        revision=None,
        cache_dir=tmp_path / "hub",
        variant=None,
        device="cpu",
        verifier=verifier,
        offline=True,
    )
    assert (snapshot.model, snapshot.revision, snapshot.is_signed) == (
        "mythologic/mimir-1",
        commit,
        True,
    )
    assert verifier.seen == [((release / "manifest.json").read_bytes(), b"bundle")]
    assert "onnx/model.onnx" in snapshot.digests


def test_offline_names_the_cache_it_could_not_read(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match=rf"@v1\.0: .*\(offline, cache {tmp_path}\)"):
        load_snapshot(
            "mythologic/mimir-1",
            revision=None,
            cache_dir=tmp_path,
            variant=None,
            device="cpu",
            offline=True,
        )


def test_hub_errors_are_reported_as_artifact_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    with pytest.raises(ArtifactError, match=r"mythologic/does-not-exist@v1\.0"):
        load_snapshot(
            "mythologic/does-not-exist",
            revision=None,
            cache_dir=tmp_path,
            variant=None,
            device="cpu",
        )
