from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest
from huggingface_hub.hf_api import CommitInfo, GitRefInfo, GitRefs

from hub import HubError, attach_signature, check_commit, tag_release
from mimir.runtime.artifact import DEFAULT_REVISION
from mimir.runtime.release import SIGNATURE_FILE

MODEL: Final = "vathosai/mimir-1"
SIGNED: Final = "a" * 40
OTHER: Final = "b" * 40
NEW: Final = "c" * 40


@dataclass
class FakeHub:
    tags: dict[str, str] = field(default_factory=dict)
    uploads: list[dict[str, object]] = field(default_factory=list)
    created: list[tuple[str, str, str]] = field(default_factory=list)

    def upload_file(
        self,
        *,
        path_or_fileobj: bytes,
        path_in_repo: str,
        repo_id: str,
        revision: str,
        commit_message: str,
        parent_commit: str,
    ) -> CommitInfo:
        self.uploads.append(
            {
                "bytes": path_or_fileobj,
                "path": path_in_repo,
                "repo": repo_id,
                "revision": revision,
                "parent": parent_commit,
                "message": commit_message,
            }
        )
        return CommitInfo(
            commit_url=f"https://huggingface.co/{repo_id}/commit/{NEW}",
            commit_message=commit_message,
            commit_description="",
            oid=NEW,
        )

    def list_repo_refs(self, repo_id: str) -> GitRefs:
        tags = [
            GitRefInfo(name=name, ref=f"refs/tags/{name}", target_commit=commit)
            for name, commit in self.tags.items()
        ]
        return GitRefs(branches=[], converts=[], tags=tags)

    def create_tag(self, repo_id: str, *, tag: str, revision: str) -> None:
        self.created.append((repo_id, tag, revision))


def test_the_signature_is_committed_on_top_of_the_signed_commit(tmp_path: Path) -> None:
    bundle = tmp_path / SIGNATURE_FILE
    bundle.write_bytes(b'{"mediaType": "bundle"}')
    hub = FakeHub()
    assert attach_signature(hub, MODEL, SIGNED, bundle) == NEW
    assert hub.uploads == [
        {
            "bytes": b'{"mediaType": "bundle"}',
            "path": SIGNATURE_FILE,
            "repo": MODEL,
            "revision": "main",
            "parent": SIGNED,
            "message": f"Sign manifest.json of {SIGNED}",
        }
    ]


def test_a_bundle_under_another_name_is_refused(tmp_path: Path) -> None:
    bundle = tmp_path / "manifest.json.sigstore.json"
    bundle.write_bytes(b"{}")
    hub = FakeHub()
    with pytest.raises(HubError, match=r"manifest\.json\.sigstore\.json: .* named"):
        attach_signature(hub, MODEL, SIGNED, bundle)
    assert hub.uploads == []


@pytest.mark.parametrize("revision", ["main", DEFAULT_REVISION, "a" * 39, "A" * 40, ""])
def test_only_full_commit_ids_are_accepted(revision: str) -> None:
    with pytest.raises(HubError, match="not a full 40-character commit id"):
        check_commit(revision)


def test_the_pinned_tag_is_created_at_the_commit() -> None:
    hub = FakeHub(tags={"v0.9": OTHER})
    assert tag_release(hub, MODEL, SIGNED) == SIGNED
    assert hub.created == [(MODEL, DEFAULT_REVISION, SIGNED)]


def test_a_tag_already_at_the_commit_is_left_alone() -> None:
    hub = FakeHub(tags={DEFAULT_REVISION: SIGNED})
    assert tag_release(hub, MODEL, SIGNED) == SIGNED
    assert hub.created == []


def test_a_tag_at_another_commit_is_never_moved() -> None:
    hub = FakeHub(tags={DEFAULT_REVISION: OTHER})
    with pytest.raises(HubError, match=f"already names {OTHER}, not {SIGNED}"):
        tag_release(hub, MODEL, SIGNED)
    assert hub.created == []
