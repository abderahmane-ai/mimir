"""The release workflows' writes to the Hugging Face Hub.

`attach-signature` commits a manifest's Sigstore bundle on top of the commit whose manifest was
signed, and fails if the branch has moved past it. `tag` points the revision this package pins
(`DEFAULT_REVISION`) at a commit, and fails if that tag already names another commit. `retag`
moves that tag onto a signed commit, for a release that changes no model bytes, such as a
licence or card update. All print the commit they leave the release at.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Final, Protocol

from huggingface_hub import HfApi
from huggingface_hub.hf_api import CommitInfo, GitRefs

from mimir.runtime.artifact import DEFAULT_MODEL, DEFAULT_REVISION
from mimir.runtime.release import SIGNATURE_FILE

BRANCH: Final = "main"
COMMIT: Final = re.compile(r"[0-9a-f]{40}")


class HubError(RuntimeError):
    pass


class Hub(Protocol):
    def upload_file(
        self,
        *,
        path_or_fileobj: bytes,
        path_in_repo: str,
        repo_id: str,
        revision: str,
        commit_message: str,
        parent_commit: str,
    ) -> CommitInfo: ...

    def list_repo_refs(self, repo_id: str) -> GitRefs: ...

    def create_tag(self, repo_id: str, *, tag: str, revision: str) -> None: ...

    def delete_tag(self, repo_id: str, *, tag: str) -> None: ...


def check_commit(revision: str) -> str:
    if COMMIT.fullmatch(revision) is None:
        message = f"revision {revision!r} is not a full 40-character commit id"
        raise HubError(message)
    return revision


def attach_signature(hub: Hub, model: str, signed: str, bundle: Path) -> str:
    """Commit `bundle` as the manifest's signature on top of `signed`; return the new commit."""
    if bundle.name != SIGNATURE_FILE:
        message = f"{bundle}: the signature must be named {SIGNATURE_FILE}"
        raise HubError(message)
    info = hub.upload_file(
        path_or_fileobj=bundle.read_bytes(),
        path_in_repo=SIGNATURE_FILE,
        repo_id=model,
        revision=BRANCH,
        commit_message=f"Sign manifest.json of {check_commit(signed)}",
        parent_commit=signed,
    )
    return info.oid


def tag_release(hub: Hub, model: str, commit: str, tag: str = DEFAULT_REVISION) -> str:
    """Point `tag` at `commit`, or leave it if it already does; return the commit."""
    check_commit(commit)
    existing = {ref.name: ref.target_commit for ref in hub.list_repo_refs(model).tags}
    if tag not in existing:
        hub.create_tag(model, tag=tag, revision=commit)
    elif existing[tag] != commit:
        message = f"{model}: tag {tag} already names {existing[tag]}, not {commit}"
        raise HubError(message)
    return commit


def retag_release(hub: Hub, model: str, commit: str, tag: str = DEFAULT_REVISION) -> str:
    """Move `tag` onto `commit`, or leave it if it already points there; return the commit."""
    check_commit(commit)
    existing = {ref.name: ref.target_commit for ref in hub.list_repo_refs(model).tags}
    if existing.get(tag) == commit:
        return commit
    if tag in existing:
        hub.delete_tag(model, tag=tag)
    hub.create_tag(model, tag=tag, revision=commit)
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a MIMIR release to the Hub.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub repository id.")
    commands = parser.add_subparsers(dest="command", required=True)
    attach = commands.add_parser("attach-signature", help="Commit the manifest's signature.")
    attach.add_argument("--revision", required=True, help="The commit whose manifest was signed.")
    attach.add_argument("--bundle", type=Path, required=True, help=f"The {SIGNATURE_FILE} file.")
    tag = commands.add_parser("tag", help=f"Point {DEFAULT_REVISION} at a signed commit.")
    tag.add_argument("--revision", required=True, help="The signed commit.")
    retag = commands.add_parser("retag", help=f"Move {DEFAULT_REVISION} onto a signed commit.")
    retag.add_argument("--revision", required=True, help="The signed commit.")
    arguments = parser.parse_args()
    hub = HfApi()
    try:
        if arguments.command == "attach-signature":
            commit = attach_signature(hub, arguments.model, arguments.revision, arguments.bundle)
        elif arguments.command == "tag":
            commit = tag_release(hub, arguments.model, arguments.revision)
        else:
            commit = retag_release(hub, arguments.model, arguments.revision)
    except HubError as error:
        sys.stderr.write(f"{error}\n")
        sys.exit(1)
    sys.stdout.write(f"{commit}\n")


if __name__ == "__main__":
    main()
