"""Manifest verification against a real Sigstore bundle signed on GitHub Actions.

The artifact and its bundle are sigstore-python's own test assets, fetched at a pinned commit
and checked by SHA-256 before use.
"""

import hashlib
from typing import Final

import httpx
import pytest

from mimir.core.errors import SignatureError
from mimir.runtime.signature import RELEASE_ISSUER, SigstoreVerifier

pytestmark = pytest.mark.integration

ASSETS: Final = (
    "https://raw.githubusercontent.com/sigstore/sigstore-python/"
    "300b502ae99ebfaace124f1f4e422a6a669369cf/test/assets/"
)
ARTIFACT_SHA256: Final = "c4e92e9ecc828bef2aa7dba1de8ac983511f7532a0df11c770d39099a25cf201"
BUNDLE_SHA256: Final = "b49e7c6b452af421e5965321fc31e004d7554af3acd89f661f8c2f0f94c37e9d"
SIGNER: Final = (
    "https://github.com/trailofbits/rfc8785.py/.github/workflows/release.yml@refs/tags/v0.1.2"
)


def fetch(name: str, sha256: str) -> bytes:
    content = (
        httpx.get(ASSETS + name, timeout=30.0, follow_redirects=True).raise_for_status().content
    )
    assert hashlib.sha256(content).hexdigest() == sha256, name
    return content


@pytest.fixture(scope="module")
def signed() -> tuple[bytes, bytes]:
    return (
        fetch("bundle_v3_github.whl", ARTIFACT_SHA256),
        fetch("bundle_v3_github.whl.sigstore", BUNDLE_SHA256),
    )


def test_a_bundle_from_the_pinned_identity_verifies(signed: tuple[bytes, bytes]) -> None:
    artifact, bundle = signed
    SigstoreVerifier(identity=SIGNER, issuer=RELEASE_ISSUER).verify(artifact, bundle, "wheel")


def test_tampered_content_is_refused(signed: tuple[bytes, bytes]) -> None:
    artifact, bundle = signed
    with pytest.raises(SignatureError, match="digest mismatch"):
        SigstoreVerifier(identity=SIGNER, issuer=RELEASE_ISSUER).verify(
            artifact + b"x", bundle, "wheel"
        )


def test_another_identity_is_refused(signed: tuple[bytes, bytes]) -> None:
    artifact, bundle = signed
    with pytest.raises(SignatureError, match="SANs do not match"):
        SigstoreVerifier().verify(artifact, bundle, "wheel")
