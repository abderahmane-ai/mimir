import pytest

from mimir.core.errors import SignatureError
from mimir.runtime.signature import RELEASE_IDENTITY, RELEASE_ISSUER, SigstoreVerifier


def test_the_release_identity_is_the_public_repository_workflow() -> None:
    assert RELEASE_IDENTITY.startswith("https://github.com/mythologic/mimir/.github/workflows/")
    assert RELEASE_ISSUER == "https://token.actions.githubusercontent.com"
    verifier = SigstoreVerifier()
    assert (verifier.identity, verifier.issuer, verifier.offline) == (
        RELEASE_IDENTITY,
        RELEASE_ISSUER,
        False,
    )


@pytest.mark.parametrize("bundle", [b"", b"not json", b'{"mediaType": "x"}'])
def test_malformed_bundles_are_refused_before_any_network_use(bundle: bytes) -> None:
    with pytest.raises(SignatureError, match=r"manifest\.json: signature verification failed"):
        SigstoreVerifier(offline=True).verify(b"{}", bundle, "manifest.json")
