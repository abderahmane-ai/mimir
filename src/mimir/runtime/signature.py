"""Sigstore verification of the release manifest.

The manifest must be signed by the release workflow of `abderahmane-ai/mimir` on GitHub Actions.
Bundles are verified against Sigstore's production trust root, refreshed over TUF unless
`offline` is set.
"""

from dataclasses import dataclass
from typing import Final, Protocol

from sigstore.errors import Error as SigstoreError
from sigstore.models import Bundle
from sigstore.verify import Verifier
from sigstore.verify.policy import Identity

from mimir.core.errors import SignatureError

RELEASE_IDENTITY: Final = (
    "https://github.com/abderahmane-ai/mimir/.github/workflows/sign-model.yml@refs/heads/main"
)
RELEASE_ISSUER: Final = "https://token.actions.githubusercontent.com"


class ManifestVerifier(Protocol):
    def verify(self, manifest: bytes, bundle: bytes, where: str) -> None:
        """Raise `SignatureError` unless `bundle` is a valid signature of `manifest`."""


@dataclass(frozen=True, slots=True)
class SigstoreVerifier:
    """Verify Sigstore bundles for one signing identity and OIDC issuer."""

    identity: str = RELEASE_IDENTITY
    issuer: str = RELEASE_ISSUER
    offline: bool = False

    def verify(self, manifest: bytes, bundle: bytes, where: str) -> None:
        try:
            parsed = Bundle.from_json(bundle)
            verifier = Verifier.production(offline=self.offline)
            verifier.verify_artifact(
                manifest, parsed, Identity(identity=self.identity, issuer=self.issuer)
            )
        except SigstoreError as error:
            message = f"{where}: signature verification failed for {self.identity}: {error}"
            raise SignatureError(message) from error
