# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.x (latest) | ✅ |

Security patches are issued as patch releases on the latest minor. We do not backport fixes to older minor versions.

## Reporting a vulnerability

**Do not open a public GitHub issue for a security vulnerability.**

Report it privately using GitHub's [private vulnerability reporting](https://github.com/Mythologic/mimir/security/advisories/new).

Include as much of the following as you have:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a minimal proof of concept.
- The version(s) affected.
- Any suggested mitigations.

### What to expect

- **Acknowledgement** within 2 business days.
- **Initial triage** within 5 business days.
- **Coordinated disclosure**: we will work with you to agree on a disclosure timeline, typically 90 days from the initial report or sooner if a fix is ready.

We treat responsible disclosure seriously and will credit reporters in the release notes unless you prefer to remain anonymous.

## Supply chain

- Model weights are loaded from a pinned Hugging Face revision. The manifest's Sigstore signature is verified against the Mythologic release identity before any file is read.
- Container images are signed with Sigstore by the release workflow and can be verified with `cosign`.
- PyPI packages are published via OIDC trusted publishing with build provenance attestations.
- No pickle is used anywhere in the model loading path.

If you find a way to bypass any of these controls, please report it.
