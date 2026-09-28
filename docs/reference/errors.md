# Errors

Every exception the package raises is a subclass of `MimirError`, so a single handler can cover the entire surface. Subclasses name exactly what went wrong:

**Input errors** — raised when the request itself is malformed:

- `ContextError` — the context is not a valid type or structure.
- `RiskLevelError` — the requested risk level is not a valid probability.
- `InputLimitError` — a request exceeds the model's limits; the message names the value and the limit.

**Artifact errors** — raised before anything is read, when verification fails:

- `IntegrityError` — a file's SHA-256 does not match the manifest.
- `SignatureError` — the manifest's Sigstore signature is invalid or does not match the expected identity.
- `GraphContractError` — the ONNX graph uses operators outside the allowlist.
- `FormatVersionError` — the release format is not supported by this package version.

**Policy errors** — raised when the loaded policy is inconsistent with the runtime or hardware:

- `PolicyError` — the policy file is malformed.
- `PolicyMismatchError` — a custom policy was made for a different model, runtime, or hardware.
- `UncertifiedRuntimeError` — the current runtime does not match the policy's certified configuration.
- `EquivalenceError` — the hardware is not listed in the certificate and a decision differed in the equivalence check.

**Server errors** — raised by `MimirClient` on network or server failures:

- `ServerConnectionError` — could not connect.
- `ServerTimeoutError` — the request timed out after retries.
- `ServerResponseError` — the server responded with an error status; subclasses preserve the status and body: `InvalidRequestError`, `AuthenticationError`, `NotFoundError`, `RateLimitError`, `ServerError`.

**Install errors**:

- `MissingExtraError` — the call requires an extra that is not installed; the message names it.

::: mimir.core.errors
