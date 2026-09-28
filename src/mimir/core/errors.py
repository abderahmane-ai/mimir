"""Exceptions raised by mimir. All derive from `MimirError`."""


class MimirError(Exception):
    """Base class for all mimir exceptions."""


class ContextError(MimirError, ValueError):
    """The context contains an unsupported or non-finite value."""


class RiskLevelError(MimirError, ValueError):
    """The requested risk level is not certified by the loaded policy."""


class InputLimitError(MimirError, ValueError):
    """The input exceeds a limit the release was tested at."""

    def __init__(self, limit: str, value: int, maximum: int) -> None:
        super().__init__(f"{limit} is {value}; the release is tested up to {maximum}")
        self.limit = limit
        self.value = value
        self.maximum = maximum


class MissingExtraError(MimirError, ImportError):
    """A feature requires an optional extra that is not installed."""

    def __init__(self, feature: str, extra: str, missing: str) -> None:
        super().__init__(
            f"{feature} requires the '{extra}' extra ({missing} is not installed): "
            f"pip install 'mimir-decisions[{extra}]'"
        )
        self.extra = extra


class ArtifactError(MimirError):
    """The model artifact cannot be fetched or loaded."""


class IntegrityError(ArtifactError):
    """A file is missing, not in the manifest, or has the wrong SHA-256."""


class SignatureError(ArtifactError):
    """The manifest signature is missing, invalid or from an untrusted identity."""


class GraphContractError(ArtifactError):
    """The graph violates its contract: opset, operators, external data or signature."""


class FormatVersionError(ArtifactError):
    """The artifact or policy format is not supported by this package version."""


class UncertifiedRuntimeError(MimirError):
    """The ONNX Runtime version, provider or provider options are not certified."""


class EquivalenceError(MimirError):
    """Decisions on the equivalence set differ from the certified ones."""


class PolicyError(MimirError):
    """The policy file is malformed or inconsistent with its arrays."""


class PolicyMismatchError(PolicyError):
    """The policy was certified for a different model, graph or runtime."""


class ClientError(MimirError):
    """A request to a MIMIR server failed."""


class ServerConnectionError(ClientError):
    """The server could not be reached or closed the connection."""


class ServerTimeoutError(ClientError):
    """The server did not respond within the timeout."""


class ServerResponseError(ClientError):
    """The server returned an error status. `status` and `body` are kept."""

    def __init__(self, status: int, body: str, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.body = body


class InvalidRequestError(ServerResponseError):
    """HTTP 400 or 422: the request failed validation."""


class AuthenticationError(ServerResponseError):
    """HTTP 401 or 403: the API key is missing, invalid or not permitted."""


class NotFoundError(ServerResponseError):
    """HTTP 404: unknown route or tool."""


class RateLimitError(ServerResponseError):
    """HTTP 429: rate limit exceeded."""


class ServerError(ServerResponseError):
    """HTTP 5xx: server failure or overload."""
