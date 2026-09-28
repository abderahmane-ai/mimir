# Python

## The local engine

```python
from mimir import Mimir

model = Mimir.from_pretrained("Mythologic/MIMIR-1")
```

| Argument | Default | Does |
|---|---|---|
| `model` | `Mythologic/MIMIR-1` | a Hugging Face Hub id or a path to a local release directory |
| `revision` | the revision this package version pins | a specific Hub commit or tag |
| `device` | `auto` | `cpu`, `cuda`, or CUDA when a compatible GPU is available |
| `variant` | the variant listed for the device | `fp32` (CPU) or `fp16` (CUDA) |
| `policy` | the release policy | path to a custom policy JSON from `mimir calibrate` |
| `cache_dir` | the Hub cache | where the model files are stored |
| `offline` | `False` | load only from the local cache, with no network access |
| `allow_unsigned` | `False` | load a local release directory that has no Sigstore signature |

`mimir.Mimir` requires `mimirai[local]` (CPU) or `mimirai[local-gpu]` (CUDA). Both extras install the same `onnxruntime` module, so keep only one in any given environment. `mimir doctor` reports the active runtime and names any conflict.

Before loading the ONNX session, `from_pretrained` checks the pinned revision, verifies the manifest's Sigstore signature against the Mythologic release identity, verifies each file's SHA-256 against the manifest, and checks the ONNX graph against its operator allowlist and signature. Nothing is read until every check passes.

For offline use, fetch once with `mimir download` and then load with `offline=True` (or set `HF_HUB_OFFLINE=1`).

`model.info()` reports the model id, revision, variant, runtime fingerprint, certified risk levels, and input limits. `model.count_tokens(context, spec)` sizes a request before sending it.

The engine is safe to share across threads.

## The HTTP client

```python
from mimir import MimirClient

remote = MimirClient("https://mimir.internal", api_key="...")
remote.choose("...", "Which team?", options=["billing", "security"])
```

`MimirClient` implements the same interface as `Mimir` — the same methods, the same signatures — so code, decision tools, and framework adapters accept either without modification. It requires only the base install.

`api_key` defaults to the `MIMIR_API_KEY` environment variable. Connection errors, timeouts, and 429, 502, 503, 504, and 529 responses are retried with exponential backoff that honours `Retry-After`. Any other error response raises a `ServerResponseError` subclass that keeps the HTTP status and the response body.

Use `MimirClient` as a context manager to ensure the underlying connection pool is released:

```python
with MimirClient("https://mimir.internal") as remote:
    result = remote.choose("...", "Which team?", options=["billing", "security"])
```

## Async and batches

Every method has an async counterpart: `adecide`, `adecide_many`, `achoose`, `ayes_no`, `averify`, `arank`, `arate`, `aestimate`.

`decide_many` and `adecide_many` take a list of `(context, spec)` pairs and batch them by token length. On the local engine, requests that share the same risk level and alpha are grouped into a single engine call.

## Decision tools

```python
from mimir import Choice

route_ticket = model.tool(
    "route_ticket",
    Choice("Which team should handle this ticket?", options=["billing", "security"]),
    description="Route a support ticket to the team that owns it.",
)
route_ticket("My card was charged twice")
route_ticket.input_schema, route_ticket.output_schema
```

A decision tool binds a spec to a name so the caller supplies only the context. The HTTP server, the MCP server, and every framework adapter expose decision tools. `input_schema` and `output_schema` are JSON Schema objects describing the tool's expected input and output.

## Types without the engine

`from mimir import Choice, Context, ChoiceResult` loads only the data models, which depend on Pydantic alone. A project that constructs requests or reads results without running the model can depend on the base install and never touch ONNX Runtime.
