# Python

## The local engine

```python
from mimir import Mimir

model = Mimir.from_pretrained("mythologic/mimir-1")
```

| Argument | Default | Does |
|---|---|---|
| `revision` | the revision this package version pins | a Hub revision |
| `device` | `auto` | `cpu`, `cuda`, or CUDA when available |
| `variant` | the variant listed for the device | `fp32` (CPU) or `fp16` (CUDA) |
| `policy` | the release policy | a custom policy from `mimir calibrate` |
| `cache_dir` | the Hub cache | where the model is stored |
| `offline` | `False` | load only from the cache, with no network access |
| `allow_unsigned` | `False` | load a local release directory that has no signature |

`mimir.Mimir` needs `mimirai[local]` (CPU) or `mimirai[local-gpu]` (CUDA); the two install the
same `onnxruntime` module, so keep one of them. `mimir doctor` reports the environment and
names a conflicting runtime.

Before a session exists, loading checks the pinned revision, the manifest's Sigstore signature
and its identity, the package versions the release supports, every file against the manifest's
SHA-256, and the graph against its contract. For offline use, fetch once with `mimir download`
and load with `offline=True` (or `HF_HUB_OFFLINE=1`).

`model.info()` reports the model, revision, variant, runtime fingerprint, certified risk levels
and input limits.

## The HTTP client

```python
from mimir import MimirClient

remote = MimirClient("https://mimir.internal", api_key="...")
remote.choose("...", "Which team?", options=["billing", "security"])
```

`MimirClient` has the same methods as `Mimir`, so code, decision tools and adapters take either.
It needs only the base install. `api_key` defaults to `$MIMIR_API_KEY`. Connection errors,
timeouts and 429, 502, 503, 504 and 529 responses are retried with exponential backoff that
honours `Retry-After`; other errors raise a `ServerResponseError` subclass keeping the status
and body.

## Async and batches

Every method has an async form: `adecide`, `adecide_many`, `achoose`, `ayes_no`, `averify`,
`arank`, `arate`, `aestimate`. `decide_many` and `adecide_many` take `(context, spec)` pairs.

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

A decision tool binds a spec to a name, so the caller supplies only the context. The HTTP
server, the MCP server and every framework adapter serve decision tools.

## Types without the engine

`from mimir import Choice, Context, ChoiceResult` loads only the data models, which depend on
Pydantic alone, so a project can build requests and read results without ONNX Runtime.
