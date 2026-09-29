# Local engine

`Mimir` runs the ONNX graph on your machine. Create one with `from_pretrained`, which
downloads the release at the revision this package version pins, verifies the manifest's
Sigstore signature and every file's SHA-256, checks the graph against its contract, and
loads it — or raises `ArtifactError` before anything is read. Safe to share across
threads. Needs `mimir-decisions[local]` (CPU) or `mimir-decisions[local-gpu]` (CUDA); the two install
the same `onnxruntime` module, so keep one of them. The fp16 (CUDA) graph ships without a
policy in this release, so `decide` raises `PolicyError` there; `decide_uncertified`
returns the raw answer.

| Argument | Default | Does |
|---|---|---|
| `model` | `Mythologic/MIMIR-1` | a Hub id or a local release directory |
| `revision` | the pinned revision | a Hub revision |
| `device` | `auto` | `cpu`, `cuda`, or CUDA when available |
| `variant` | the variant listed for the device | `fp32` (CPU) or `fp16` (CUDA) |
| `policy` | the release policy | path to a custom policy JSON from `mimir calibrate` |
| `cache_dir` | the Hub cache | where the model is stored |
| `offline` | `False` | load only from the cache, with no network access |
| `allow_unsigned` | `False` | load a local directory that has no signature |

`info()` reports the model, revision, variant, runtime fingerprint, certified risk
levels and input limits. `count_tokens(context, spec)` sizes a request before sending
it; a request over a limit raises `InputLimitError` naming the value and the limit.

::: mimir.runtime.engine.Mimir
