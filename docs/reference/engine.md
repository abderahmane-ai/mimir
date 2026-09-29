# Local engine

`Mimir` runs the model on your machine with Torch. Create one with `from_pretrained`, which
downloads the release at the revision this package version pins, verifies the manifest's
Sigstore signature and every file's SHA-256, and loads the weights — or raises `ArtifactError` before anything is read. Safe to share across
threads. Needs `mimir-decisions[local]`; the same install serves the CPU and CUDA. `decide_uncertified`
returns the raw answer.

| Argument | Default | Does |
|---|---|---|
| `model` | `Mythologic/MIMIR-1` | a Hub id or a local release directory |
| `revision` | the pinned revision | a Hub revision |
| `device` | `auto` | `cpu`, `cuda`, or CUDA when a GPU is visible |
| `variant` | the variant listed for the device | `fp32` |
| `policy` | the release policy | path to a custom policy JSON from `mimir calibrate` |
| `cache_dir` | the Hub cache | where the model is stored |
| `offline` | `False` | load only from the cache, with no network access |
| `allow_unsigned` | `False` | load a local directory that has no signature |

`info()` reports the model, revision, variant, runtime fingerprint, certified risk
levels and input limits. `count_tokens(context, spec)` sizes a request before sending
it; a request over a limit raises `InputLimitError` naming the value and the limit.

::: mimir.runtime.engine.Mimir
