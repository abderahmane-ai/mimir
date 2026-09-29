# Troubleshooting

## `MissingExtraError`: import requires an extra that is not installed

You imported only the data models but called something that requires the engine, the server, or a framework adapter. The exception message names the missing extra:

```bash
pip install "mimir-decisions[local]"            # the local engine (CPU and CUDA)
pip install "mimir-decisions[local,server]"     # engine plus the HTTP server
pip install "mimir-decisions[local,mcp]"        # engine plus the MCP server
```

## The model will not load

`ArtifactError` (and its subclasses `IntegrityError`, `SignatureError`, `FormatVersionError`) means a download, signature, or integrity check failed before any model file was read. Run `mimir doctor --verify`: it reports the environment, loads the model, and runs the equivalence check.

`EquivalenceError` means the hardware is not listed in the certificate, so the first load ran the release's equivalence set and found a decision that differed. Either run `mimir calibrate` on that hardware's decisions, or confirm that your hardware is listed in the certificate.

`PolicyMismatchError` means a custom policy was certified on different weights than the ones being loaded. Policies are bound to the exact weights they were made on — run `mimir calibrate` where you run the model.

`ValueError: device 'cuda' needs a CUDA torch build with a visible GPU` means torch cannot see a GPU. Install a CUDA-enabled torch, or pass `device="cpu"`.

## A decision defers and you expected an answer

A deferral means the answer came in below the operating floor, and the answer is still there. In `standard` mode there is no floor: if you see `deferred`, you asked for `threshold` or `certified` mode. Either lower `min_confidence`, pass a higher `risk`, or decide in `standard` mode and treat `confidence` as advisory.

See [The certificate](certificate.md) for the modes and how to recalibrate.

## The server answers 503 `not_ready`

The server starts listening immediately and loads the model in the background. Decision endpoints answer 503 until the model is ready. Poll `GET /readyz`, or configure your load balancer's readiness probe to wait for it before routing traffic.

## An MCP tool call fails

Invalid arguments, a model still loading, and engine failures are MCP tool errors that name the cause. A deferred decision is not a tool error — it is a normal result telling the agent to review the answer. If every call defers, the generic tools are running in `threshold` or `certified` mode: switch them to `standard` (the default), or lower the floor.

## Still stuck

`mimir doctor` reports the full environment. Every exception the package raises is a subclass of `MimirError`; the full hierarchy is in [Errors](../reference/errors.md).
