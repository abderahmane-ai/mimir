# Troubleshooting

## `MissingExtraError`: import requires an extra that is not installed

You imported only the data models but called something that requires the engine, the server, or a framework adapter. The exception message names the missing extra:

```bash
pip install "mimir-decisions[local]"            # the local engine (CPU)
pip install "mimir-decisions[local-gpu]"        # the local engine (CUDA)
pip install "mimir-decisions[local,server]"     # engine plus the HTTP server
pip install "mimir-decisions[local,mcp]"        # engine plus the MCP server
```

## The model will not load

`ArtifactError` (and its subclasses `IntegrityError`, `SignatureError`, `GraphContractError`, `FormatVersionError`) means a download, signature, or integrity check failed before any model file was read. Run `mimir doctor --verify`: it reports the environment, loads the model, and runs the equivalence check.

`UncertifiedRuntimeError` means the runtime does not match the policy's certified configuration — most commonly the wrong variant for the device (`fp32` on CPU, `fp16` on CUDA). See the `device` and `variant` arguments in [Python](../surfaces/python.md#the-local-engine). If the CUDA provider is listed but cannot open a session, `device="auto"` falls back to the CPU release; `device="cuda"` raises instead.

`EquivalenceError` means the hardware is not listed in the certificate, so the first load ran the release's equivalence set and found a decision that differed. Either run `mimir calibrate` on that hardware's decisions, or confirm that the CPU execution provider is listed in the certificate and route the load there.

`PolicyMismatchError` means a custom policy was certified on a different model, runtime, or hardware than the one being loaded. Policies are bound to the exact configuration they were made on — run `mimir calibrate` where you run the model.

## A certified call raises `PolicyError`

The loaded variant has no policy. In this release that is the fp16 (CUDA) graph, reached only with an explicit `device="cuda"` or `variant`: use `decide_uncertified` for the raw answer, or pass `device="cpu"` (or leave the default `auto`) to load the fp32 graph with the release policy. A server in the same state answers 409 `no_policy` on the certified routes.

## The server answers 503 `not_ready`

The server starts listening immediately and loads the model in the background. Decision endpoints answer 503 until the model is ready. Poll `GET /readyz`, or configure your load balancer's readiness probe to wait for it before routing traffic.

## A decision defers and you expected an answer

Read `result.deferral.reason`:

- `below_threshold` — confidence missed the certified threshold. The context may be ambiguous, or the risk level you asked for may be too strict. Try lowering the risk, or calibrate on your own data closer to your distribution.
- `out_of_distribution` — the context is too unlike the data the thresholds were certified on. Either bring the inputs closer to the training distribution or recalibrate with `mimir calibrate` on your own data.
- `no_certified_threshold` — nothing is certified for this decision type at this risk level. Check `model.info().risk_levels` for what is available, or run `mimir calibrate`.

See [The certificate](certificate.md) for a full explanation of deferral and how to recalibrate.

## An MCP tool call fails

Invalid arguments, a model still loading, and engine failures are MCP tool errors that name the cause. A deferred decision is not a tool error — it is a normal result telling the agent to escalate. If every call defers, the problem is in the request, not the transport.

## Still stuck

`mimir doctor` reports the full environment and names any conflicting ONNX Runtime installation. Every exception the package raises is a subclass of `MimirError`; the full hierarchy is in [Errors](../reference/errors.md).
