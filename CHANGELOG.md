# Changelog

All notable changes to `mimir-decisions` are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## 1.1.0

### Added

- **Operating modes.** Every decision method, tool, HTTP route, and MCP server takes `mode`: `standard` answers every request and never defers; `threshold` defers answers below `min_confidence` while keeping the answer; `certified` defers answers below the release's threshold at `risk` and attaches the certificate when the answer passes. `ToolCallCheck` runs in `threshold` mode at 0.5 by default.
- **`actionable` and `certified` on every result.** `actionable` is true for `decided` and `abstained`; `certified` is true when the answer passes the policy's threshold at the requested risk.
- **Torch engine.** The local engine runs the model with Torch on safetensors weights: the same install serves CPU and CUDA, replacing the `local-gpu` extra. The release ships one fp32 weights file with its policy; the policy fingerprint names the weights, Torch version, device, and hardware.
- **`mimir bench`** reports the certified share alongside accuracy, coverage, and realised risk.

### Fixed

- The Vercel AI example pins the MCP handshake to the legacy era: the client's modern discovery times out while a cold server loads its weights, and its fallback then reuses the poisoned connection.

## 1.0.2

### Fixed

- `device="auto"` serves the best configuration that can decide: the device's variant when it
  carries a policy, else the certified CPU release. A CUDA machine whose GPU variant has no
  policy, or whose CUDA provider cannot open a session, loads fp32 on CPU instead of failing;
  an explicit `device` or `variant` stays strict and raises.

## 1.0.1

### Fixed

- `device="auto"` falls back to the CPU release when the CUDA execution provider is listed but
  cannot open a session, instead of failing the load; `device="cuda"` still raises, naming the
  cause and the fix.

## 1.0.0

Initial release, targeting `Mythologic/MIMIR-1`.

### Added

- **`Mimir`** — local inference engine on ONNX Runtime. `fp32` on CPU with the shipped policy; `fp16` on CUDA ships without a policy in this release. Loads from the Hugging Face Hub at a pinned revision; verifies the Sigstore manifest signature, every file's SHA-256, and the ONNX graph against its operator allowlist before reading anything.
- **Seven decision types** over passages, tables, and JSON fields: `Choice`, `MultiChoice`, `YesNo`, `Verify`, `Rank`, `Rate`, `Estimate`. Every result carries calibrated probabilities, the relevant context slices, and a certified deferral signal.
- **`MimirClient`** — the same interface over HTTP, with exponential backoff and `Retry-After` support. No engine dependency; works from the base install.
- **`mimir serve`** — FastAPI HTTP server with request batching, bearer authentication, Prometheus metrics, and Jev's `/v1/systemone` compat endpoint.
- **`mimir mcp`** — MCP server over stdio and Streamable HTTP. Configured tools expose only a context argument; generic tools expose the full decision surface.
- **Tool-call checks** — policy-governed gating of agent tool calls, returning `ALLOW`, `DENY`, or `ESCALATE` with a one-sentence reason.
- **Framework adapters** for OpenAI Agents SDK, LangChain, PydanticAI, CrewAI, Google ADK, Microsoft Agent Framework, LlamaIndex, and smolagents. Each adapter wires decision tools and, where the framework provides a hook, tool-call checks.
- **`mimir calibrate`** — certify confidence thresholds on your own labelled decisions and write a custom policy.
- **`mimir bench`** — report accuracy, coverage, and realised risk per decision type on held-out labelled data.
- **`mimir download`**, **`mimir doctor`**, **`mimir schema`** — offline model fetch, environment diagnostics with optional equivalence verification, and JSON Schema export.
- **`mimir.compat.systemone.v1`** — request and response translation for Jev `/v1/systemone`.
- **`mimir.compat.laya.v1`** — `load(...).predict(state, questions)` in Laya 0.3.20's shape.
- **Container images** — `ghcr.io/abderahmane-ai/mimir:{version}-cpu` and `:{version}-cuda`. Images carry the runtime only; the model is downloaded, verified, and cached on first start. Images are signed with Sigstore by the release workflow.
