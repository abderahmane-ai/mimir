# Changelog

All notable changes to `mimir-decisions` are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

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
