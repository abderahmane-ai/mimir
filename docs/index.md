# MIMIR

MIMIR is a non-generative decision model by Mythologic. Give it a context, a question, and the options; it returns a typed answer, calibrated probabilities, the parts of the context it relied on, and a certified verdict on whether the answer may be acted on.

`mimir-decisions` is its Python package: a local engine on ONNX Runtime, an HTTP server, an MCP server, and adapters for eight agent frameworks, all sharing one typed contract.

```bash
pip install "mimir-decisions[local]"        # CPU engine
pip install "mimir-decisions[local-gpu]"    # CUDA engine
pip install mimir-decisions                  # data models and HTTP client only
```

Python 3.11 or later.

---

## Where to start

| Page | For |
|---|---|
| [Quickstart](quickstart.md) | a first decision in three lines |
| [Use cases](guide/usecases.md) | six runnable scripts: route, verify, rank, rate, estimate, gate |
| [Decisions](guide/decisions.md) | the seven decision types and what a context can contain |
| [The certificate](guide/certificate.md) | what `DECIDED`, `ABSTAINED`, and `DEFERRED` promise and why |
| [Tool-call checks](guide/tool-call-checks.md) | gating an agent's tool calls with rules you write |
| [HTTP server](surfaces/http.md) | serving MIMIR over HTTP with batching and Prometheus metrics |
| [MCP server](surfaces/mcp.md) | exposing MIMIR as MCP tools to any agent or client |
| [Frameworks](frameworks/openai-agents.md) | native adapters for eight agent SDKs |
| [Migrating from Jev](migrating/jev.md), [from Laya](migrating/laya.md) | drop-in compatibility paths |
