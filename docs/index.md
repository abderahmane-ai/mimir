# MIMIR

MIMIR is a non-generative decision model by Mythologic. Give it a context, a question and the
options; it returns a typed answer, calibrated probabilities, the parts of the context it relied
on, and a certified verdict on whether the answer may be acted on.

`mimirai` is its Python package: a local engine on ONNX Runtime, an HTTP server, an MCP server,
and adapters for eight agent frameworks, all speaking one typed contract.

```bash
pip install "mimirai[local]"        # CPU engine
pip install "mimirai[local-gpu]"    # CUDA engine
pip install mimirai                 # data models and the HTTP client only
```

Python 3.11 or later.

| Start here | For |
|---|---|
| [Quickstart](quickstart.md) | a first decision in three lines |
| [Decisions](guide/decisions.md) | the seven decision types and what a context can hold |
| [The certificate](guide/certificate.md) | what `DECIDED`, `ABSTAINED` and `DEFERRED` promise |
| [Tool-call checks](guide/tool-call-checks.md) | gating an agent's tool calls with written rules |
| [HTTP server](surfaces/http.md), [MCP server](surfaces/mcp.md) | serving MIMIR to other processes and agents |
| [Frameworks](frameworks/openai-agents.md) | MIMIR inside an agent framework |
| [Migrating](migrating/jev.md) | moving from Jev or Laya |
