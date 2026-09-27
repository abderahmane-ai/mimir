# Changelog

## 1.0.0

The first release, for `vathosai/mimir-1`.

- `Mimir`, the local engine on ONNX Runtime: `fp32` on CPU, `fp16` on CUDA, each with its own
  certificate. Seven decision types over passages, tables and JSON fields, with calibrated
  probabilities, abstention, relevant context and certified deferral.
- `MimirClient`, the same interface over HTTP.
- `mimir serve`: the HTTP API with batching, bearer keys, Prometheus metrics and Jev's
  `/v1/systemone` format.
- `mimir mcp`: configured and generic decision tools over stdio and Streamable HTTP.
- Tool-call checks, and adapters for the OpenAI Agents SDK, LangChain, PydanticAI, CrewAI,
  Google ADK, Microsoft Agent Framework, LlamaIndex and smolagents.
- `mimir calibrate` and `mimir bench`: certify thresholds on your own labelled decisions.
- `mimir.compat.systemone.v1` and `mimir.compat.laya.v1` for migrating from Jev and Laya.
