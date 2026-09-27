# HTTP server

```bash
pip install "mimirai[local,server]"
MIMIR_API_KEYS=key-one,key-two mimir serve --host 0.0.0.0 --tools tools.yaml
```

| Route | Does |
|---|---|
| `POST /v1/decide` | one certified decision: `{context, decision, risk, alpha}` |
| `POST /v1/decide/uncertified` | the model's raw answer: `{context, decision}` |
| `POST /v1/decide/batch` | up to 64 decisions in one call (`--max-batch-items`) |
| `POST /v1/tools/{name}` | a tool from `--tools`, given only `{context}` |
| `POST /v1/systemone` | Jev's request and response format ([Migrating from Jev](../migrating/jev.md)) |
| `GET /v1/models` | model, revision, variant, runtime fingerprint, certified risk levels |
| `GET /healthz`, `GET /readyz` | liveness, and readiness once the model is loaded and verified |
| `GET /metrics` | Prometheus: requests, latency, batch sizes, statuses |

The OpenAPI 3.1 document is [`openapi.json`](https://github.com/vathosai/mimir/blob/main/openapi.json).

## Tools file

```yaml
tools:
  - name: route_ticket
    description: Route a support ticket to the team that owns it.
    decision:
      type: choice
      question: Which team should handle this ticket?
      options:
        billing: "Billing: payments, refunds and invoices"
        security: "Security: account access, passwords and fraud"
```

The same file configures the MCP server.

## Behaviour

- **Loading.** The server listens at once and loads the model in the background; decisions
  answer 503 `not_ready` with `Retry-After` until it is ready, and `/readyz` turns ready.
- **Batching.** Concurrent requests of equal risk and alpha share an engine call, run when their
  tokens fill the release's batch budget or the oldest has waited 5 ms (`--batch-tokens`,
  `--batch-wait-ms`).
- **Authentication.** Keys come from `MIMIR_API_KEYS`, comma-separated; every route but the
  probes needs `Authorization: Bearer <key>`. A server without keys starts only on a loopback
  address unless given `--allow-no-auth`.
- **Limits.** Bodies over 4 MiB (`--max-body-bytes`) get 413 before they are read. Option, level
  and context-token limits are the release's.

## Errors

Every error body is `{"error": {"type", "message"}}`, and each message names the value and the
limit.

| Status | `type` |
|---|---|
| 401 | `unauthorized` |
| 404 | `not_found` |
| 409 | `no_policy` |
| 413 | `too_large` |
| 422 | `invalid_request`, `input_limit`, `risk_level`, `invalid_context` |
| 500 | `internal` |
| 503 | `not_ready` |
