# HTTP server

```bash
pip install "mimir-decisions[local,server]"
MIMIR_API_KEYS=key-one,key-two mimir serve --host 0.0.0.0 --tools tools.yaml
```

## Routes

| Route | Does |
|---|---|
| `POST /v1/decide` | one decision: `{context, decision, mode, min_confidence, risk, alpha}` |
| `POST /v1/decide/uncertified` | the model's raw answer with no policy applied: `{context, decision}` |
| `POST /v1/decide/batch` | up to 64 decisions in one call (`--max-batch-items`) |
| `POST /v1/tools/{name}` | a named tool from `--tools`, given only `{context}` |
| `POST /v1/systemone` | Jev's request and response format ([Migrating from Jev](../migrating/jev.md)) |
| `GET /v1/models` | model id, revision, variant, runtime fingerprint, certified risk levels |
| `GET /healthz` | liveness — always 200 once the process is up |
| `GET /readyz` | readiness — 200 once the model has loaded and been verified |
| `GET /metrics` | Prometheus: request counts, latency histograms, batch sizes, status codes |

The OpenAPI 3.1 document is [`openapi.json`](https://github.com/abderahmane-ai/mimir/blob/main/openapi.json).

## Tools file

```yaml
tools:
  - name: route_ticket
    description: Route a support ticket to the team that owns it.
    decision:
      type: choice
      question: Which department should handle this request?
      options:
        billing: "Billing: invoices, payments, refunds"
        technical: "Technical: bugs, outages, system errors"
        sales: "Sales: pricing, new contracts"
        other: "Other: everything else"
```

The same file configures the MCP server, so both servers expose the same tools with no duplication.

## Behaviour

**Loading.** The server binds and starts accepting connections immediately, then loads the model in the background. Decision endpoints answer 503 `not_ready` with a `Retry-After` header until the model is ready. `/readyz` turns 200 at the same moment.

**Batching.** Concurrent requests with the same mode, floor, risk level, and alpha are batched into a single engine call. A batch is dispatched when its accumulated tokens fill the release's batch budget or when the oldest request has waited 5 ms. Both limits are tunable with `--batch-tokens` and `--batch-wait-ms`.

**Authentication.** Keys are read from `MIMIR_API_KEYS`, comma-separated. Every route except `/healthz` and `/readyz` requires `Authorization: Bearer <key>`. A server started without keys will only bind a loopback address (`127.0.0.1`, `::1`) unless started with `--allow-no-auth`.

**Body limit.** Requests over 4 MiB (`--max-body-bytes`) are rejected with 413 before the body is read.

## Errors

Every error response body is `{"error": {"type": "...", "message": "..."}}`. The message names the specific value and limit that caused the error.

| Status | `type` |
|---|---|
| 401 | `unauthorized` |
| 404 | `not_found` |
| 409 | `no_policy` |
| 413 | `too_large` |
| 422 | `invalid_request`, `input_limit`, `risk_level`, `invalid_context` |
| 500 | `internal` |
| 503 | `not_ready` |
