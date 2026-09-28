# Migrating from Jev

Moving from Jev is primarily a change of endpoint URL. `mimir serve` answers Jev's `/v1/systemone` requests in Jev's response shape, so your existing callers need no code changes to start routing traffic to MIMIR.

```bash
curl https://mimir.internal/v1/systemone \
  -H "Authorization: Bearer $MIMIR_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations"}
      }
    }
  }'
```

## Question mapping

| Jev question type | MIMIR decision |
|---|---|
| `choice` | `Choice` over the criteria keys, each key's description read as the option label |
| `noul` | `Choice` over `false` and `true` |
| `score`, 3 or more levels | `Rate` over the levels in order |
| `score`, 2 levels | `Choice` over the two levels |

Answers carry calibrated probabilities at 1% risk, and `confidence` as Jev defines it. `usage` reports the input tokens MIMIR encoded; output tokens are zero, since MIMIR generates none.

## Going further

`/v1/systemone` gives you the answer and the probabilities. Moving a question at a time to [`/v1/decide`](../surfaces/http.md) also gives you the decision status, the deferral reason, the certificate, and the relevant context — none of which Jev's format exposes.

In Python, `mimir.compat.systemone.v1` converts in both directions:

```python
from mimir.compat.systemone.v1 import translate, respond

# Convert a Jev request to a MIMIR DecideRequest
mimir_request = translate(jev_request)

# Convert a MIMIR result back to Jev's response shape
jev_response = respond(mimir_result, jev_request)
```
