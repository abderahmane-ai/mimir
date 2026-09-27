# Migrating from Jev

Moving from Jev is a change of endpoint URL. `mimir serve` answers Jev's `/v1/systemone`
requests in Jev's response shape:

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

| Jev question | MIMIR decision |
|---|---|
| `choice` | `Choice` over the criteria keys, each key's description read as the option |
| `noul` | `Choice` over `false` and `true` |
| `score`, 3 or more levels | `Rate` over the levels in order |
| `score`, 2 levels | `Choice` over the two levels |

Answers carry calibrated probabilities at risk 1%, and `confidence` as Jev defines it. `usage`
reports the input tokens MIMIR encoded and no output tokens, since MIMIR generates none.

To go further than Jev's format, move a question at a time to [`/v1/decide`](../surfaces/http.md):
it returns the status, the certificate and the relevant context that Jev's answers have no
field for. In Python, `mimir.compat.systemone.v1` converts both ways (`translate` and
`respond`).
