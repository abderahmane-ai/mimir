# Quickstart

```bash
pip install "mimirai[local]"
```

```python
from mimir import Mimir

model = Mimir.from_pretrained("mythologic/mimir-1")
result = model.choose(
    "My card was charged twice for the same order.",
    "Which team should handle this ticket?",
    options={"billing": "Billing: payments, refunds", "security": "Security: account access"},
)
```

The first call downloads the model from the Hugging Face Hub at the revision this package
version pins, verifies its signature and every file, and loads it.

```python
result.status            # Status.DECIDED, Status.ABSTAINED or Status.DEFERRED
result.answer            # "billing", or None when no option applies
result.probabilities     # calibrated probability of each option id
result.relevant_context  # the parts of the context the answer relied on, most relevant first
result.certificate       # the certified threshold the decision was checked against
```

`answer` is what the model thinks. `status` is what you may do with it:

- `DECIDED`: act on `answer`; it is certified at the requested risk.
- `ABSTAINED`: no listed option applies, and that is certified.
- `DEFERRED`: do not act; escalate. `result.deferral.reason` says why.

## Next

- [Decisions](guide/decisions.md): yes/no questions, claims, rankings, ratings and estimates, over
  passages, tables and JSON.
- [The certificate](guide/certificate.md): what the risk level means, and how to certify on your
  own data.
- [Python](surfaces/python.md): batching, async, offline use and the HTTP client.
