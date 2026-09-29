# Quickstart

```bash
pip install "mimir-decisions[local]"
```

```python
from mimir import Mimir

model = Mimir.from_pretrained("Mythologic/MIMIR-1")
result = model.choose(
    "My card was charged twice for the same order.",
    "Which team should handle this ticket?",
    options={"billing": "Billing: payments, refunds", "security": "Security: account access"},
)
```

The first call downloads the model from the Hugging Face Hub at the revision this package version pins, verifies the manifest's Sigstore signature, checks every file against the SHA-256 in the manifest, and loads it. Nothing is read until every check passes.

On a machine with a CUDA GPU, `device="auto"` serves the GPU; pass `device="cpu"` to stay on the CPU. The same weights serve both.

```python
result.status            # Status.DECIDED, Status.ABSTAINED or Status.DEFERRED
result.answer            # "billing", or None when no option applies
result.probabilities     # calibrated probability of each option id
result.relevant_context  # the parts of the context the answer relied on, most relevant first
result.certificate       # the evidence, when the answer is certified
```

`answer` is always what the model thinks. `status` says whether it cleared the operating floor:

- `DECIDED` — act on `answer`.
- `ABSTAINED` — no listed option applies.
- `DEFERRED` — the answer came in below the floor (`threshold` or `certified` mode); have a person review it. The answer is still there.

## Next

- [Decisions](guide/decisions.md) — yes/no questions, claim verification, rankings, ratings, and estimates, over passages, tables, and JSON.
- [The certificate](guide/certificate.md) — what the risk level guarantees, and how to certify thresholds on your own data.
- [Python](surfaces/python.md) — batching, async, offline use, and the HTTP client.
