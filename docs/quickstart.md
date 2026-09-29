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

On a machine with a CUDA GPU, `device="auto"` selects the fp16 graph, which ships without a policy in this release: `decide` raises `PolicyError` there. Pass `device="cpu"` for certified decisions, or use `decide_uncertified` for the raw answer.

```python
result.status            # Status.DECIDED, Status.ABSTAINED or Status.DEFERRED
result.answer            # "billing", or None when no option applies
result.probabilities     # calibrated probability of each option id
result.relevant_context  # the parts of the context the answer relied on, most relevant first
result.certificate       # the certified threshold the decision was checked against
```

`answer` is what the model thinks. `status` is what you may do with it:

- `DECIDED` — act on `answer`; it is certified at the requested risk level.
- `ABSTAINED` — no listed option applies, and that conclusion is certified.
- `DEFERRED` — do not act; escalate. `result.deferral.reason` says why: `below_threshold`, `out_of_distribution`, or `no_certified_threshold`.

## Next

- [Decisions](guide/decisions.md) — yes/no questions, claim verification, rankings, ratings, and estimates, over passages, tables, and JSON.
- [The certificate](guide/certificate.md) — what the risk level guarantees, and how to certify thresholds on your own data.
- [Python](surfaces/python.md) — batching, async, offline use, and the HTTP client.
