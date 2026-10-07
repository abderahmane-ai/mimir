# Quickstart

```bash
pip install "mimir-decisions[local]"
```

```python
from mimir import Context, Field, Mimir, Passage

model = Mimir.from_pretrained("Mythologic/MIMIR-1")

# Context pairs ticket prose with typed structured fields (amounts, IDs, metadata)
context = Context(
    passages=[
        Passage(
            title="Ticket #4091",
            text="Hi, we were billed twice ($2,400 total) on invoice INV-8821. Please refund the $1,200 duplicate today or we will cancel our plan.",
        )
    ],
    fields=Field.from_json({
        "invoice_id": "INV-8821",
        "duplicate_amount": 1200,
        "customer_plan": "enterprise",
    }),
)

# Choose with risk dial: 0.05 for high-throughput agents, 0.01 for mission-critical SLA
result = model.choose(
    context,
    "Which department should handle this request?",
    options={
        "billing": "Billing: invoices, payments, refunds",
        "technical": "Technical: bugs, outages, system errors",
        "sales": "Sales: pricing, new contracts",
        "other": "Other: everything else",
    },
    risk=0.05,  # 5% risk floor (95% SLA) for high-throughput automated execution
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
- `ABSTAINED` — no listed option applies (clean OOD rejection).
- `DEFERRED` — the answer came in below the floor (`threshold` or `certified` mode); have a person review it. The answer is still there.

### The Risk Dial: Choosing Your Operating SLA

The release certifies thresholds across four finite-sample risk levels (`model.info().risk_levels = (0.005, 0.01, 0.02, 0.05)`):

- **`risk=0.01` (99.0% SLA)** — **Mission-critical aerospace / financial SLA**: Extremely conservative statistical floor. Only near-certain answers execute automatically; anything uncertain is deferred for human review.
- **`risk=0.05` (95.0% SLA)** — **High-throughput web agent / customer workflow**: The optimal operating point for autonomous agents, unlocking high automated throughput while maintaining mathematical error guarantees on held-out data.
- **`mode="standard"`** — Returns the raw calibrated argmax without finite-sample risk deferral.

A ticket's real text decides; the same options over a one-line summary can come back `ABSTAINED`. Pairing prose with typed fields (`Field.from_json`) provides the strongest signal for the decision engine. [Decisions](guide/decisions.md) covers the question and option shapes that decide.

## Next

- [Decisions](guide/decisions.md) — yes/no questions, claim verification, rankings, ratings, and estimates, over passages, tables, and JSON.
- [The certificate](guide/certificate.md) — what the risk level guarantees, and how to certify thresholds on your own data.
- [Python](surfaces/python.md) — batching, async, offline use, and the HTTP client.
