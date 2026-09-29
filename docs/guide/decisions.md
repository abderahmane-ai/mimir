# Decisions

A decision is a spec paired with a context. `model.decide(context, spec, risk=0.01)` returns the result type that matches the spec. Each spec also has a shortcut method on the model so you can skip the spec constructor for simple cases.

| Spec | Shortcut | Answer | Typical use |
|---|---|---|---|
| `Choice(question, options)` | `choose` | an option id, or `None` when no option applies | routing, classification, triage |
| `MultiChoice(question, options)` | — | the option ids that apply | tagging, when several options can apply |
| `YesNo(question)` | `yes_no` | `True` or `False` | filters, boolean checks |
| `Verify(claim)` | `verify` | `supported`, `contradicted`, or `not_enough_information` | grounding an answer in evidence, fact-checking |
| `Rank(question, candidates)` | `rank` | candidate ids ordered best first | shortlisting, choosing among candidates |
| `Rate(question, levels)` | `rate` | a level id; levels are given lowest first | severity, priority, grading |
| `Estimate(question, low, high, unit)` | `estimate` | a number in `[low, high]`, with a confidence interval | cost, time, quantity |

`YesNo` fixes its two option texts (`no`, `yes`); a two-option `Choice` lets you write the texts yourself. Both are binary decisions.

Options, candidates, and levels are either a list of ids or a mapping from id to a description the model reads as context for that option.

## Context

A context is anything the model reads as the evidence for the decision. It can be:

- a `str` — a single passage of text;
- a `list[str]` — multiple passages, each treated as a separate unit;
- a `dict` — read as a flat JSON state, with keys as field names;
- a `Context` — a structured combination of typed passages, tables, and fields.

```python
from mimir import Context, Field, Passage, Rate, Table

context = Context(
    passages=[Passage(title="Ticket #4412", text="The export has failed every night this week.")],
    tables=[Table.from_rows([["2026-03-02", "failed"]], header=["date", "status"])],
    fields=Field.from_json({"customer": {"plan": "enterprise", "seats": 240}}),
)
result = model.decide(context, Rate("How urgent is this?", ["low", "medium", "high"]), risk=0.01)
```

- Numbers and dates in tables and fields are read as typed values, not as plain text.
- `Table.from_dataframe(frame)` reads a pandas or polars DataFrame: its columns become the header and missing values become blank cells.
- `Field.from_json(value)` flattens nested JSON into fields named by their path (`customer.plan`, `customer.seats`).

## Writing a context

The context is read as a sequence of units: a passage is one, a table's caption and header is one, each table row is one, and each field is one. `relevant_context` reports relevance per unit, so the unit you choose is what a result can point at.

- Prose — a document, ticket, or policy — is a passage.
- A table gives one unit per record. Use it when the decision is about which record applies, or when a person must audit exactly what was read. The header names the columns the decision reads.
- A field is one named value; `Field.from_json` flattens nested JSON into dotted keys (`customer.seats`). Key names are read, so use words.
- Only table cells and field values are typed: `$1,200` is the number 1200, `12%` is 12.0, `4 March 2026` is a date. A number inside prose is read as text, so put the values a decision turns on in cells or fields.

Start with the simplest shape that carries the evidence. `decide_many` batches any mix of shapes.

## Choosing a spec

The decision types are not interchangeable under the shipped policy:

| Spec | Certifies at | Coverage at risk 1% |
|---|---|---|
| `YesNo`, `Choice`, `Verify` | every risk level | about a third of held-out calls |
| `Rank` | every risk level but 0.5% | about a third of held-out calls |
| `Rate` | 5% only | 1.4% of held-out calls |
| `MultiChoice`, `Estimate` | none; they defer at every level | — |

A deferral is the designed outcome, not a failure: `DECIDED` at 1% risk means the release's threshold certified roughly the top third of calls on its held-out data, so build the escalation path for the rest. The table is the release's data; the number that applies to you is in `result.certificate.coverage`, and `mimir bench` measures it on your labels.

`MultiChoice` and `Estimate` can still be used: their results carry probabilities and, for `Estimate`, an interval, but their status is always `DEFERRED` unless you load a policy calibrated for them (`mimir calibrate`). The fp16 (CUDA) graph ships without a policy at all: `decide` raises `PolicyError` there, and `decide_uncertified` returns the raw answer.

## Writing the question and options

- One question per decision; it is read with every option.
- The option text is what the model reads, and the id is what comes back, so give a mapping with descriptions when the ids alone are codes: `{"billing": "Billing: payments and refunds"}`.
- `Verify` takes the claim, and its verdicts are fixed; `YesNo` fixes `yes` and `no`. A two-point scale belongs in `Choice` or `YesNo`, not `Rate`, which needs at least three levels.
- `ABSTAINED` is a certified answer — "no listed option applies" — so an option only belongs in the list if it can genuinely apply.

When a result is not what you expected, read `relevant_context` first: if the evidence is not listed, the context did not carry it. [Troubleshooting](troubleshooting.md) reads the deferral reasons.

## Results

Every result carries: `status`, `answer`, `confidence`, `relevant_context`, `deferral`, `certificate`, and `latency_ms`.

All types except `Estimate` also carry `probabilities`. Choice, yes/no, and verify results add `abstain_probability` and a conformal `prediction_set`. A rating result includes the contiguous levels in its set. An estimate result includes its `interval` at `alpha`.

`relevant_context` lists passages, tables, table rows, and fields by their share of the evidence, highest first, so you can show users exactly what the model relied on.

## Batching and async

`decide_many` takes a list of `(context, spec)` pairs and batches them by token length for efficiency. Every method has an async counterpart: `adecide`, `adecide_many`, `achoose`, `ayes_no`, `averify`, `arank`, `arate`, `aestimate`.

`decide_uncertified` returns the raw model answer with no policy applied — no certificate, no deferral. Use it when you want the model's view without any threshold enforcement.

## Limits

The maximum number of options, levels, candidates, and context tokens for a given release are reported by `model.info()`. A request that exceeds any limit raises `InputLimitError`, which names the value that exceeded the limit and the limit itself.
