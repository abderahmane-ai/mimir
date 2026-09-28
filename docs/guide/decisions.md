# Decisions

A decision is a spec paired with a context. `model.decide(context, spec, risk=0.01)` returns the result type that matches the spec. Each spec also has a shortcut method on the model so you can skip the spec constructor for simple cases.

| Spec | Shortcut | Answer |
|---|---|---|
| `Choice(question, options)` | `choose` | an option id, or `None` when no option applies |
| `MultiChoice(question, options)` | — | the option ids that apply |
| `YesNo(question)` | `yes_no` | `True` or `False` |
| `Verify(claim)` | `verify` | `supported`, `contradicted`, or `not_enough_information` |
| `Rank(question, candidates)` | `rank` | candidate ids ordered best first |
| `Rate(question, levels)` | `rate` | a level id; levels are given lowest first |
| `Estimate(question, low, high, unit)` | `estimate` | a number in `[low, high]`, with a confidence interval |

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

## Results

Every result carries: `status`, `answer`, `confidence`, `relevant_context`, `deferral`, `certificate`, and `latency_ms`.

All types except `Estimate` also carry `probabilities`. Choice, yes/no, and verify results add `abstain_probability` and a conformal `prediction_set`. A rating result includes the contiguous levels in its set. An estimate result includes its `interval` at `alpha`.

`relevant_context` lists passages, tables, table rows, and fields by their share of the evidence, highest first, so you can show users exactly what the model relied on.

## Batching and async

`decide_many` takes a list of `(context, spec)` pairs and batches them by token length for efficiency. Every method has an async counterpart: `adecide`, `adecide_many`, `achoose`, `ayes_no`, `averify`, `arank`, `arate`, `aestimate`.

`decide_uncertified` returns the raw model answer with no policy applied — no certificate, no deferral. Use it when you want the model's view without any threshold enforcement.

## Limits

The maximum number of options, levels, candidates, and context tokens for a given release are reported by `model.info()`. A request that exceeds any limit raises `InputLimitError`, which names the value that exceeded the limit and the limit itself.
