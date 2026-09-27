# Decisions

A decision is a spec and a context. `model.decide(context, spec, risk=0.01)` returns the
result type matching the spec; each spec also has a shortcut method.

| Spec | Shortcut | Answer |
|---|---|---|
| `Choice(question, options)` | `choose` | an option id, or None when no option applies |
| `MultiChoice(question, options)` | | the option ids that apply |
| `YesNo(question)` | `yes_no` | `True` or `False` |
| `Verify(claim)` | `verify` | `supported`, `contradicted` or `not_enough_information` |
| `Rank(question, candidates)` | `rank` | candidate ids, best first |
| `Rate(question, levels)` | `rate` | a level id; levels are given lowest first |
| `Estimate(question, low, high, unit)` | `estimate` | a number in `[low, high]`, with an interval |

Options, candidates and levels are a list of ids, or a mapping from id to a description the
model reads.

## Context

A context is a string, a list of strings, a dict read as a JSON state, or a `Context` of
passages, tables and fields.

```python
from mimir import Context, Field, Passage, Rate, Table

context = Context(
    passages=[Passage(title="Ticket #4412", text="The export has failed every night this week.")],
    tables=[Table.from_rows([["2026-03-02", "failed"]], header=["date", "status"])],
    fields=Field.from_json({"customer": {"plan": "enterprise", "seats": 240}}),
)
result = model.decide(context, Rate("How urgent is this?", ["low", "medium", "high"]), risk=0.01)
```

- Numbers and dates in tables and fields are read as typed values, not only as text.
- `Table.from_dataframe(frame)` reads a pandas or polars DataFrame: its columns are the header
  and missing values are blank cells.
- `Field.from_json(value)` flattens nested JSON into fields named by their path.

## Results

Every result has `status`, `answer`, `confidence`, `relevant_context`, `deferral`,
`certificate` and `latency_ms`, and all but `Estimate` carry `probabilities`. Choice, yes/no and
verify results add `abstain_probability` and a conformal `prediction_set`, a rating the
contiguous levels in its set, and an estimate its `interval` at `alpha`. `relevant_context`
lists passages, tables, table rows and fields by their share of the evidence, highest first.

## Many decisions

`decide_many` takes `(context, spec)` pairs and batches them by length; every method has an
async form (`adecide`, `adecide_many`, `achoose`, ...). `decide_uncertified` returns the
model's raw answer with no policy applied: no certificate, no deferral.

Limits (the most options, levels and context tokens) are the release's and are in
`model.info()`; a request over one raises `InputLimitError` naming the value and the limit.
