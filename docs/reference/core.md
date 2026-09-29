# Core

The contract every surface shares: decision specs, contexts, results, decision tools, and tool-call checks. It depends only on Pydantic, so a project can build requests and read results without installing Torch.

## Decider

`Decider` is the interface both `Mimir` and `MimirClient` implement. It defines:

- `decide(context, spec)` and `decide_many(pairs)` — the primary decision methods, over any spec type. Both take `mode`, `min_confidence`, `risk`, and `alpha`.
- One shortcut per spec: `choose`, `yes_no`, `verify`, `rank`, `rate`, `estimate`.
- Async counterparts for every method: `adecide`, `adecide_many`, `achoose`, and so on.
- `decide_uncertified(context, spec)` — the raw model answer with no policy applied: no calibration, no certificate.
- `tool(name, spec, description)` — bind a spec to a name to create a decision tool.
- `tool_call_check(rules, tools)` — create a check that decides an agent's tool calls against rules.
- `info()` — the loaded model's id, revision, variant, runtime fingerprint, certified risk levels, and input limits.

::: mimir.core.decider

::: mimir.core.decisions

::: mimir.core.context

::: mimir.core.results

::: mimir.core.tools

::: mimir.core.checks
