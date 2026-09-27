# mimirai

Typed, calibrated and certified decisions from **MIMIR**, a non-generative decision model by
VathosAI. Give it a context, a question and the options; get back a typed answer, calibrated
probabilities, the parts of the context it relied on, and a certified signal for when to act
and when to escalate.

```bash
pip install "mimirai[local]"        # CPU engine
pip install "mimirai[local-gpu]"    # CUDA engine
pip install mimirai                 # data models and the HTTP client only
```

Python 3.11 or later.

## Quickstart

```python
from mimir import Mimir

model = Mimir.from_pretrained("vathosai/mimir-1")
result = model.choose(
    "My card was charged twice for the same order.",
    "Which team should handle this ticket?",
    options={"billing": "Billing: payments, refunds", "security": "Security: account access"},
)

result.status         # Status.DECIDED, Status.ABSTAINED or Status.DEFERRED
result.answer         # an option id, or None when no option applies
result.probabilities  # calibrated probability of each option id
result.certificate    # the certified threshold the decision was checked against
```

`answer` is the model's prediction. `status` is the policy's verdict on it:

- `DECIDED`: the answer is an option and is certified at the requested risk.
- `ABSTAINED`: the answer is "none of the options" and is certified.
- `DEFERRED`: not certified; `result.deferral.reason` is `below_threshold`,
  `out_of_distribution` or `no_certified_threshold`.

## Decisions

| Spec | Answer |
|---|---|
| `Choice(question, options)` | an option id, or None |
| `MultiChoice(question, options)` | the option ids that apply |
| `YesNo(question)` | `True` or `False` |
| `Verify(claim)` | `supported`, `contradicted` or `not_enough_information` |
| `Rank(question, candidates)` | candidate ids, best first |
| `Rate(question, levels)` | a level id, levels given lowest first |
| `Estimate(question, low, high, unit)` | a number in `[low, high]` with an interval |

```python
from mimir import Context, Field, Passage, Rate, Table

context = Context(
    passages=[Passage(title="Ticket #4412", text="The export has failed every night this week.")],
    tables=[Table.from_rows([["2026-03-02", "failed"]], header=["date", "status"])],
    fields=Field.from_json({"customer": {"plan": "enterprise", "seats": 240}}),
)
result = model.decide(context, Rate("How urgent is this?", ["low", "medium", "high"]), risk=0.01)
```

A context can also be a string, a list of strings, or a dict read as a JSON state. Numbers and
dates in tables and fields are typed. `Table.from_dataframe(frame)` reads a pandas or polars
DataFrame: its columns are the header and missing values are blank cells. `decide_many` batches many decisions, and every method
has an async form (`adecide`, `adecide_many`, ...).

## Certification

`decide` takes a risk level certified by the loaded policy (`model.info().risk_levels`). A
decision is taken only when its calibrated confidence reaches a threshold certified on held-out
data to keep the error rate of taken decisions at or below that risk with 95% confidence, and
when the context passes the out-of-distribution gate. `decide_uncertified` returns the raw
model answer with no policy applied.

A certificate covers one exact configuration: model files, variant, ONNX Runtime version,
execution provider and options. On hardware the certificate does not list, the first load runs
the release's equivalence set and requires every decision to match. To certify thresholds on
your own labelled data:

```bash
mimir calibrate labelled.jsonl --risk 0.01 --confidence 0.95 --out policy.json
```

```python
model = Mimir.from_pretrained("vathosai/mimir-1", policy="policy.json")
```

## Remote use

```python
from mimir import MimirClient

remote = MimirClient("https://mimir.internal", api_key="...")
remote.choose("...", "Which team?", options=["billing", "security"])
```

`MimirClient` has the same interface as `Mimir`, so code and tools accept either.

## Decision tools

```python
from mimir import Choice

route_ticket = model.tool(
    "route_ticket",
    Choice("Which team should handle this ticket?", options=["billing", "security"]),
    description="Route a support ticket to the team that owns it.",
)
route_ticket("My card was charged twice")
route_ticket.input_schema, route_ticket.output_schema
```

Tools can also be declared in a YAML file, which the HTTP and MCP servers load:

```yaml
tools:
  - name: route_ticket
    description: Route a support ticket to the team that owns it.
    decision:
      type: choice
      question: Which team should handle this ticket?
      options: [billing, security]
```

## Tool-call checks

A tool-call check decides, against rules you write, whether an agent's pending tool call may
run. A certified yes allows it, a certified no denies it, and anything else escalates it to a
person.

```python
check = model.tool_call_check(
    ["Refunds above 500 dollars need a manager's approval."], tools=["issue_refund"]
)
outcome = check("issue_refund", {"order": "4412", "amount": 900})
outcome.permission    # Permission.ALLOW, Permission.DENY or Permission.ESCALATE
outcome.reason        # one sentence for the agent or the approver
```

Write the rules the call is judged against; without them the check has nothing to decide by.

## Agent frameworks

Each adapter turns decision tools into the framework's own tools, and a tool-call check into
its own approval hook where it has one.

| Framework | Install | Tools | Tool-call check |
|---|---|---|---|
| OpenAI Agents SDK | `mimirai[openai-agents]` | `as_function_tool` | `guard`: escalations pause the run for approval |
| LangChain, LangGraph | `mimirai[langchain]` | `as_structured_tool` | `ToolCallCheckMiddleware`: escalations interrupt with the human-in-the-loop request |
| PydanticAI | `mimirai[pydantic-ai]` | `as_toolset` | `guard`: escalations end the run with `DeferredToolRequests` |
| CrewAI | `mimirai[crewai]` | `as_crewai_tool` | `tool_call_hook`: escalations go to your approver |
| Google ADK | `mimirai[adk]` | `as_adk_tool` | `tool_call_callback`: escalations ask for ADK confirmation |
| Microsoft Agent Framework | `mimirai[agent-framework]` | `as_function_tool` | `ToolCallCheckMiddleware`: only certified calls run |
| LlamaIndex | `mimirai[llamaindex]` | `as_llamaindex_tool` | none: no hook before a tool call |
| smolagents | `mimirai[smolagents]` | `as_smolagents_tool` | none: no hook before a tool call |

```python
from agents import Agent
from mimir.integrations.openai_agents import as_function_tool

agent = Agent(name="support", tools=[as_function_tool(route_ticket)])
```

Every framework also reaches MIMIR through its own MCP client, and so does any other language:
[`examples/`](examples) has a native, an MCP and a checked agent for each framework, and a
Vercel AI SDK agent in TypeScript.

## HTTP server

```bash
pip install "mimirai[local,server]"
MIMIR_API_KEYS=key-one,key-two mimir serve --host 0.0.0.0 --tools tools.yaml
```

| Route | Does |
|---|---|
| `POST /v1/decide` | one certified decision: `{context, decision, risk, alpha}` |
| `POST /v1/decide/uncertified` | the model's raw answer: `{context, decision}` |
| `POST /v1/decide/batch` | up to 64 decisions in one call |
| `POST /v1/tools/{name}` | a tool from `--tools`, given only `{context}` |
| `POST /v1/systemone` | Jev's request and response format |
| `GET /v1/models` | model, revision, runtime and certified risk levels |
| `GET /healthz`, `GET /readyz` | liveness, and readiness once the model is loaded |
| `GET /metrics` | Prometheus metrics |

Concurrent requests are batched. With keys in `MIMIR_API_KEYS`, every route but the probes
needs `Authorization: Bearer <key>`; a server without keys listens only on loopback unless
started with `--allow-no-auth`. The OpenAPI 3.1 document is `openapi.json`.

## MCP server

Each configured tool becomes an MCP tool that takes only a context; `--generic-tools` adds
`mimir_choose`, `mimir_verify`, `mimir_rank` and `mimir_rate`. A deferred decision is a normal
result telling the agent to escalate.

```bash
uvx --from "mimirai[local,mcp]" mimirai mcp --tools tools.yaml               # stdio
MIMIR_API_KEYS=... mimir mcp --http --host 0.0.0.0 --tools tools.yaml       # Streamable HTTP at /mcp
mimir mcp --tools tools.yaml --remote https://mimir.internal                  # forward to a server
mimir serve --mcp --tools tools.yaml                                          # HTTP API and /mcp together
```

In Claude Code:

```bash
claude mcp add mimir -- uvx --from "mimirai[local,mcp]" mimirai mcp --tools /path/to/tools.yaml
claude mcp add --transport http mimir https://mimir.internal/mcp --header "Authorization: Bearer ..."
```

Claude Desktop, Cursor and VS Code take the same command, or the same URL and header, in their
MCP server configuration.

<!-- mcp-name: io.github.vathosai/mimir -->

## Containers

```bash
docker run -p 8000:8000 -e MIMIR_API_KEYS=... -v mimir-models:/models ghcr.io/vathosai/mimir:1.0.0-cpu
docker run --gpus all -p 8000:8000 -e MIMIR_API_KEYS=... -v mimir-models:/models ghcr.io/vathosai/mimir:1.0.0-cuda
```

Images carry the runtime, never the model: it is downloaded and verified into `/models` on
first start. To run from that cache with no network, end the command with
`serve --host 0.0.0.0 --model-cache /models --offline`.

## Command line

| Command | Description |
|---|---|
| `mimir serve` | the HTTP server; `--mcp` also serves MCP at `/mcp` |
| `mimir mcp` | the MCP server, over stdio or `--http` |
| `mimir decide` | one decision from flags, or a JSON request on stdin |
| `mimir bench FILE` | accuracy, coverage and realised risk on labelled decisions |
| `mimir calibrate FILE` | certify thresholds on labelled decisions |
| `mimir schema` | JSON Schemas of every spec, result and request |
| `mimir download` | download and verify a release for offline use |
| `mimir doctor` | report the environment; `--verify` loads the model |

## Integrity

Releases are loaded from a pinned Hugging Face revision. Before anything is read, the
manifest's Sigstore signature is verified against the VathosAI release workflow, every file is
checked against the manifest's SHA-256, and the ONNX graph is checked against its operator
allowlist and signature. No pickle is used anywhere.

## Migrating

`mimir.compat.systemone.v1` converts Jev `/v1/systemone` requests and answers, and
`mimir.compat.laya.v1` offers `load(...).predict(state, questions)` in Laya 0.3.20's shape.

## License

The `mimirai` package is licensed under Apache 2.0. The MIMIR model weights are distributed
under their own license on the Hugging Face Hub.
