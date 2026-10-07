# mimir-decisions

**Decisions your agents can act on.** MIMIR is a non-generative decision model: give it
a context, a question and the options, and get back a typed answer with calibrated
probabilities, the evidence behind it, and a certified verdict on whether to act or
escalate. No text generated. Nothing to parse. Nothing to hallucinate.

It beats Laya and GLiNER2.5-Decide head-to-head on six of ten tasks — by 54.7 points on
Banking77, 42.3 on MASSIVE, 36.3 on typed decisions — and where it cannot back an
answer, it abstains instead of guessing.

```bash
pip install "mimir-decisions[local]"        # the local engine (CPU and CUDA)
pip install mimir-decisions                  # data models and HTTP client only
```

Python 3.11+. Documentation: <https://abderahmane-ai.github.io/mimir/>

---

## Why MIMIR

Most agents route, classify, and verify using a general-purpose language model: slow, expensive, and impossible to audit. MIMIR is built for structured decisions. It runs on Torch in milliseconds, returns calibrated probabilities with every answer, and issues a certificate — measured evidence that answers passing its threshold stayed at or below the risk level you ask for, on held-out data.

- **No generation.** Answers are drawn from the options you supply, not synthesised. The model cannot hallucinate an answer that wasn't on the list.
- **Calibrated confidence.** Probabilities are not softmax scores; they are calibrated to match realised accuracy on held-out data.
- **Answers, always.** Every decision returns the model's prediction with its probabilities. In `threshold` and `certified` modes, answers below the floor come back `deferred` for review — never withheld.
- **One typed contract.** Seven decision types — choice, multi-choice, yes/no, verify, rank, rate, estimate — all returning the same result shape, over any context.
- **Portable.** The same Python interface works locally on CPU or GPU, over HTTP, and over MCP. Framework adapters exist for eight agent SDKs.

---

## Quickstart

```python
from mimir import Context, Field, Mimir, Passage

model = Mimir.from_pretrained("Mythologic/MIMIR-1")

# Context pairs prose passages with typed structured fields (amounts, IDs, metadata)
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

result.status         # Status.DECIDED, Status.ABSTAINED or Status.DEFERRED
result.answer         # "billing", or None when no option applies
result.probabilities  # calibrated probability of each option id
result.certificate    # mathematical proof of risk bound on held-out data
```

`answer` is always the model's prediction. `status` says whether it cleared the operating floor:

- `DECIDED` — act on `answer`.
- `ABSTAINED` — no listed option applies (clean OOD rejection).
- `DEFERRED` — the answer came in below the certificate floor; route to a human reviewer.

### The Risk Dial: From Mission-Critical SLA to High-Throughput Agents

The release certifies thresholds across four finite-sample risk levels (`model.info().risk_levels = (0.005, 0.01, 0.02, 0.05)`):

- **`risk=0.01` (99.0% SLA)** — **Mission-critical aerospace / financial SLA**: Sets an ultra-strict statistical floor. Only near-certain answers execute automatically; anything uncertain is deferred for human review rather than risked.
- **`risk=0.05` (95.0% SLA)** — **High-throughput web agent / customer workflow**: The sweet spot for autonomous agents, unlocking automated execution on legitimate requests while still mathematically bounding error rates on held-out data.
- **`mode="standard"`** — Returns the calibrated argmax without finite-sample risk deferral.

Give the model the ticket as a person wrote it paired with typed fields — the same options over a one-line summary can come back `ABSTAINED`. The [Decisions guide](https://abderahmane-ai.github.io/mimir/guide/decisions/) covers the question and option shapes that decide.

The first call downloads the model from the Hugging Face Hub at the revision this package version pins, verifies its Sigstore signature, checks every file against the manifest's SHA-256, and loads it.

---

## Decision types

| Spec | Answer |
|---|---|
| `Choice(question, options)` | an option id, or `None` |
| `MultiChoice(question, options)` | the option ids that apply |
| `YesNo(question)` | `True` or `False` |
| `Verify(claim)` | `supported`, `contradicted`, or `not_enough_information` |
| `Rank(question, candidates)` | candidate ids, best first |
| `Rate(question, levels)` | a level id; levels given lowest first |
| `Estimate(question, low, high, unit)` | a number in `[low, high]`, with a confidence interval |

```python
from mimir import Context, Field, Passage, Rate, Table

context = Context(
    passages=[Passage(title="Ticket #4412", text="The export has failed every night this week.")],
    tables=[Table.from_rows([["2026-03-02", "failed"]], header=["date", "status"])],
    fields=Field.from_json({"customer": {"plan": "enterprise", "seats": 240}}),
)
result = model.decide(context, Rate("How urgent is this?", ["low", "medium", "high"]), risk=0.01)
```

A context can be a string, a list of strings, a dict read as a JSON state, or a `Context` of typed passages, tables, and fields. `Table.from_dataframe(frame)` reads a pandas or polars DataFrame. `decide_many` batches multiple decisions, and every method has an async counterpart (`adecide`, `adecide_many`, …).

---

## Certification

`decide` answers every request in `standard` mode. `threshold` mode defers answers below your `min_confidence`; `certified` mode defers answers below the release's threshold at `risk` (`model.info().risk_levels`), and attaches the certificate when the answer passes. Dial `risk=0.01` (99% SLA) for mission-critical operations where false actions carry heavy penalties, or `risk=0.05` (95% SLA) for high-throughput autonomous agents. `decide_uncertified` returns the raw model answer with no policy applied.

A certificate covers one exact configuration: weights, Torch version, device, and hardware. On hardware not listed in the certificate, the first load runs the release's equivalence set and requires every decision to match. To certify thresholds on your own labelled data:

```bash
mimir calibrate labelled.jsonl --risk 0.01 --confidence 0.95 --out policy.json
```

```python
model = Mimir.from_pretrained("Mythologic/MIMIR-1", policy="policy.json")
```

---

## Remote use

```python
from mimir import MimirClient

remote = MimirClient("https://mimir.internal", api_key="...")
remote.choose(
    "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    "Which department should handle this request?",
    options={
        "billing": "Billing: invoices, payments, refunds",
        "technical": "Technical: bugs, outages, system errors",
        "sales": "Sales: pricing, new contracts",
        "other": "Other: everything else",
    },
)
```

`MimirClient` has the same interface as `Mimir`, so all code, decision tools, and framework adapters accept either. It requires only the base install. Connection errors, timeouts, and 429/502/503/504/529 responses are retried with exponential backoff that honours `Retry-After`.

---

## Decision tools

```python
from mimir import Choice

route_ticket = model.tool(
    "route_ticket",
    Choice(
        "Which department should handle this request?",
        {
            "billing": "Billing: invoices, payments, refunds",
            "technical": "Technical: bugs, outages, system errors",
            "sales": "Sales: pricing, new contracts",
            "other": "Other: everything else",
        },
    ),
    description="Route a support ticket to the team that owns it.",
)
route_ticket(
    "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
)
route_ticket.input_schema, route_ticket.output_schema
```

Tools can also be declared in a YAML file, which the HTTP and MCP servers load:

```yaml
tools:
  - name: route_ticket
    description: Route a support ticket to the team that owns it.
    decision:
      type: choice
      question: Which department should handle this request?
      options:
        billing: "Billing: invoices, payments, refunds"
        technical: "Technical: bugs, outages, system errors"
        sales: "Sales: pricing, new contracts"
        other: "Other: everything else"
```

---

## Tool-call checks

A tool-call check decides, against rules you write, whether an agent's pending tool call may run. A confident yes allows it, a confident no denies it, and anything else escalates to a person.

```python
check = model.tool_call_check(
    ["Refunds above 500 dollars need a manager's approval."], tools=["issue_refund"]
)
outcome = check("issue_refund", {"order": "4412", "amount": 900})
outcome.permission    # Permission.ALLOW, Permission.DENY or Permission.ESCALATE
outcome.reason        # one sentence for the agent or the approver
```

---

## Agent frameworks

Each adapter turns decision tools into the framework's native tool type and wires a tool-call check into that framework's own approval hook.

| Framework | Install | Tools | Tool-call check |
|---|---|---|---|
| OpenAI Agents SDK | `mimir-decisions[openai-agents]` | `as_function_tool` | `guard`: escalations pause the run for approval |
| LangChain / LangGraph | `mimir-decisions[langchain]` | `as_structured_tool` | `ToolCallCheckMiddleware`: escalations interrupt with the human-in-the-loop request |
| PydanticAI | `mimir-decisions[pydantic-ai]` | `as_toolset` | `guard`: escalations end the run with `DeferredToolRequests` |
| CrewAI | `mimir-decisions[crewai]` | `as_crewai_tool` | `tool_call_hook`: escalations go to your approver |
| Google ADK | `mimir-decisions[adk]` | `as_adk_tool` | `tool_call_callback`: escalations ask for ADK confirmation |
| Microsoft Agent Framework | `mimir-decisions[agent-framework]` | `as_function_tool` | `ToolCallCheckMiddleware`: only confident calls run |
| LlamaIndex | `mimir-decisions[llamaindex]` | `as_llamaindex_tool` | none |
| smolagents | `mimir-decisions[smolagents]` | `as_smolagents_tool` | none |

```python
from agents import Agent
from mimir.integrations.openai_agents import as_function_tool

agent = Agent(name="support", tools=[as_function_tool(route_ticket)])
```

Every framework also reaches MIMIR through its own MCP client. [`examples/`](examples) has a native, an MCP, and a checked agent for each framework, plus a Vercel AI SDK agent in TypeScript.

---

## HTTP server

```bash
pip install "mimir-decisions[local,server]"
MIMIR_API_KEYS=key-one,key-two mimir serve --host 0.0.0.0 --tools tools.yaml
```

| Route | Does |
|---|---|
| `POST /v1/decide` | one decision: `{context, decision, mode, min_confidence, risk, alpha}` |
| `POST /v1/decide/uncertified` | the model's raw answer: `{context, decision}` |
| `POST /v1/decide/batch` | up to 64 decisions in one call |
| `POST /v1/tools/{name}` | a tool from `--tools`, given only `{context}` |
| `POST /v1/systemone` | Jev's request and response format |
| `GET /v1/models` | model, revision, runtime and certified risk levels |
| `GET /healthz`, `GET /readyz` | liveness, and readiness once the model is loaded |
| `GET /metrics` | Prometheus metrics |

Concurrent requests are batched. With keys in `MIMIR_API_KEYS`, every route except the probes requires `Authorization: Bearer <key>`. A server with no keys listens only on loopback unless started with `--allow-no-auth`. The OpenAPI 3.1 document is [`openapi.json`](openapi.json).

---

## MCP server

Each configured tool becomes an MCP tool that takes only a context; `--generic-tools` adds `mimir_choose`, `mimir_verify`, `mimir_rank`, and `mimir_rate`. A deferred decision is a normal result telling the agent to escalate.

```bash
uvx --from "mimir-decisions[local,mcp]" mimir-decisions mcp --tools tools.yaml               # stdio
MIMIR_API_KEYS=... mimir mcp --http --host 0.0.0.0 --tools tools.yaml       # Streamable HTTP at /mcp
mimir mcp --tools tools.yaml --remote https://mimir.internal                  # forward to a server
mimir serve --mcp --tools tools.yaml                                          # HTTP API and /mcp together
```

In Claude Code:

```bash
claude mcp add mimir -- uvx --from "mimir-decisions[local,mcp]" mimir-decisions mcp --tools /path/to/tools.yaml
claude mcp add --transport http mimir https://mimir.internal/mcp --header "Authorization: Bearer ..."
```

Claude Desktop, Cursor, and VS Code take the same command or the same URL and header in their MCP configuration. The server is registered in the MCP Registry as `io.github.abderahmane-ai/mimir`.

<!-- mcp-name: io.github.abderahmane-ai/mimir -->

---

## Containers

```bash
docker run -p 8000:8000 -e MIMIR_API_KEYS=... -v mimir-models:/models ghcr.io/abderahmane-ai/mimir:1.1.0-cpu
docker run --gpus all -p 8000:8000 -e MIMIR_API_KEYS=... -v mimir-models:/models ghcr.io/abderahmane-ai/mimir:1.1.0-cuda
```

Images carry the runtime, never the model weights. On first start, the model is downloaded at the revision the package version pins, verified, and cached in `/models`. To run from that cache with no network access, append `serve --host 0.0.0.0 --model-cache /models --offline`.

Images are signed with Sigstore by the release workflow:

```bash
cosign verify ghcr.io/abderahmane-ai/mimir:1.1.0-cpu \
  --certificate-identity https://github.com/abderahmane-ai/mimir/.github/workflows/release.yml@refs/heads/main \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

---

## Command line

| Command | Description |
|---|---|
| `mimir serve` | the HTTP server; `--mcp` also serves MCP at `/mcp` |
| `mimir mcp` | the MCP server, over stdio or `--http` |
| `mimir decide` | one decision from flags, or a JSON request on stdin |
| `mimir bench FILE` | accuracy, coverage, certified share and realised risk on labelled decisions |
| `mimir calibrate FILE` | certify thresholds on labelled decisions |
| `mimir schema` | JSON Schemas of every spec, result and request |
| `mimir download` | download and verify a release for offline use |
| `mimir doctor` | report the environment; `--verify` loads the model and runs the equivalence check |

---

## Integrity

Releases are loaded from a pinned Hugging Face revision. Before any model file is read, the manifest's Sigstore signature is verified against the `abderahmane-ai/mimir` release workflow and every file is checked against the manifest's SHA-256. No pickle is used anywhere.

---

## Migrating

`mimir.compat.systemone.v1` converts Jev `/v1/systemone` requests and responses, and `mimir.compat.laya.v1` exposes `load(...).predict(state, questions)` in Laya 0.3.20's shape. See the [migration guides](https://abderahmane-ai.github.io/mimir/migrating/jev/) for step-by-step instructions.

---

## Licensing

The MIMIR SDK is licensed under [Apache-2.0](LICENSE); that license covers the software only,
not the MIMIR model weights. The MIMIR-1 model weights are licensed separately under the
[MIMIR Model License](MODEL-LICENSE.md).

Eligible community users may use MIMIR-1 commercially without royalties, subject to the
MIMIR Model License. Anyone may download, benchmark, evaluate and prototype with MIMIR
without registering.

Organizations exceeding the Revenue Threshold (US$1,000,000 annual gross revenue), or
requiring enterprise, OEM, redistribution, hosting, or other additional rights, may obtain a
commercial agreement from Mythologic. See [COMMERCIAL-LICENSING.md](COMMERCIAL-LICENSING.md).
