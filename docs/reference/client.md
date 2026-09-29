# HTTP client

`MimirClient` speaks to a `mimir serve` server with the same methods as `Mimir`, so
code, decision tools and adapters take either. It needs only the base install, and
`api_key` defaults to `$MIMIR_API_KEY`. Connection errors, timeouts and 429, 502, 503,
504 and 529 responses are retried with exponential backoff that honours `Retry-After`;
other errors raise a `ServerResponseError` subclass keeping the status and body. Usable
as a context manager; close it when done.

```python
from mimir import MimirClient

with MimirClient("https://mimir.internal") as remote:
    result = remote.choose(
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

::: mimir.client.http.MimirClient
