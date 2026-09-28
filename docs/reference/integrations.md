# Integrations

Each adapter turns decision tools into its framework's own tools, and a tool-call check
into the framework's own approval hook where it has one. Every framework also reaches
MIMIR through its own MCP client; [`examples/`](https://github.com/abderahmane-ai/mimir/tree/main/examples)
has a native, an MCP and a checked agent for each one.

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

::: mimir.integrations.openai_agents

::: mimir.integrations.langchain

::: mimir.integrations.pydantic_ai

::: mimir.integrations.crewai

::: mimir.integrations.google_adk

::: mimir.integrations.agent_framework

::: mimir.integrations.llamaindex

::: mimir.integrations.smolagents
