"""A CrewAI crew using a running MIMIR MCP server over Streamable HTTP, through `mcps=`.

CrewAI names MCP tools after their server: over HTTP the agent sees
`127_0_0_1_8000_mcp_route_ticket`, while over stdio the whole launch command goes into the
name and truncates it past recognition. Start the server with
`make mcp TOOLS=examples/tools.yaml`, set `OPENAI_API_KEY`, then run
`make example NAME=crewai/mcp_agent`. With `MIMIR_API_KEYS` set on the server, pass one of the
keys as `api_key`.
"""

from typing import Final

from crewai import LLM, Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.mcp import MCPServerHTTP

MODEL: Final = "openai/gpt-5.5"
URL: Final = "http://127.0.0.1:8000/mcp"


def mimir_server(url: str = URL, api_key: str | None = None) -> MCPServerHTTP:
    """The MIMIR MCP server at `url`, authorised with `api_key` when the server has keys."""
    headers = None if api_key is None else {"Authorization": f"Bearer {api_key}"}
    return MCPServerHTTP(url=url, headers=headers)


def build_crew(server: MCPServerHTTP, llm: BaseLLM | None = None) -> Crew:
    """A one-agent crew routing the ticket given as the `ticket` input, with the server's tools."""
    agent = Agent(
        role="Support dispatcher",
        goal="Send every ticket to the team that owns it.",
        backstory="You route tickets with the route_ticket tool and escalate deferred decisions.",
        llm=LLM(model=MODEL) if llm is None else llm,
        mcps=[server],
    )
    task = Task(
        description="Route this ticket: {ticket}",
        expected_output="The team that will answer, or that a person will review the ticket.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task])


def main() -> None:
    crew = build_crew(mimir_server())
    print(crew.kickoff(inputs={"ticket": "I was charged twice for order 4412."}))


if __name__ == "__main__":
    main()
