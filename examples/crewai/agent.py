"""A CrewAI crew that routes support tickets with a MIMIR decision tool.

Install `mimir-decisions[local,crewai]`, set `OPENAI_API_KEY`, then run
`make example NAME=crewai/agent`.
"""

from typing import Final

from crewai import LLM, Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM

from mimir import Choice, Decider, Mimir
from mimir.integrations.crewai import as_crewai_tool

MODEL: Final = "openai/gpt-5.5"
TEAMS: Final = {
    "billing": "Billing: payments, refunds and invoices",
    "security": "Security: account access, passwords and fraud",
    "shipping": "Shipping: deliveries, tracking and returns",
}


def build_crew(decider: Decider, llm: BaseLLM | None = None) -> Crew:
    """A one-agent crew routing the ticket given as the `ticket` input, answering with `decider`."""
    route_ticket = decider.tool(
        "route_ticket",
        Choice("Which team should handle this ticket?", TEAMS),
        description="Route a customer support ticket to the team that owns it.",
    )
    agent = Agent(
        role="Support dispatcher",
        goal="Send every ticket to the team that owns it.",
        backstory="You route tickets with route_ticket and escalate deferred decisions.",
        llm=LLM(model=MODEL) if llm is None else llm,
        tools=[as_crewai_tool(route_ticket)],
    )
    task = Task(
        description="Route this ticket: {ticket}",
        expected_output="The team that will answer, or that a person will review the ticket.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task])


def main() -> None:
    crew = build_crew(Mimir.from_pretrained("Mythologic/MIMIR-1"))
    print(crew.kickoff(inputs={"ticket": "I was charged twice for order 4412."}))


if __name__ == "__main__":
    main()
