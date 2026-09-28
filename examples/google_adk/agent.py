"""A Google ADK agent that routes support tickets with a MIMIR decision tool.

Install `mimir-decisions[local,adk]`, set `GOOGLE_API_KEY`, then run
`make example NAME=google_adk/agent`.
"""

import asyncio
from typing import Final

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from mimir import Choice, Decider, Mimir
from mimir.integrations.google_adk import as_adk_tool

MODEL: Final = "gemini-3.5-flash"
TEAMS: Final = {
    "billing": "Billing: payments, refunds and invoices",
    "security": "Security: account access, passwords and fraud",
    "shipping": "Shipping: deliveries, tracking and returns",
}
INSTRUCTIONS: Final = (
    "Route the customer's ticket with route_ticket, then tell the customer which team will "
    "answer. If the decision is deferred, say that a person will review the ticket."
)


def build_agent(decider: Decider, model: str | BaseLlm = MODEL) -> LlmAgent:
    """The support agent, answering with `decider`."""
    route_ticket = decider.tool(
        "route_ticket",
        Choice("Which team should handle this ticket?", TEAMS),
        description="Route a customer support ticket to the team that owns it.",
    )
    return LlmAgent(
        name="support", model=model, instruction=INSTRUCTIONS, tools=[as_adk_tool(route_ticket)]
    )


async def ask(agent: LlmAgent, text: str) -> str:
    """The agent's final reply to `text`, in a new in-memory session."""
    runner = InMemoryRunner(agent=agent, app_name="support")
    session = await runner.session_service.create_session(app_name="support", user_id="customer")
    message = types.Content(role="user", parts=[types.Part(text=text)])
    reply = ""
    async for event in runner.run_async(
        user_id="customer", session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            reply = "".join(part.text or "" for part in event.content.parts)
    return reply


async def main() -> None:
    agent = build_agent(Mimir.from_pretrained("Mythologic/MIMIR-1"))
    print(await ask(agent, "I was charged twice for order 4412."))


if __name__ == "__main__":
    asyncio.run(main())
