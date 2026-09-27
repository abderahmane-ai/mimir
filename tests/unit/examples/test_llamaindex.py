import asyncio

from llama_index.core.agent.workflow import FunctionAgent

from examples.llamaindex import agent
from mimir.core.results import ChoiceResult
from tests.conftest import RecordingDecider
from tests.unit.mimir.integrations.test_llamaindex import _script


def test_the_support_agent_routes_the_ticket() -> None:
    decider = RecordingDecider()
    built = agent.build_agent(decider, _script("route_ticket", {"context": "charged twice"}))
    assert isinstance(built, FunctionAgent)

    async def main() -> str:
        return str(await built.run(user_msg="Help."))

    answer = asyncio.run(main())
    assert ChoiceResult.model_validate_json(answer).answer == "billing"
    assert decider.calls[0].requests[0][1].option_ids == tuple(agent.TEAMS)
