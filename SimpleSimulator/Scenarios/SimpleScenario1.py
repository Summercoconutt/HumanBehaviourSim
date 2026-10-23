from Agents.SimpleAgent1 import SimpleAgent1
from Simulator.SimulatorCore import (
    AbstractAgent,
    AbstractScenario,
    Decision,
    ResourceState,
    State,
)


class SimpleScenario1(AbstractScenario):
    @property
    def initial_resources(self) -> dict[type[AbstractAgent], ResourceState]:
        return {SimpleAgent1: {}}

    @property
    def context(self) -> ResourceState:
        return {"rounds": 5}

    @property
    def decisions(self) -> list[Decision]:
        return [
            Decision(
                text_body=f"Round {round_number}: Choose a number.",
                decision=[1, 2, 3, 4, 5],
            )
            for round_number in range(1, 6)
        ]

    def is_complete(self, state: State) -> bool:
        return state.step >= 5