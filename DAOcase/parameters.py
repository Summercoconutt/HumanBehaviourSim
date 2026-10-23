import bootstrap
from Agents.SimpleAgent1 import SimpleAgent1
from Simulator.SimulatorCore import AbstractParameters, AgentConfiguration


class DaoParameters(AbstractParameters):
    @property
    def agent_configurations(self) -> list[AgentConfiguration]:
        return [AgentConfiguration(agent_type=SimpleAgent1, count=3)]
