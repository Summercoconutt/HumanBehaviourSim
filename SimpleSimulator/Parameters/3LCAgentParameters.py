from Agents.LinearClassifierAgent import LCAgent
from Simulator.SimulatorCore import AbstractParameters, AgentConfiguration


class SimpleParameters(AbstractParameters):
    @property
    def agent_configurations(self) -> list[AgentConfiguration]:
        return [AgentConfiguration(agent_type=LCAgent, count=3)]