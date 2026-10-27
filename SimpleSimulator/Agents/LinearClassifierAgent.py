from Simulator.SimulatorCore import AbstractAgent, AgentDecision, Decision, State


class LCAgent(AbstractAgent):
    agent_id = "Simple1"

    def decide(self, decision: Decision, state: State) -> AgentDecision:
        if not decision.decision:
            raise ValueError("The current scenario must provide at least one option.")

        return AgentDecision(
            agent_id=self.agent_id,
            decision=decision,
            choice=decision.decision[0],
        )