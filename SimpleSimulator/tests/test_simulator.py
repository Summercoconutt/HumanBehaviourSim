import unittest

from Agents.SimpleAgent1 import SimpleAgent1
from Parameters.SimpleParameters1 import SimpleParameters
from Scenarios.SimpleScenario1 import SimpleScenario1
from Simulator.SimulatorCore import (
    AbstractAgent,
    AbstractParameters,
    AgentConfiguration,
    AgentDecision,
    Decision,
    SimulatorCore,
)


class ThreeAgents(SimpleParameters):
    @property
    def agent_configurations(self):
        return [AgentConfiguration(SimpleAgent1, 3)]


class UpdatingAgent(AbstractAgent):
    def decide(self, decision, state):
        first_agent_id = next(iter(state.agent_resources))
        choice = state.agent_resources[first_agent_id]["budget"]
        own_budget = state.agent_resources[self.agent_id]["budget"]
        # 这些直接修改只应影响输入副本，不能偷偷修改真实世界状态。
        state.agent_resources[self.agent_id]["history"].append(state.step)
        state.context["history"].append(self.agent_id)
        return AgentDecision(
            agent_id=self.agent_id,
            decision=decision,
            choice=choice,
            resource_updates={self.agent_id: {"budget": own_budget - 1}},
            context_updates={"last_budget": own_budget},
        )


class UpdatingParameters(AbstractParameters):
    @property
    def agent_configurations(self):
        return [AgentConfiguration(UpdatingAgent, 2)]


class UpdatingScenario(SimpleScenario1):
    def __init__(self):
        self.resources = {UpdatingAgent: {"budget": 10, "history": []}}
        self.background = {"history": []}

    @property
    def initial_resources(self):
        return self.resources

    @property
    def context(self):
        return self.background

    @property
    def decisions(self):
        return [Decision("Budget check", [9, 10]) for _ in range(2)]

    def is_complete(self, state):
        return state.step >= 2


class SimulatorChecks(unittest.TestCase):
    def test_original_example_finishes(self):
        core = SimulatorCore()
        result = core(SimpleScenario1(), SimpleParameters())
        agent_count = len(result.final_state.agent_resources)
        self.assertEqual(result.final_state.step, 5)
        self.assertEqual(
            [row.choice for row in result.decision_log], [1] * (5 * agent_count)
        )
        self.assertEqual(core.environment.now, 5)

    def test_multiple_agents_have_distinct_ids_and_fifteen_records(self):
        result = SimulatorCore()(SimpleScenario1(), ThreeAgents())
        self.assertEqual(len(result.final_state.agent_resources), 3)
        self.assertEqual(len(result.decision_log), 15)
        first_round_ids = [row.agent_id for row in result.decision_log[:3]]
        self.assertEqual(len(set(first_round_ids)), 3)

    def test_round_inputs_are_isolated_and_returned_updates_are_applied(self):
        scenario = UpdatingScenario()
        result = SimulatorCore()(scenario, UpdatingParameters())
        self.assertEqual([row.choice for row in result.decision_log], [10, 10, 9, 9])
        for resources in result.final_state.agent_resources.values():
            self.assertEqual(resources, {"budget": 8, "history": []})
        resources = list(result.final_state.agent_resources.values())
        self.assertIsNot(resources[0]["history"], resources[1]["history"])
        self.assertEqual(result.final_state.context, {"history": [], "last_budget": 9})
        self.assertEqual(scenario.resources[UpdatingAgent]["budget"], 10)
        self.assertEqual(scenario.background, {"history": []})

    def test_reusing_core_starts_a_new_run_and_preserves_old_results(self):
        core = SimulatorCore()
        first = core(SimpleScenario1(), SimpleParameters())
        second = core(SimpleScenario1(), ThreeAgents())
        self.assertEqual(
            len(first.decision_log), 5 * len(first.final_state.agent_resources)
        )
        self.assertGreaterEqual(len(first.final_state.agent_resources), 1)
        self.assertEqual(len(second.decision_log), 15)
        self.assertEqual(core.environment.now, 5)
        core.state.context["rounds"] = 100
        self.assertEqual(second.final_state.context["rounds"], 5)

    def test_inconsistent_stop_condition_gives_a_clear_error(self):
        class TooManyRounds(SimpleScenario1):
            def is_complete(self, state):
                return state.step >= 6

        with self.assertRaisesRegex(RuntimeError, "ran out of decisions"):
            SimulatorCore()(TooManyRounds(), SimpleParameters())


if __name__ == "__main__":
    unittest.main()
