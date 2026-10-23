import unittest

import bootstrap
from parameters import DaoParameters
from scenario import UniswapGovernanceScenario
from Simulator.SimulatorCore import SimulatorCore


class UniswapProposalChecks(unittest.TestCase):
    def test_three_proposals_accept_the_first_option(self):
        scenario = UniswapGovernanceScenario()
        self.assertEqual(
            [list(proposal.choices) for proposal in scenario.proposals],
            [[1, 2], [1, 2], [1, 2, 3]],
        )
        for proposal in scenario.proposals:
            self.assertEqual(proposal.choices[0], 1)
            self.assertGreater(proposal.votes_by_choice[1], 0)
            self.assertTrue(proposal.title)
            self.assertTrue(proposal.body)

        result = SimulatorCore()(scenario, DaoParameters())
        agent_count = len(result.final_state.agent_resources)
        self.assertEqual(result.final_state.step, 3)
        self.assertEqual(len(result.decision_log), 3 * agent_count)
        self.assertTrue(all(row.choice == 1 for row in result.decision_log))
        self.assertEqual(result.final_state.context["space"], "uniswapgovernance.eth")
        self.assertEqual(len(result.final_state.context["proposals"]), 3)


if __name__ == "__main__":
    unittest.main()
