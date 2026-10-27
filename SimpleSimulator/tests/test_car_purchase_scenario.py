import unittest

from Scenarios.CarPurchaseScenario import CarPurchaseScenario
from Parameters.LinearClassifierParameters import LinearClassifierParameters
from Simulator.SimulatorCore import SimulatorCore, State


class CarPurchaseScenarioChecks(unittest.TestCase):
    def test_scenario_has_ten_car_purchase_decisions(self):
        scenario = CarPurchaseScenario()
        decisions = scenario.decisions

        self.assertEqual(len(decisions), 10)
        self.assertEqual(len(scenario.context["cars"]), 10)
        self.assertTrue(all(decision.decision == [0, 1] for decision in decisions))
        self.assertTrue(
            all("Would you buy this car?" in decision.text_body for decision in decisions)
        )
        self.assertTrue(scenario.is_complete(State(step=10)))

    def test_classifier_agents_answer_all_ten_cars(self):
        result = SimulatorCore()(
            CarPurchaseScenario(),
            LinearClassifierParameters(),
        )

        self.assertEqual(result.final_state.step, 10)
        self.assertEqual(len(result.decision_log), 30)
        self.assertTrue(
            all(answer.choice in [0, 1] for answer in result.decision_log)
        )
