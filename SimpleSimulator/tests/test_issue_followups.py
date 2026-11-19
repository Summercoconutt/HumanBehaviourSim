import unittest

from Agents import LinearClassifierAgent
from Agents.LinearClassifierAgent import LCAgent, car_data_split_indices, load_data
from Agents.PreferenceAgent import (
    COST_SENSITIVE_WEIGHTS,
    SAFETY_SEEKING_WEIGHTS,
    PreferenceAgent,
    PreferenceProfile,
    preference_scores,
    repeated_choices,
)
from Evaluation.three_class import (
    CONTROLLED_BASELINE,
    changed_columns,
    controlled_sensitivity,
    development_cross_validation,
    vary_attribute,
)
from Parameters.LinearClassifierParameters import (
    LinearClassifierParameters,
    SingleLinearClassifierParameters,
)
from Scenarios.CarPurchaseScenario import ACCEPTABILITY_QUESTION, CarPurchaseScenario
from Simulator.SimulatorCore import Decision, State


class SharedModelChecks(unittest.TestCase):
    def test_three_agents_share_one_fit_and_keep_separate_ids(self):
        before = LinearClassifierAgent.three_class_fit_count
        parameters = LinearClassifierParameters()
        self.assertEqual(LinearClassifierAgent.three_class_fit_count, before + 1)
        agents = []
        for configuration in parameters.agent_configurations:
            for _ in range(configuration.count):
                agents.append(configuration.agent_type(**configuration.init_kwargs))
        self.assertEqual(len(agents), 3)
        self.assertEqual(LinearClassifierAgent.three_class_fit_count, before + 1)
        self.assertIs(agents[0].classifier, agents[1].classifier)
        self.assertIs(agents[1].classifier, agents[2].classifier)
        agents[0].agent_id = "left"
        agents[1].agent_id = "right"
        self.assertEqual(agents[0].agent_id, "left")
        self.assertEqual(agents[1].agent_id, "right")

    def test_single_agent_option_uses_the_same_training_size(self):
        single = SingleLinearClassifierParameters()
        default = LinearClassifierParameters()
        self.assertEqual(single.count, 1)
        self.assertEqual(single.model.training_size, default.model.training_size)
        self.assertGreater(single.model.training_size, 1000)


class SensitivityChecks(unittest.TestCase):
    def test_a_pair_changes_exactly_one_attribute(self):
        modified = vary_attribute(CONTROLLED_BASELINE, "Buying_Price", "vhigh")
        self.assertEqual(changed_columns(CONTROLLED_BASELINE, modified), ["Buying_Price"])
        modified_safety = vary_attribute(CONTROLLED_BASELINE, "Safety", "low")
        self.assertEqual(changed_columns(CONTROLLED_BASELINE, modified_safety), ["Safety"])

    def test_sensitivity_reuses_the_fitted_classifier(self):
        agent = LCAgent()
        classifier = agent.classifier
        pairs = controlled_sensitivity(agent.model)
        self.assertIs(agent.classifier, classifier)
        attributes = {pair.attribute for pair in pairs}
        self.assertEqual(attributes, {"Buying_Price", "Safety"})


class DevelopmentProtocolChecks(unittest.TestCase):
    def test_cross_validation_stays_inside_development_data(self):
        scores = development_cross_validation()
        self.assertEqual(len(scores), 5)
        self.assertTrue(all(score.validation_size > 0 for score in scores))
        self.assertTrue(all(0 <= score.logistic_accuracy <= 1 for score in scores))
        self.assertTrue(all(0 <= score.logistic_macro_f1 <= 1 for score in scores))
        self.assertTrue(
            all(score.logistic_macro_f1 > score.majority_macro_f1 for score in scores)
        )


class PreferenceChecks(unittest.TestCase):
    def test_same_inputs_and_seed_match_and_different_weights_diverge(self):
        car = {
            "Buying_Price": "vhigh",
            "Maintenance_Price": "med",
            "No_of_Doors": "4",
            "Person_Capacity": "4",
            "Size_of_Luggage": "med",
            "Safety": "high",
            "Fuel_Type": "petrol",
            "Horse_power": 200.0,
        }
        probabilities = (0.34, 0.33, 0.33)
        cost = preference_scores(probabilities, car, COST_SENSITIVE_WEIGHTS)
        safety = preference_scores(probabilities, car, SAFETY_SEEKING_WEIGHTS)
        self.assertEqual(cost.index(max(cost)), 0)
        self.assertEqual(safety.index(max(safety)), 2)
        profile = PreferenceProfile("cost", dict(COST_SENSITIVE_WEIGHTS), seed=3, noise_scale=0)
        self.assertEqual(
            repeated_choices(profile, car, probabilities),
            repeated_choices(profile, car, probabilities),
        )
        self.assertEqual(len(set(repeated_choices(profile, car, probabilities))), 1)
        noisy = PreferenceProfile("noisy", {"cost": 0.0, "safety": 0.0}, seed=1, noise_scale=3)
        self.assertGreater(len(set(repeated_choices(noisy, car, probabilities, repeats=40))), 1)

    def test_injected_model_is_not_retrained(self):
        model = LinearClassifierParameters().model
        before = LinearClassifierAgent.three_class_fit_count
        agent = PreferenceAgent(
            weights=dict(SAFETY_SEEKING_WEIGHTS),
            model=model,
            name="safety_seeking",
        )
        self.assertEqual(LinearClassifierAgent.three_class_fit_count, before)
        answer = agent.decide(
            Decision(ACCEPTABILITY_QUESTION, [0, 1, 2]),
            State(context=dict(CONTROLLED_BASELINE)),
        )
        again = agent.decide(
            Decision(ACCEPTABILITY_QUESTION, [0, 1, 2]),
            State(context=dict(CONTROLLED_BASELINE)),
        )
        self.assertEqual(answer.choice, again.choice)
        self.assertIn(answer.choice, [0, 1, 2])


class PromptAndProxyChecks(unittest.TestCase):
    def test_prompt_asks_about_acceptability(self):
        decision = CarPurchaseScenario().decisions[0]
        self.assertIn(ACCEPTABILITY_QUESTION, decision.text_body)
        self.assertIn("2 = good", decision.text_body)
        self.assertNotIn("Would you buy", decision.text_body)
        self.assertNotIn("willingness", decision.text_body.lower())

    def test_proxy_groups_are_held_out_and_labelled_as_proxies(self):
        _rows, targets = load_data()
        _train, _val, test_idx = car_data_split_indices(targets)
        test_set = set(int(index) for index in test_idx)
        for name in ("sporty_proxy", "electric_proxy"):
            scenario = CarPurchaseScenario(scenario_type=name)
            self.assertGreater(scenario.sample_count, 0)
            self.assertTrue(set(scenario.selected_indices) <= test_set)
            self.assertIn("proxy", scenario.scenario_type)


if __name__ == "__main__":
    unittest.main()
