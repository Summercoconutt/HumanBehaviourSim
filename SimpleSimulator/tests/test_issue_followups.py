import unittest

from Agents import LinearClassifierAgent
from Agents.LinearClassifierAgent import (
    LCAgent,
    load_data,
)
from Agents.PreferenceAgent import (
    COST_SENSITIVE_WEIGHTS,
    SAFETY_SEEKING_WEIGHTS,
    PreferenceAgent,
    PreferenceProfile,
    preference_utility,
    repeated_choices,
)
from Evaluation.reports import (
    CONTROLLED_BASELINE,
    changed_columns,
    classification_metrics,
    controlled_sensitivity,
    development_cross_validation,
    evaluate_held_out_groups,
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
        before = LinearClassifierAgent.baseline_fit_count
        parameters = LinearClassifierParameters()
        self.assertEqual(LinearClassifierAgent.baseline_fit_count, before + 1)
        agents = []
        for configuration in parameters.agent_configurations:
            for _ in range(configuration.count):
                agents.append(configuration.agent_type(**configuration.init_kwargs))
        self.assertEqual(len(agents), 3)
        self.assertEqual(LinearClassifierAgent.baseline_fit_count, before + 1)
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
        self.assertEqual(single.model.training_size, 1208)


class PredictiveQualityChecks(unittest.TestCase):
    def test_hand_calculated_confusion_counts(self):
        metrics = classification_metrics(
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0.1, 0.8, 0.7, 0.9],
        )
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 0)
        self.assertEqual(metrics["true_positive"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["predicted_positive_rate"], 0.75)
        self.assertAlmostEqual(metrics["true_positive_prevalence"], 0.5)
        self.assertIsNotNone(metrics["macro_f1"])

    def test_single_class_group_does_not_invent_class_averages(self):
        metrics = classification_metrics([0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3])
        self.assertIsNone(metrics["macro_f1"])
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertGreater(metrics["brier_score"], 0)
        self.assertIn("single true class", metrics["note"])

    def test_holdout_report_separates_prevalence_from_predictions(self):
        groups = {group.scenario_type: group for group in evaluate_held_out_groups(LCAgent())}
        low_safety = groups["low_safety_family"]
        self.assertEqual(low_safety.true_positive_prevalence, 0.0)
        self.assertIsNone(low_safety.macro_f1)
        self.assertGreater(groups["full"].sample_count, groups["budget"].sample_count)
        self.assertIn("sporty_proxy", groups)
        self.assertIn("electric_proxy", groups)


class SensitivityChecks(unittest.TestCase):
    def test_a_pair_changes_exactly_one_attribute(self):
        modified = vary_attribute(CONTROLLED_BASELINE, "Buying_Price", "vhigh")
        self.assertEqual(changed_columns(CONTROLLED_BASELINE, modified), ["Buying_Price"])
        modified_safety = vary_attribute(CONTROLLED_BASELINE, "Safety", "low")
        self.assertEqual(changed_columns(CONTROLLED_BASELINE, modified_safety), ["Safety"])

    def test_sensitivity_reuses_the_fitted_classifier(self):
        agent = LCAgent()
        classifier = agent.classifier
        pairs = controlled_sensitivity(agent)
        self.assertIs(agent.classifier, classifier)
        attributes = {pair.attribute for pair in pairs}
        self.assertEqual(attributes, {"Buying_Price", "Safety"})
        self.assertTrue(any(pair.attribute == "Buying_Price" for pair in pairs))
        self.assertTrue(any(pair.attribute == "Safety" for pair in pairs))


class DevelopmentProtocolChecks(unittest.TestCase):
    def test_cross_validation_stays_inside_development_data(self):
        scores = development_cross_validation()
        self.assertEqual(len(scores), 5)
        self.assertTrue(all(score.validation_size > 0 for score in scores))
        self.assertTrue(all(0 <= score.logistic_brier <= 1 for score in scores))
        self.assertTrue(all(0 <= score.majority_brier <= 1 for score in scores))


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
        cost = preference_utility(0.5, car, COST_SENSITIVE_WEIGHTS)
        safety = preference_utility(0.5, car, SAFETY_SEEKING_WEIGHTS)
        self.assertLess(cost, 0)
        self.assertGreater(safety, 0)
        profile = PreferenceProfile("cost", dict(COST_SENSITIVE_WEIGHTS), seed=3, noise_scale=0)
        self.assertEqual(repeated_choices(profile, car, 0.5), repeated_choices(profile, car, 0.5))
        self.assertEqual(len(set(repeated_choices(profile, car, 0.5))), 1)
        noisy = PreferenceProfile("noisy", {"cost": 0.0, "safety": 0.0}, seed=1, noise_scale=3)
        self.assertEqual(set(repeated_choices(noisy, car, 0.5, repeats=40)), {0, 1})

    def test_injected_model_is_not_retrained(self):
        model = LinearClassifierParameters().model
        before = LinearClassifierAgent.baseline_fit_count
        agent = PreferenceAgent(
            weights=dict(SAFETY_SEEKING_WEIGHTS),
            model=model,
            name="safety_seeking",
        )
        self.assertEqual(LinearClassifierAgent.baseline_fit_count, before)
        answer = agent.decide(
            Decision(ACCEPTABILITY_QUESTION, [0, 1]),
            State(context=dict(CONTROLLED_BASELINE)),
        )
        again = agent.decide(
            Decision(ACCEPTABILITY_QUESTION, [0, 1]),
            State(context=dict(CONTROLLED_BASELINE)),
        )
        self.assertEqual(answer.choice, again.choice)
        self.assertIn(answer.choice, [0, 1])


class PromptAndProxyChecks(unittest.TestCase):
    def test_prompt_asks_about_acceptability(self):
        decision = CarPurchaseScenario().decisions[0]
        self.assertIn(ACCEPTABILITY_QUESTION, decision.text_body)
        self.assertNotIn("Would you buy", decision.text_body)
        self.assertNotIn("willingness", decision.text_body.lower())

    def test_proxy_groups_are_held_out_and_labelled_as_proxies(self):
        _rows, targets = load_data()
        from Agents.LinearClassifierAgent import car_data_split_indices

        _train, _val, test_idx = car_data_split_indices(targets)
        test_set = set(int(index) for index in test_idx)
        for name in ("sporty_proxy", "electric_proxy"):
            scenario = CarPurchaseScenario(scenario_type=name)
            self.assertGreater(scenario.sample_count, 0)
            self.assertTrue(set(scenario.selected_indices) <= test_set)
            self.assertIn("proxy", scenario.scenario_type)


if __name__ == "__main__":
    unittest.main()
