import unittest

from Agents.LinearClassifierAgent import (
    LCAgent,
    car_data_split_indices,
    features_from_context,
    load_data,
    score_three_class,
)
from Scenarios.CarPurchaseScenario import (
    SCENARIO_FILTERS,
    CarPurchaseScenario,
    compare_held_out_scenarios,
    comparison_deltas,
    summarize_scenario,
)
from Simulator.SimulatorCore import Decision, State


class ConfigurableScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.targets = load_data()
        cls.train_idx, cls.val_idx, cls.test_idx = car_data_split_indices(cls.targets)

    def test_splits_are_disjoint_and_cover_all_rows(self):
        train, val, test = map(set, (self.train_idx, self.val_idx, self.test_idx))
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(len(train | val | test), len(self.rows))

    def test_default_remains_ten_random_held_out_cars(self):
        scenario = CarPurchaseScenario()
        self.assertEqual(scenario.sample_count, 10)
        self.assertEqual(len(scenario.decisions), 10)
        self.assertEqual(len(set(scenario.selected_indices)), 10)
        self.assertTrue(set(scenario.selected_indices) <= set(self.test_idx))

    def test_seed_reproducibility(self):
        a = CarPurchaseScenario(sample_count=12, random_state=7)
        b = CarPurchaseScenario(sample_count=12, random_state=7)
        c = CarPurchaseScenario(sample_count=12, random_state=8)
        self.assertEqual(a.selected_indices, b.selected_indices)
        self.assertNotEqual(a.selected_indices, c.selected_indices)

    def test_named_scenarios_use_only_held_out_rows(self):
        for name in SCENARIO_FILTERS:
            with self.subTest(name=name):
                scenario = CarPurchaseScenario(scenario_type=name)
                self.assertTrue(set(scenario.selected_indices) <= set(self.test_idx))
                self.assertTrue(all(SCENARIO_FILTERS[name](car) for car in scenario.cars))
                self.assertEqual(scenario.sample_count, len(scenario.decisions))
                self.assertTrue(scenario.is_complete(State(step=scenario.sample_count)))

    def test_full_scenario_covers_entire_held_out_set(self):
        scenario = CarPurchaseScenario(scenario_type="full")
        self.assertEqual(set(scenario.selected_indices), set(self.test_idx))

    def test_sample_count_and_invalid_filters(self):
        scenario = CarPurchaseScenario(scenario_type="budget", sample_count=8)
        self.assertEqual(scenario.sample_count, 8)
        with self.assertRaisesRegex(ValueError, "Unknown scenario_type"):
            CarPurchaseScenario(scenario_type="unknown")
        with self.assertRaisesRegex(ValueError, "positive"):
            CarPurchaseScenario(sample_count=0)
        with self.assertRaisesRegex(ValueError, "contains only"):
            CarPurchaseScenario(scenario_type="budget", sample_count=9999)

    def test_held_out_comparison_uses_one_model_and_matches_votes(self):
        agent = LCAgent()
        first = compare_held_out_scenarios(agent)
        second = compare_held_out_scenarios(agent)
        self.assertEqual(
            [summary.scenario_type for summary in first],
            ["full", "budget", "premium", "safe_family", "low_safety_family"],
        )
        for summary in first:
            self.assertGreater(summary.sample_count, 0)
            self.assertEqual(sum(summary.class_counts), summary.sample_count)
            self.assertAlmostEqual(sum(summary.mean_probabilities), 1.0, places=6)
        self.assertEqual(first[0].sample_count, len(self.test_idx))
        self.assertEqual(first, second)

        scenario = CarPurchaseScenario(scenario_type="budget")
        self.assertTrue(set(scenario.selected_indices) <= set(self.test_idx))
        summary = summarize_scenario(scenario, agent)
        recounted = [0, 0, 0]
        for car in scenario.cars:
            prediction, _probabilities = score_three_class(
                agent.model,
                features_from_context(car),
            )
            answer = agent.decide(
                Decision("Car acceptability", [0, 1, 2]),
                State(context=car),
            )
            recounted[prediction] += 1
            self.assertEqual(answer.choice, prediction)
        self.assertEqual(summary.class_counts, tuple(recounted))
        deltas = comparison_deltas(first)
        self.assertEqual(
            set(deltas),
            {"budget_minus_premium", "safe_family_minus_low_safety_family"},
        )
        for share_delta, probability_delta in deltas.values():
            self.assertGreaterEqual(share_delta, -1.0)
            self.assertLessEqual(share_delta, 1.0)
            self.assertGreaterEqual(probability_delta, -1.0)
            self.assertLessEqual(probability_delta, 1.0)


if __name__ == "__main__":
    unittest.main()
