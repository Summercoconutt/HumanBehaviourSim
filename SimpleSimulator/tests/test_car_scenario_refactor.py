import unittest

from Agents.LinearClassifierAgent import car_data_split_indices, load_data
from Scenarios.CarPurchaseScenario import CarPurchaseScenario, SCENARIO_FILTERS
from Simulator.SimulatorCore import State


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


if __name__ == "__main__":
    unittest.main()
