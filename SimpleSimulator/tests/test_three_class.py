import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from Agents import LinearClassifierAgent
from Agents.LinearClassifierAgent import (
    THREE_CLASS_MAPPING,
    THREE_CLASS_NAMES,
    LCAgent,
    encode_features,
    features_from_context,
    initialize_three_class_classifier,
    load_data,
    load_labeled_rows,
    score_three_class,
    train_three_class_model,
)
from Evaluation.three_class import (
    compare_three_class_scenarios,
    format_three_class_report,
    partition_counts,
)
from Scenarios.CarPurchaseScenario import COMPARISON_SCENARIOS, CarPurchaseScenario


class ThreeClassModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.raw_labels = load_labeled_rows()
        cls.model = train_three_class_model(cls.rows, cls.raw_labels)

    def test_label_map_groups_good_and_vgood_and_adds_no_neutral_class(self):
        self.assertEqual(THREE_CLASS_NAMES, ("unacceptable", "acceptable", "good"))
        self.assertEqual(set(THREE_CLASS_MAPPING), {"unacc", "acc", "good", "vgood"})
        self.assertEqual(THREE_CLASS_MAPPING["unacc"], 0)
        self.assertEqual(THREE_CLASS_MAPPING["acc"], 1)
        self.assertEqual(THREE_CLASS_MAPPING["good"], 2)
        self.assertEqual(THREE_CLASS_MAPPING["vgood"], 2)
        self.assertNotIn("neutral", THREE_CLASS_NAMES)

    def test_loaded_targets_and_agent_use_three_classes(self):
        _features, targets = load_data()
        self.assertEqual(set(targets), {0, 1, 2})
        before = LinearClassifierAgent.three_class_fit_count
        agent = LCAgent()
        self.assertEqual(LinearClassifierAgent.three_class_fit_count, before + 1)
        self.assertEqual(list(agent.classifier.classes_), [0, 1, 2])

    def test_three_class_fit_uses_the_training_partition_only(self):
        from Agents.LinearClassifierAgent import car_data_split_indices

        class_targets = [THREE_CLASS_MAPPING[label] for label in self.raw_labels]
        train_idx, val_idx, _test_idx = car_data_split_indices(class_targets)
        self.assertEqual(self.model.training_size, len(train_idx))
        self.assertEqual(list(self.model.classifier.classes_), [0, 1, 2])
        self.assertLess(int(self.model.classifier.n_iter_.max()), 2000)

        train_val_rows = [self.rows[int(index)] for index in np.concatenate([train_idx, val_idx])]
        train_val_targets = [
            class_targets[int(index)] for index in np.concatenate([train_idx, val_idx])
        ]
        wider = initialize_three_class_classifier()
        wider.fit(
            encode_features(train_val_rows, self.model.vectorizer),
            train_val_targets,
        )
        self.assertIsInstance(wider, LogisticRegression)
        self.assertFalse(np.allclose(self.model.classifier.coef_, wider.coef_))

    def test_training_once_serves_every_scenario(self):
        before = LinearClassifierAgent.three_class_fit_count
        summaries = compare_three_class_scenarios(self.model)
        self.assertEqual(LinearClassifierAgent.three_class_fit_count, before)
        self.assertEqual(
            [summary.scenario_type for summary in summaries],
            list(COMPARISON_SCENARIOS),
        )

    def test_scenario_counts_cover_each_group_and_the_groups_differ(self):
        from Agents.LinearClassifierAgent import car_data_split_indices

        class_targets = [THREE_CLASS_MAPPING[label] for label in self.raw_labels]
        _train_idx, _val_idx, test_idx = car_data_split_indices(class_targets)
        summaries = compare_three_class_scenarios(self.model)
        count_rows = []
        for summary in summaries:
            scenario = CarPurchaseScenario(scenario_type=summary.scenario_type)
            self.assertTrue(set(scenario.selected_indices) <= set(test_idx))
            self.assertEqual(summary.sample_count, scenario.sample_count)
            self.assertEqual(sum(summary.class_counts), summary.sample_count)
            self.assertTrue(all(count >= 0 for count in summary.class_counts))
            self.assertAlmostEqual(sum(summary.mean_probabilities), 1.0, places=6)
            recounted = [0, 0, 0]
            for car in scenario.cars:
                prediction, _probabilities = score_three_class(
                    self.model,
                    features_from_context(car),
                )
                self.assertIn(prediction, (0, 1, 2))
                recounted[prediction] += 1
            self.assertEqual(summary.class_counts, tuple(recounted))
            count_rows.append(summary.class_counts)
        self.assertGreater(len(set(count_rows)), 1)

    def test_report_states_the_map_and_separates_prediction_from_purchase(self):
        report = format_three_class_report(self.model)
        self.assertIn("good and vgood = good", report)
        self.assertIn("no neutral class", report)
        self.assertIn("not observed purchases", report)
        self.assertIn("simulator agent uses this same three-class model", report)
        self.assertIn("safe_family", report)
        self.assertIn("low_safety_family", report)
        self.assertIn("share difference", report)
        self.assertIn("EXPLORATORY full holdout", report)
        for partition in partition_counts(self.raw_labels):
            self.assertEqual(sum(partition.class_counts), partition.sample_count)
        self.assertEqual(
            [partition.sample_count for partition in partition_counts(self.raw_labels)],
            [1208, 260, 260],
        )


if __name__ == "__main__":
    unittest.main()
