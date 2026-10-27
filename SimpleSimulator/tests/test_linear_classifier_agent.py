import unittest

from sklearn.feature_extraction import DictVectorizer

from Agents.LinearClassifierAgent import (
    LCAgent,
    convert_probability_to_binary_vote,
    encode_features,
    load_data,
    split_data,
)
from Simulator.SimulatorCore import Decision, State


class LinearClassifierAgentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = LCAgent()

    def test_dataset_has_binary_targets_and_fitted_features(self):
        features, targets = load_data()
        self.assertEqual(len(features), len(targets))
        self.assertEqual(set(targets), {0, 1})
        self.assertEqual(len(self.agent.vectorizer.get_feature_names_out()), 26)
        encoded_features = encode_features(
            features,
            DictVectorizer(sparse=False),
            fit=True,
        )
        splits = split_data(encoded_features, targets)
        self.assertEqual(
            sum(len(split) for split in splits[:3]),
            len(features),
        )

    def test_decision_returns_a_binary_choice(self):
        state = State(
            context={
                "Buying_Price": "vhigh",
                "Maintenance_Price": "vhigh",
                "No_of_Doors": "2",
                "Person_Capacity": "2",
                "Size_of_Luggage": "small",
                "Safety": "low",
                "Fuel_Type": "petrol",
                "Horse_power": 238,
            }
        )
        answer = self.agent.decide(Decision("Car acceptability", [0, 1]), state)
        self.assertIn(answer.choice, [0, 1])

    def test_binary_vote_uses_probability_threshold(self):
        self.assertEqual(convert_probability_to_binary_vote(0.49), 0)
        self.assertEqual(convert_probability_to_binary_vote(0.5), 1)

    def test_decision_requires_car_features_in_context(self):
        with self.assertRaisesRegex(ValueError, "missing classifier features"):
            self.agent.decide(Decision("Car acceptability", [0, 1]), State())


if __name__ == "__main__":
    unittest.main()
