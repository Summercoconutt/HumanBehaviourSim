"""Acceptability model with three classes: unacceptable, acceptable, and good.

good includes the original vgood rows. There is no neutral class.
The simulator agent predicts one of 0, 1, or 2.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from Models.data import (
    DATA_PATH,
    FeatureRow,
    car_data_split_indices,
    encode_features,
    features_from_context,
    load_labeled_rows,
)
from Simulator.SimulatorCore import AbstractAgent, AgentDecision, Decision, State


# unacc -> unacceptable; acc -> acceptable; good and vgood -> good.
THREE_CLASS_MAPPING = {"unacc": 0, "acc": 1, "good": 2, "vgood": 2}
THREE_CLASS_NAMES = ("unacceptable", "acceptable", "good")

three_class_fit_count = 0


def class_targets_from_labels(raw_labels: list[str]) -> list[int]:
    return [THREE_CLASS_MAPPING[label] for label in raw_labels]


def load_data(
    path=DATA_PATH,
) -> tuple[list[FeatureRow], list[int]]:
    """Load vehicle rows and the three-class acceptability targets."""
    features, raw_labels = load_labeled_rows(path)
    return features, class_targets_from_labels(raw_labels)


def initialize_three_class_classifier() -> LogisticRegression:
    """Logistic regression for three classes.

    A 1,000-iteration limit did not converge on this fit. The fit reached
    1,339 iterations, so the limit is 2,000.
    """
    return LogisticRegression(max_iter=2000, random_state=42)


@dataclass
class ThreeClassModel:
    """One trained three-class acceptability classifier shared by any number of agents.

    Sharing this object does not give those agents different preferences.
    """

    vectorizer: DictVectorizer
    classifier: LogisticRegression
    training_size: int


def train_three_class_model(
    rows: list[FeatureRow] | None = None,
    raw_labels: list[str] | None = None,
) -> ThreeClassModel:
    """Fit preprocessing and logistic regression on the training partition only.

    The split is stratified on the three-class labels. Test rows are not fitting targets.
    """
    global three_class_fit_count
    if (rows is None) != (raw_labels is None):
        raise ValueError("Provide both vehicle rows and raw labels, or neither.")
    if rows is None or raw_labels is None:
        rows, raw_labels = load_labeled_rows()
    if len(rows) != len(raw_labels):
        raise ValueError("Each vehicle row needs one acceptability label.")

    class_targets = class_targets_from_labels(raw_labels)
    train_idx, _val_idx, _test_idx = car_data_split_indices(class_targets)
    training_rows = [rows[int(index)] for index in train_idx]
    training_targets = [class_targets[int(index)] for index in train_idx]
    if set(training_targets) != {0, 1, 2}:
        raise ValueError(
            "The training partition must contain unacceptable, acceptable, and good."
        )

    vectorizer = DictVectorizer(sparse=False)
    train_features = encode_features(training_rows, vectorizer, fit=True)
    classifier = initialize_three_class_classifier()
    classifier.fit(train_features, training_targets)
    three_class_fit_count += 1
    return ThreeClassModel(
        vectorizer=vectorizer,
        classifier=classifier,
        training_size=len(training_rows),
    )


def score_three_class(
    model: ThreeClassModel,
    features: FeatureRow,
) -> tuple[int, NDArray]:
    """Return the argmax class and probabilities ordered as unacceptable, acceptable, good."""
    encoded_features = encode_features([features], model.vectorizer)
    raw_probabilities = model.classifier.predict_proba(encoded_features)[0]
    ordered = np.zeros(len(THREE_CLASS_NAMES), dtype=float)
    for index, class_id in enumerate(model.classifier.classes_):
        ordered[int(class_id)] = float(raw_probabilities[index])
    if abs(float(ordered.sum()) - 1.0) > 1e-6:
        raise RuntimeError("Three-class probabilities do not cover every class.")
    prediction = int(np.argmax(ordered))
    return prediction, ordered


class LCAgent(AbstractAgent):
    """Simulator agent. It votes 0, 1, or 2 for the three acceptability classes."""

    agent_id = "Simple1"

    def __init__(self, model: ThreeClassModel | None = None) -> None:
        self.model = model if model is not None else train_three_class_model()
        self.vectorizer = self.model.vectorizer
        self.classifier = self.model.classifier

    def decide(self, decision: Decision, state: State) -> AgentDecision:
        if not decision.decision:
            raise ValueError("The current scenario must provide at least one option.")

        feature_context = state.context
        cars = state.context.get("cars")
        if isinstance(cars, list):
            if state.step >= len(cars):
                raise RuntimeError(
                    f"No car data is available for simulation step {state.step}."
                )
            car = cars[state.step]
            if not isinstance(car, dict):
                raise ValueError(
                    f"Car data at simulation step {state.step} must be a mapping."
                )
            feature_context = car

        features = features_from_context(feature_context)
        choice, _probabilities = score_three_class(self.model, features)
        if choice not in decision.decision:
            raise ValueError(
                "The scenario choices must include the predicted class "
                "(0 = unacceptable, 1 = acceptable, 2 = good)."
            )
        return AgentDecision(
            agent_id=self.agent_id,
            decision=decision,
            choice=choice,
        )
