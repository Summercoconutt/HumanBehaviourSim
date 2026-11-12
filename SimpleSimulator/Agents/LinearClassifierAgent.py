import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from Simulator.SimulatorCore import AbstractAgent, AgentDecision, Decision, State


FEATURE_COLUMNS = (
    "Buying_Price",
    "Maintenance_Price",
    "No_of_Doors",
    "Person_Capacity",
    "Size_of_Luggage",
    "Safety",
    "Fuel_Type",
    "Horse_power",
)
# Binary acceptability mapping. This is not an observed purchase or a willingness-to-pay label.
# unacc -> 0; acc, good, and vgood -> 1.
LABEL_MAPPING = {"unacc": 0, "acc": 1, "good": 1, "vgood": 1}
POSITIVE_LABELS = {label for label, value in LABEL_MAPPING.items() if value == 1}
DATA_PATH = Path(__file__).resolve().parents[2] / "Data" / "final.csv"

FeatureRow = dict[str, str | float]


def load_data(
    path: Path = DATA_PATH,
) -> tuple[list[FeatureRow], list[int]]:
    with path.open(newline="", encoding="utf-8") as data_file:
        reader = csv.DictReader(data_file, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in dataset: {path}")

        missing_columns = set(FEATURE_COLUMNS + ("Car_Acceptability",)) - set(
            reader.fieldnames
        )
        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing_columns)}"
            )

        features: list[FeatureRow] = []
        targets: list[int] = []
        for row_number, row in enumerate(reader, start=2):
            feature_row: FeatureRow = {}
            for column in FEATURE_COLUMNS[:-1]:
                value = row[column]
                if value is None:
                    raise ValueError(
                        f"Missing {column} value on dataset row {row_number}."
                    )
                feature_row[column] = value
            try:
                feature_row["Horse_power"] = float(row["Horse_power"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid Horse_power value on dataset row {row_number}."
                ) from error

            label = row["Car_Acceptability"]
            if label not in LABEL_MAPPING:
                raise ValueError(
                    f"Unknown Car_Acceptability label on dataset row {row_number}: "
                    f"{label!r}"
                )
            features.append(feature_row)
            targets.append(LABEL_MAPPING[label])

    if not features:
        raise ValueError(f"Dataset contains no data rows: {path}")
    return features, targets


def encode_features(
    features: list[FeatureRow],
    vectorizer: DictVectorizer,
    *,
    fit: bool = False,
) -> NDArray:
    if fit:
        return vectorizer.fit_transform(features)
    return vectorizer.transform(features)


def car_data_split_indices(
    targets: list[int],
) -> tuple[NDArray, NDArray, NDArray]:
    """One shared, deterministic 70/15/15 split for training and scenarios.

    Call with the original row ordering and unchanged binary targets. The test
    partition is never used for fitting either the classifier or vectorizer.
    """
    indices = np.arange(len(targets))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.15,
        random_state=42,
        stratify=targets,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=3 / 17,
        random_state=42,
        stratify=np.asarray(targets)[train_val_idx],
    )
    return train_idx, val_idx, test_idx


def split_data(
    features: NDArray,
    targets: list[int],
) -> tuple[NDArray, NDArray, NDArray, list[int], list[int], list[int]]:
    """Legacy numeric-array interface, now backed by the shared split indices."""
    train_idx, val_idx, test_idx = car_data_split_indices(targets)
    return (
        features[train_idx],
        features[val_idx],
        features[test_idx],
        [targets[i] for i in train_idx],
        [targets[i] for i in val_idx],
        [targets[i] for i in test_idx],
    )


def initialize_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=42)


@dataclass
class BaselineModel:
    """One trained acceptability classifier shared by any number of agents.

    Sharing this object does not give those agents different preferences.
    """

    vectorizer: DictVectorizer
    classifier: LogisticRegression
    training_size: int


baseline_fit_count = 0


def train_baseline_model(
    rows: list[FeatureRow] | None = None,
    targets: list[int] | None = None,
) -> BaselineModel:
    """Fit preprocessing and logistic regression on the training partition only."""
    global baseline_fit_count
    if rows is None or targets is None:
        rows, targets = load_data()
    train_idx, _val_idx, _test_idx = car_data_split_indices(targets)
    training_rows = [rows[int(i)] for i in train_idx]
    training_targets = [targets[int(i)] for i in train_idx]
    vectorizer = DictVectorizer(sparse=False)
    train_features = encode_features(training_rows, vectorizer, fit=True)
    classifier = initialize_classifier()
    classifier.fit(train_features, training_targets)
    baseline_fit_count += 1
    return BaselineModel(
        vectorizer=vectorizer,
        classifier=classifier,
        training_size=len(training_rows),
    )


def calculate_baseline_vote_probability(
    classifier: LogisticRegression,
    vectorizer: DictVectorizer,
    features: FeatureRow,
) -> float:
    encoded_features = encode_features([features], vectorizer)
    positive_class_index = list(classifier.classes_).index(1)
    return float(classifier.predict_proba(encoded_features)[0][positive_class_index])


def convert_probability_to_binary_vote(
    probability: float,
    threshold: float = 0.5,
) -> int:
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    return int(probability >= threshold)


def features_from_context(feature_context: Mapping[str, object]) -> FeatureRow:
    """Coerce one vehicle mapping into the classifier's training representation."""
    missing_features = set(FEATURE_COLUMNS) - feature_context.keys()
    if missing_features:
        raise ValueError(
            f"Scenario context is missing classifier features: "
            f"{sorted(missing_features)}"
        )

    features: FeatureRow = {}
    for column in FEATURE_COLUMNS[:-1]:
        value = feature_context[column]
        if value is None:
            raise ValueError(f"Scenario feature {column} cannot be None.")
        features[column] = str(value)
    try:
        features["Horse_power"] = float(feature_context["Horse_power"])
    except (TypeError, ValueError) as error:
        raise ValueError("Scenario Horse_power must be numeric.") from error
    return features


class LCAgent(AbstractAgent):
    agent_id = "Simple1"

    def __init__(self, model: BaselineModel | None = None) -> None:
        # Reuse a model fitted once per experiment. A missing model trains the baseline.
        self.model = model if model is not None else train_baseline_model()
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
        probability = calculate_baseline_vote_probability(
            self.classifier,
            self.vectorizer,
            features,
        )
        vote = convert_probability_to_binary_vote(probability)
        if vote not in decision.decision:
            raise ValueError(
                "The scenario choices must include the classifier's binary vote "
                "(0 or 1)."
            )

        return AgentDecision(
            agent_id=self.agent_id,
            decision=decision,
            choice=vote,
        )
