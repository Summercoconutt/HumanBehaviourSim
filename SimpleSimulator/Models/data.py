"""Shared vehicle table, features, and the frozen train/validation/test split."""

import csv
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


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
RAW_LABELS = frozenset({"unacc", "acc", "good", "vgood"})
DATA_PATH = Path(__file__).resolve().parents[2] / "Data" / "final.csv"

FeatureRow = dict[str, str | float]


def load_labeled_rows(
    path: Path = DATA_PATH,
) -> tuple[list[FeatureRow], list[str]]:
    """Load vehicle rows and the original Car_Acceptability strings."""
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
        raw_labels: list[str] = []
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
            if label not in RAW_LABELS:
                raise ValueError(
                    f"Unknown Car_Acceptability label on dataset row {row_number}: "
                    f"{label!r}"
                )
            features.append(feature_row)
            raw_labels.append(label)

    if not features:
        raise ValueError(f"Dataset contains no data rows: {path}")
    return features, raw_labels


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

    Call with the original row ordering and the three-class targets. The test
    partition is never used for fitting the classifier or the vectorizer.
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
