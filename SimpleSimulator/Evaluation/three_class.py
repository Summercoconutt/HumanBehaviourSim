"""Three-class acceptability training summary, scenario report, and sensitivity.

The decision is the class with the highest probability. It was not tuned on the holdout.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from Models.data import (
    FEATURE_COLUMNS,
    FeatureRow,
    car_data_split_indices,
    encode_features,
    features_from_context,
    load_labeled_rows,
)
from Models.three_class import (
    THREE_CLASS_NAMES,
    ThreeClassModel,
    class_targets_from_labels,
    initialize_three_class_classifier,
    score_three_class,
    train_three_class_model,
)
from Scenarios.CarPurchaseScenario import COMPARISON_SCENARIOS, CarPurchaseScenario


@dataclass(frozen=True)
class ThreeClassScenarioSummary:
    scenario_type: str
    sample_count: int
    class_counts: tuple[int, int, int]
    mean_probabilities: tuple[float, float, float]


@dataclass(frozen=True)
class PartitionCounts:
    name: str
    sample_count: int
    class_counts: tuple[int, int, int]


def _counts(targets: list[int]) -> tuple[int, int, int]:
    return (
        sum(target == 0 for target in targets),
        sum(target == 1 for target in targets),
        sum(target == 2 for target in targets),
    )


def partition_counts(
    raw_labels: list[str] | None = None,
) -> list[PartitionCounts]:
    """Count three-class labels on the stratified split."""
    if raw_labels is None:
        _rows, raw_labels = load_labeled_rows()
    class_targets = class_targets_from_labels(raw_labels)
    train_idx, val_idx, test_idx = car_data_split_indices(class_targets)
    named = (
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    )
    return [
        PartitionCounts(
            name=name,
            sample_count=len(indices),
            class_counts=_counts([class_targets[int(index)] for index in indices]),
        )
        for name, indices in named
    ]


def score_partition(
    model: ThreeClassModel,
    partition: str,
    rows: list[FeatureRow] | None = None,
    raw_labels: list[str] | None = None,
) -> dict[str, object]:
    """Score one frozen partition. This does not refit the model."""
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be 'validation' or 'test'.")
    if (rows is None) != (raw_labels is None):
        raise ValueError("Provide both vehicle rows and raw labels, or neither.")
    if rows is None or raw_labels is None:
        rows, raw_labels = load_labeled_rows()
    class_targets = class_targets_from_labels(raw_labels)
    _train_idx, val_idx, test_idx = car_data_split_indices(class_targets)
    indices = val_idx if partition == "validation" else test_idx
    y_true: list[int] = []
    y_pred: list[int] = []
    for index in indices:
        prediction, _probabilities = score_three_class(
            model,
            features_from_context(rows[int(index)]),
        )
        y_true.append(class_targets[int(index)])
        y_pred.append(prediction)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "sample_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
        ),
        "confusion": matrix,
    }


def summarize_three_class_scenario(
    scenario: CarPurchaseScenario,
    model: ThreeClassModel,
) -> ThreeClassScenarioSummary:
    """Score every selected car once. The scenario definition is unchanged."""
    if scenario.sample_count == 0:
        raise ValueError("Cannot summarize a scenario with no cars.")
    counts = [0, 0, 0]
    probability_sums = np.zeros(3, dtype=float)
    for car in scenario.cars:
        prediction, probabilities = score_three_class(
            model,
            features_from_context(car),
        )
        counts[prediction] += 1
        probability_sums += probabilities
    sample_count = len(scenario.cars)
    return ThreeClassScenarioSummary(
        scenario_type=scenario.scenario_type,
        sample_count=sample_count,
        class_counts=(counts[0], counts[1], counts[2]),
        mean_probabilities=(
            float(probability_sums[0] / sample_count),
            float(probability_sums[1] / sample_count),
            float(probability_sums[2] / sample_count),
        ),
    )


def compare_three_class_scenarios(
    model: ThreeClassModel | None = None,
) -> list[ThreeClassScenarioSummary]:
    """Score the predefined held-out groups with one three-class model."""
    if model is None:
        model = train_three_class_model()
    return [
        summarize_three_class_scenario(
            CarPurchaseScenario(scenario_type=name),
            model,
        )
        for name in COMPARISON_SCENARIOS
    ]


def _count_line(name: str, sample_count: int, counts: tuple[int, int, int]) -> str:
    return (
        f"{name:<22}{sample_count:>6}{counts[0]:>8}{counts[1]:>8}{counts[2]:>8}"
    )


def _share(summary: ThreeClassScenarioSummary, index: int) -> float:
    return summary.class_counts[index] / summary.sample_count


def _difference_line(
    title: str,
    left: ThreeClassScenarioSummary,
    right: ThreeClassScenarioSummary,
) -> list[str]:
    share_delta = tuple(_share(left, index) - _share(right, index) for index in range(3))
    probability_delta = tuple(
        left.mean_probabilities[index] - right.mean_probabilities[index]
        for index in range(3)
    )
    return [
        (
            f"{title} share difference: "
            f"unacceptable {share_delta[0]:+.4f}, "
            f"acceptable {share_delta[1]:+.4f}, good {share_delta[2]:+.4f}"
        ),
        (
            f"{title} mean probability difference: "
            f"unacceptable {probability_delta[0]:+.4f}, "
            f"acceptable {probability_delta[1]:+.4f}, good {probability_delta[2]:+.4f}"
        ),
    ]


def format_three_class_report(
    model: ThreeClassModel | None = None,
) -> str:
    """One training summary plus the scenario class distributions."""
    rows, raw_labels = load_labeled_rows()
    if model is None:
        model = train_three_class_model(rows, raw_labels)
    validation = score_partition(model, "validation", rows, raw_labels)
    holdout = score_partition(model, "test", rows, raw_labels)
    summaries = compare_three_class_scenarios(model)
    by_name = {summary.scenario_type: summary for summary in summaries}
    validation_matrix: NDArray = validation["confusion"]  # type: ignore[assignment]
    holdout_matrix: NDArray = holdout["confusion"]  # type: ignore[assignment]

    lines = [
        "Three-class acceptability model",
        "Label map: unacc = unacceptable (0); acc = acceptable (1); good and vgood = good (2).",
        "There is no neutral class. vgood is grouped with good.",
        "Split: stratified 70/15/15 on the three-class labels, random_state=42.",
        "Fit: DictVectorizer and logistic regression on the training rows only.",
        (
            "Iteration limit: 2,000. This fit used "
            f"{int(model.classifier.n_iter_.max())} iterations."
        ),
        "Decision rule: the class with the highest probability. It was not tuned on the holdout.",
        "The simulator agent uses this same three-class model.",
        (
            f"Training rows: {model.training_size}. "
            f"Classes: {', '.join(THREE_CLASS_NAMES)}."
        ),
        "",
        "Class counts on the frozen split",
        f"{'split':<22}{'n':>6}{'unacc':>8}{'acc':>8}{'good':>8}",
    ]
    for partition in partition_counts(raw_labels):
        lines.append(
            _count_line(partition.name, partition.sample_count, partition.class_counts)
        )
    lines.extend(
        [
            "",
            "Development validation (these rows were not used to fit this model)",
            (
                f"n={validation['sample_count']}  "
                f"accuracy={validation['accuracy']:.4f}  "
                f"macro_f1={validation['macro_f1']:.4f}"
            ),
            "Confusion matrix, rows are true class, columns are predicted class.",
            "Order: unacceptable, acceptable, good.",
            "  " + " ".join(str(int(value)) for value in validation_matrix[0]),
            "  " + " ".join(str(int(value)) for value in validation_matrix[1]),
            "  " + " ".join(str(int(value)) for value in validation_matrix[2]),
            "",
            "EXPLORATORY full holdout classification (the same 260 test rows as scenario full)",
            "These rows were already used to design the scenarios, so this is not a fresh final exam.",
            (
                f"n={holdout['sample_count']}  "
                f"accuracy={holdout['accuracy']:.4f}  "
                f"macro_f1={holdout['macro_f1']:.4f}"
            ),
            "Order: unacceptable, acceptable, good.",
            "  " + " ".join(str(int(value)) for value in holdout_matrix[0]),
            "  " + " ".join(str(int(value)) for value in holdout_matrix[1]),
            "  " + " ".join(str(int(value)) for value in holdout_matrix[2]),
            "",
            "EXPLORATORY held-out scenario class counts",
            "These test rows were already used to design the scenarios.",
            "Counts are argmax decisions. Shares, not raw counts, are compared across groups of different sizes.",
            "Mean probabilities are the average class probabilities.",
            (
                f"{'scenario':<22}{'n':>6}{'unacc':>8}{'acc':>8}{'good':>8}"
                f"{'p_unacc':>10}{'p_acc':>10}{'p_good':>10}"
            ),
        ]
    )
    for summary in summaries:
        lines.append(
            f"{summary.scenario_type:<22}{summary.sample_count:>6}"
            f"{summary.class_counts[0]:>8}{summary.class_counts[1]:>8}"
            f"{summary.class_counts[2]:>8}{summary.mean_probabilities[0]:>10.4f}"
            f"{summary.mean_probabilities[1]:>10.4f}{summary.mean_probabilities[2]:>10.4f}"
        )
    lines.extend(
        _difference_line(
            "budget minus premium",
            by_name["budget"],
            by_name["premium"],
        )
    )
    lines.extend(
        _difference_line(
            "safe_family minus low_safety_family",
            by_name["safe_family"],
            by_name["low_safety_family"],
        )
    )
    lines.extend(
        [
            "The same three-class model produces different class distributions across these scenarios.",
            "This phase shows the concept on the current vehicle table.",
            "A later study would need properly sourced choice data and a longer experiment.",
            "Scores are predicted acceptability classes, not observed purchases.",
        ]
    )
    return "\n".join(lines)


CONTROLLED_BASELINE: FeatureRow = {
    "Buying_Price": "med",
    "Maintenance_Price": "med",
    "No_of_Doors": "4",
    "Person_Capacity": "4",
    "Size_of_Luggage": "med",
    "Safety": "med",
    "Fuel_Type": "petrol",
    "Horse_power": 200.0,
}


def changed_columns(left: FeatureRow, right: FeatureRow) -> list[str]:
    return [column for column in FEATURE_COLUMNS if left[column] != right[column]]


def vary_attribute(car: FeatureRow, column: str, value: str | float) -> FeatureRow:
    if column not in FEATURE_COLUMNS:
        raise ValueError(f"Unknown vehicle attribute {column!r}.")
    updated = dict(car)
    updated[column] = value
    changed = changed_columns(car, updated)
    if changed != [column]:
        raise ValueError(f"Expected only {column} to change, got {changed}.")
    return updated


@dataclass(frozen=True)
class SensitivityPair:
    attribute: str
    baseline_value: str
    modified_value: str
    baseline_class: int
    modified_class: int
    baseline_probabilities: tuple[float, float, float]
    modified_probabilities: tuple[float, float, float]

    @property
    def flipped(self) -> bool:
        return self.baseline_class != self.modified_class


def score_pair(
    model: ThreeClassModel,
    baseline: FeatureRow,
    column: str,
    modified_value: str | float,
) -> SensitivityPair:
    modified = vary_attribute(baseline, column, modified_value)
    baseline_class, baseline_probabilities = score_three_class(
        model,
        features_from_context(baseline),
    )
    modified_class, modified_probabilities = score_three_class(
        model,
        features_from_context(modified),
    )
    return SensitivityPair(
        attribute=column,
        baseline_value=str(baseline[column]),
        modified_value=str(modified_value),
        baseline_class=baseline_class,
        modified_class=modified_class,
        baseline_probabilities=tuple(float(value) for value in baseline_probabilities),
        modified_probabilities=tuple(float(value) for value in modified_probabilities),
    )


def controlled_sensitivity(
    model: ThreeClassModel | None = None,
) -> list[SensitivityPair]:
    """Change one constructed-car attribute at a time with a frozen classifier."""
    if model is None:
        model = train_three_class_model()
    baseline = dict(CONTROLLED_BASELINE)
    pairs = [
        score_pair(model, baseline, "Buying_Price", level)
        for level in ("low", "high", "vhigh")
    ]
    pairs.extend(
        score_pair(model, baseline, "Safety", level) for level in ("low", "high")
    )
    return pairs


def format_sensitivity(pairs: list[SensitivityPair]) -> str:
    lines = [
        "Three-class sensitivity on one constructed vehicle",
        "Exactly one attribute changes. This is model sensitivity, not observed consumer behaviour.",
        "Classes: 0 = unacceptable, 1 = acceptable, 2 = good.",
        f"{'attribute':<16}{'from':>8}{'to':>8}{'class':>8}{'new':>6}{'flip':>8}{'dp_good':>10}",
    ]
    for pair in pairs:
        good_delta = pair.modified_probabilities[2] - pair.baseline_probabilities[2]
        lines.append(
            f"{pair.attribute:<16}{pair.baseline_value:>8}{pair.modified_value:>8}"
            f"{pair.baseline_class:>8}{pair.modified_class:>6}"
            f"{str(pair.flipped):>8}{good_delta:>10.4f}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class FoldScore:
    fold: int
    validation_size: int
    logistic_accuracy: float
    logistic_macro_f1: float
    majority_accuracy: float
    majority_macro_f1: float


def development_cross_validation(n_splits: int = 5) -> list[FoldScore]:
    """Compare three-class logistic regression with a majority baseline.

    The check uses only the training and validation rows. Each fold fits its own vectorizer.
    """
    rows, raw_labels = load_labeled_rows()
    targets = class_targets_from_labels(raw_labels)
    train_idx, val_idx, test_idx = car_data_split_indices(targets)
    development = np.concatenate([train_idx, val_idx])
    test_set = {int(index) for index in test_idx}
    if test_set & {int(index) for index in development}:
        raise RuntimeError("Development data overlaps the frozen test partition.")
    labels = np.asarray(targets)[development]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: list[FoldScore] = []
    for fold_number, (train_pos, valid_pos) in enumerate(
        splitter.split(development, labels),
        start=1,
    ):
        train_ids = [int(development[position]) for position in train_pos]
        valid_ids = [int(development[position]) for position in valid_pos]
        if test_set & set(train_ids + valid_ids):
            raise RuntimeError("A development fold contains a frozen test row.")
        vectorizer = DictVectorizer(sparse=False)
        train_matrix = encode_features(
            [rows[index] for index in train_ids],
            vectorizer,
            fit=True,
        )
        classifier = initialize_three_class_classifier()
        train_labels = [targets[index] for index in train_ids]
        classifier.fit(train_matrix, train_labels)
        valid_matrix = encode_features(
            [rows[index] for index in valid_ids],
            vectorizer,
            fit=False,
        )
        valid_labels = [targets[index] for index in valid_ids]
        predictions = classifier.predict(valid_matrix)
        majority = max(set(train_labels), key=train_labels.count)
        majority_predictions = [majority] * len(valid_labels)
        scores.append(
            FoldScore(
                fold=fold_number,
                validation_size=len(valid_labels),
                logistic_accuracy=float(accuracy_score(valid_labels, predictions)),
                logistic_macro_f1=float(
                    f1_score(
                        valid_labels,
                        predictions,
                        average="macro",
                        labels=[0, 1, 2],
                        zero_division=0,
                    )
                ),
                majority_accuracy=float(accuracy_score(valid_labels, majority_predictions)),
                majority_macro_f1=float(
                    f1_score(
                        valid_labels,
                        majority_predictions,
                        average="macro",
                        labels=[0, 1, 2],
                        zero_division=0,
                    )
                ),
            )
        )
    return scores


def format_development_cv(scores: list[FoldScore]) -> str:
    logistic_accuracy = sum(score.logistic_accuracy for score in scores) / len(scores)
    majority_accuracy = sum(score.majority_accuracy for score in scores) / len(scores)
    logistic_f1 = sum(score.logistic_macro_f1 for score in scores) / len(scores)
    majority_f1 = sum(score.majority_macro_f1 for score in scores) / len(scores)
    lines = [
        "Development-set stratified 5-fold check (training + validation only)",
        "The decision is argmax. It was not chosen from the holdout.",
        f"{'fold':>6}{'n':>6}{'log_acc':>10}{'log_f1':>10}{'maj_acc':>10}{'maj_f1':>10}",
    ]
    for score in scores:
        lines.append(
            f"{score.fold:>6}{score.validation_size:>6}"
            f"{score.logistic_accuracy:>10.4f}{score.logistic_macro_f1:>10.4f}"
            f"{score.majority_accuracy:>10.4f}{score.majority_macro_f1:>10.4f}"
        )
    lines.append(
        f"mean logistic accuracy {logistic_accuracy:.4f}, majority {majority_accuracy:.4f}"
    )
    lines.append(
        f"mean logistic macro F1 {logistic_f1:.4f}, majority {majority_f1:.4f}"
    )
    lines.append("This check does not use the frozen test rows.")
    return "\n".join(lines)


def format_full_evaluation(model: ThreeClassModel | None = None) -> str:
    """Scenario comparison, holdout quality, and the development-set check."""
    if model is None:
        model = train_three_class_model()
    return "\n\n".join(
        [
            format_three_class_report(model),
            format_development_cv(development_cross_validation()),
            format_sensitivity(controlled_sensitivity(model)),
        ]
    )
