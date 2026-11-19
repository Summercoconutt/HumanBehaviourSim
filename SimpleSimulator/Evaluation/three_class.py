"""Three-class acceptability training summary and scenario report.

The binary baseline remains the deployed scenario agent. This report scores the
same held-out groups with a separate three-class model. The decision is the
argmax class. It was not tuned on the holdout.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from Agents.LinearClassifierAgent import (
    LABEL_MAPPING,
    THREE_CLASS_NAMES,
    THREE_CLASS_MAPPING,
    FeatureRow,
    ThreeClassModel,
    car_data_split_indices,
    features_from_context,
    load_labeled_rows,
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
    """Count three-class labels on the frozen binary split."""
    if raw_labels is None:
        _rows, raw_labels = load_labeled_rows()
    binary_targets = [LABEL_MAPPING[label] for label in raw_labels]
    class_targets = [THREE_CLASS_MAPPING[label] for label in raw_labels]
    train_idx, val_idx, test_idx = car_data_split_indices(binary_targets)
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
    binary_targets = [LABEL_MAPPING[label] for label in raw_labels]
    class_targets = [THREE_CLASS_MAPPING[label] for label in raw_labels]
    _train_idx, val_idx, test_idx = car_data_split_indices(binary_targets)
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
        "Split: the frozen binary 70/15/15 split. Three-class labels did not redraw it.",
        "Fit: DictVectorizer and logistic regression on the training rows only.",
        (
            "Iteration limit: 2,000. This fit used "
            f"{int(model.classifier.n_iter_.max())} iterations. "
            "The binary model still uses 1,000."
        ),
        "Decision rule: the class with the highest probability. It was not tuned on the holdout.",
        "The binary 0/1 model is unchanged and remains the deployed scenario agent.",
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
