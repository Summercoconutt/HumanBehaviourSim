"""Predictive-quality, sensitivity, and development-set reports.

The published holdout scenario table was already inspected while the scenarios
were designed. Metrics on that same holdout are therefore exploratory. Model
comparison and the fixed 0.5 threshold belong to the development split.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold

from Agents.LinearClassifierAgent import (
    FEATURE_COLUMNS,
    LCAgent,
    FeatureRow,
    calculate_baseline_vote_probability,
    car_data_split_indices,
    convert_probability_to_binary_vote,
    encode_features,
    features_from_context,
    initialize_classifier,
    load_data,
)
from Scenarios.CarPurchaseScenario import (
    COMPARISON_SCENARIOS,
    SCENARIO_FILTERS,
    CarPurchaseScenario,
    format_scenario_comparison,
    summarize_scenario,
)


PROBABILITY_THRESHOLD = 0.5
PROXY_SCENARIOS = ("sporty_proxy", "electric_proxy")
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


@dataclass(frozen=True)
class GroupQuality:
    scenario_type: str
    sample_count: int
    true_positive_prevalence: float
    predicted_positive_rate: float
    accuracy: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    macro_f1: float | None
    balanced_accuracy: float | None
    brier_score: float
    log_loss: float
    note: str


def _clip(probabilities: list[float]) -> NDArray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float],
) -> dict[str, object]:
    """Score one group. Single-class groups leave class-average metrics unset."""
    if not y_true:
        raise ValueError("Cannot score an empty group.")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    clipped = _clip(y_prob)
    present_classes = set(y_true)
    note = ""
    macro_f1: float | None
    balanced_accuracy: float | None
    if len(present_classes) < 2:
        macro_f1 = None
        balanced_accuracy = None
        note = "Macro F1 and balanced accuracy are undefined for a single true class."
    else:
        macro_f1 = float(
            f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)
        )
        balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "sample_count": len(y_true),
        "true_positive_prevalence": sum(y_true) / len(y_true),
        "predicted_positive_rate": sum(y_pred) / len(y_pred),
        "accuracy": (true_negative + true_positive) / len(y_true),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "note": note,
    }


def score_cars(
    agent: LCAgent,
    cars: list[FeatureRow],
) -> tuple[list[float], list[int]]:
    probabilities: list[float] = []
    votes: list[int] = []
    for car in cars:
        probability = calculate_baseline_vote_probability(
            agent.classifier,
            agent.vectorizer,
            features_from_context(car),
        )
        probabilities.append(probability)
        votes.append(convert_probability_to_binary_vote(probability, PROBABILITY_THRESHOLD))
    return probabilities, votes


def evaluate_scenario(
    scenario: CarPurchaseScenario,
    agent: LCAgent,
    targets: list[int],
) -> GroupQuality:
    y_true = [targets[index] for index in scenario.selected_indices]
    y_prob, y_pred = score_cars(agent, scenario.cars)
    metrics = classification_metrics(y_true, y_pred, y_prob)
    return GroupQuality(scenario_type=scenario.scenario_type, **metrics)  # type: ignore[arg-type]


def evaluate_held_out_groups(
    agent: LCAgent | None = None,
) -> list[GroupQuality]:
    """Exploratory predictive quality on the already inspected holdout."""
    rows, targets = load_data()
    del rows
    if agent is None:
        agent = LCAgent()
    names = (*COMPARISON_SCENARIOS, *PROXY_SCENARIOS)
    return [
        evaluate_scenario(CarPurchaseScenario(scenario_type=name), agent, targets)
        for name in names
    ]


def _format_optional(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.4f}"


def format_quality_report(groups: list[GroupQuality]) -> str:
    lines = [
        "EXPLORATORY holdout predictive quality",
        "These rows were already used to design the scenarios, so this is not a fresh final exam.",
        "Predicted positive rate is the share of probabilities >= 0.5. It is not accuracy.",
        "Overlapping scenario groups are not independent samples.",
        (
            f"{'scenario':<22}{'n':>6}{'prevalence':>12}{'pred_rate':>12}"
            f"{'accuracy':>10}{'macro_f1':>12}{'bal_acc':>12}{'brier':>10}"
        ),
    ]
    for group in groups:
        lines.append(
            f"{group.scenario_type:<22}{group.sample_count:>6}"
            f"{group.true_positive_prevalence:>12.4f}{group.predicted_positive_rate:>12.4f}"
            f"{group.accuracy:>10.4f}{_format_optional(group.macro_f1):>12}"
            f"{_format_optional(group.balanced_accuracy):>12}{group.brier_score:>10.4f}"
        )
        lines.append(
            "  confusion [tn fp / fn tp] = "
            f"[{group.true_negative} {group.false_positive} / "
            f"{group.false_negative} {group.true_positive}]"
            f"  log_loss={group.log_loss:.4f}"
        )
        if group.note:
            lines.append(f"  note: {group.note}")
    lines.append(
        "Labels are Car_Acceptability (unacc=0, acc/good/vgood=1), not observed purchases."
    )
    return "\n".join(lines)


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
    baseline_probability: float
    modified_probability: float
    delta_probability: float
    baseline_vote: int
    modified_vote: int

    @property
    def flipped(self) -> bool:
        return self.baseline_vote != self.modified_vote


def score_pair(
    agent: LCAgent,
    baseline: FeatureRow,
    column: str,
    modified_value: str | float,
) -> SensitivityPair:
    modified = vary_attribute(baseline, column, modified_value)
    baseline_probability, modified_probability = score_cars(agent, [baseline, modified])[0]
    baseline_vote = convert_probability_to_binary_vote(baseline_probability)
    modified_vote = convert_probability_to_binary_vote(modified_probability)
    return SensitivityPair(
        attribute=column,
        baseline_value=str(baseline[column]),
        modified_value=str(modified_value),
        baseline_probability=baseline_probability,
        modified_probability=modified_probability,
        delta_probability=modified_probability - baseline_probability,
        baseline_vote=baseline_vote,
        modified_vote=modified_vote,
    )


def controlled_sensitivity(agent: LCAgent | None = None) -> list[SensitivityPair]:
    """Change one constructed-car attribute at a time with a frozen classifier."""
    if agent is None:
        agent = LCAgent()
    baseline = dict(CONTROLLED_BASELINE)
    pairs = [
        score_pair(agent, baseline, "Buying_Price", level)
        for level in ("low", "high", "vhigh")
    ]
    pairs.extend(
        score_pair(agent, baseline, "Safety", level) for level in ("low", "high")
    )
    return pairs


def format_sensitivity(pairs: list[SensitivityPair]) -> str:
    lines = [
        "Model sensitivity on one constructed vehicle (exactly one attribute changes)",
        "This is not a causal estimate and not observed consumer behaviour.",
        "budget still constrains buying and maintenance; premium constrains buying price only.",
        "The pairs below are the symmetric single-feature comparison.",
        f"{'attribute':<16}{'from':>8}{'to':>8}{'p_base':>10}{'p_new':>10}{'delta':>10}{'flip':>6}",
    ]
    for pair in pairs:
        lines.append(
            f"{pair.attribute:<16}{pair.baseline_value:>8}{pair.modified_value:>8}"
            f"{pair.baseline_probability:>10.4f}{pair.modified_probability:>10.4f}"
            f"{pair.delta_probability:>10.4f}{str(pair.flipped):>6}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class FoldScore:
    fold: int
    validation_size: int
    logistic_balanced_accuracy: float
    logistic_macro_f1: float
    logistic_brier: float
    majority_balanced_accuracy: float
    majority_macro_f1: float
    majority_brier: float


def development_cross_validation(
    n_splits: int = 5,
) -> list[FoldScore]:
    """Compare logistic regression with a majority baseline inside development data.

    The frozen test partition is not used. Preprocessing is fit inside each fold.
    """
    rows, targets = load_data()
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
        reference = DictVectorizer(sparse=False)
        reference.fit([rows[index] for index in train_ids])
        if list(vectorizer.get_feature_names_out()) != list(reference.get_feature_names_out()):
            raise RuntimeError("Fold preprocessing was not fit on fold-training rows only.")
        classifier = initialize_classifier()
        train_labels = [targets[index] for index in train_ids]
        classifier.fit(train_matrix, train_labels)
        valid_rows = [rows[index] for index in valid_ids]
        valid_labels = [targets[index] for index in valid_ids]
        valid_matrix = encode_features(valid_rows, vectorizer, fit=False)
        positive_index = list(classifier.classes_).index(1)
        probabilities = classifier.predict_proba(valid_matrix)[:, positive_index]
        predictions = (probabilities >= PROBABILITY_THRESHOLD).astype(int)
        prior = sum(train_labels) / len(train_labels)
        majority_vote = int(prior >= PROBABILITY_THRESHOLD)
        majority_predictions = [majority_vote] * len(valid_labels)
        majority_probabilities = [prior] * len(valid_labels)
        scores.append(
            FoldScore(
                fold=fold_number,
                validation_size=len(valid_labels),
                logistic_balanced_accuracy=float(
                    balanced_accuracy_score(valid_labels, predictions)
                ),
                logistic_macro_f1=float(
                    f1_score(
                        valid_labels,
                        predictions,
                        average="macro",
                        labels=[0, 1],
                        zero_division=0,
                    )
                ),
                logistic_brier=float(brier_score_loss(valid_labels, _clip(list(probabilities)))),
                majority_balanced_accuracy=float(
                    balanced_accuracy_score(valid_labels, majority_predictions)
                ),
                majority_macro_f1=float(
                    f1_score(
                        valid_labels,
                        majority_predictions,
                        average="macro",
                        labels=[0, 1],
                        zero_division=0,
                    )
                ),
                majority_brier=float(
                    brier_score_loss(valid_labels, _clip(majority_probabilities))
                ),
            )
        )
    return scores


def format_development_cv(scores: list[FoldScore]) -> str:
    logistic_brier = sum(score.logistic_brier for score in scores) / len(scores)
    majority_brier = sum(score.majority_brier for score in scores) / len(scores)
    logistic_balanced = sum(score.logistic_balanced_accuracy for score in scores) / len(scores)
    majority_balanced = sum(score.majority_balanced_accuracy for score in scores) / len(scores)
    lines = [
        "Development-set stratified 5-fold check (training + validation only)",
        "The 0.5 threshold is pre-specified. It was not chosen from the holdout.",
        f"{'fold':>6}{'n':>6}{'log_bal':>10}{'log_f1':>10}{'log_brier':>12}"
        f"{'maj_bal':>10}{'maj_f1':>10}{'maj_brier':>12}",
    ]
    for score in scores:
        lines.append(
            f"{score.fold:>6}{score.validation_size:>6}"
            f"{score.logistic_balanced_accuracy:>10.4f}{score.logistic_macro_f1:>10.4f}"
            f"{score.logistic_brier:>12.4f}{score.majority_balanced_accuracy:>10.4f}"
            f"{score.majority_macro_f1:>10.4f}{score.majority_brier:>12.4f}"
        )
    lines.append(
        "mean logistic balanced accuracy "
        f"{logistic_balanced:.4f}, majority {majority_balanced:.4f}"
    )
    lines.append(
        f"mean logistic Brier {logistic_brier:.4f}, majority {majority_brier:.4f}"
    )
    lines.append("Lower Brier is better. This check does not use the frozen test rows.")
    return "\n".join(lines)


def format_full_evaluation(agent: LCAgent | None = None) -> str:
    if agent is None:
        agent = LCAgent()
    comparison = [
        summarize_scenario(CarPurchaseScenario(scenario_type=name), agent)
        for name in COMPARISON_SCENARIOS
    ]
    return "\n\n".join(
        [
            format_scenario_comparison(comparison),
            format_quality_report(evaluate_held_out_groups(agent)),
            format_development_cv(development_cross_validation()),
        ]
    )


def proxy_filters_are_marked() -> bool:
    return "sporty_proxy" in SCENARIO_FILTERS and "electric_proxy" in SCENARIO_FILTERS
