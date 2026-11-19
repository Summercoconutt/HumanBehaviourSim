import random
from collections.abc import Callable
from dataclasses import dataclass

from Models.data import (
    FEATURE_COLUMNS,
    FeatureRow,
    car_data_split_indices,
    features_from_context,
)
from Models.three_class import LCAgent, load_data, score_three_class
from Simulator.SimulatorCore import (
    AbstractAgent,
    AbstractScenario,
    Decision,
    ResourceState,
    State,
)


# budget and premium are descriptive groups, not a symmetric price experiment.
# budget constrains both price columns; premium constrains buying price only.
# sporty_proxy and electric_proxy use Fuel_Type and Horse_power. Those two
# columns are not in the original UCI Car Evaluation data, and this repository
# does not record how they were generated. They are explicitly proxies.
SCENARIO_FILTERS: dict[str, Callable[[FeatureRow], bool]] = {
    "random": lambda car: True,
    "full": lambda car: True,
    "budget": lambda car: (
        car["Buying_Price"] in ("low", "med")
        and car["Maintenance_Price"] in ("low", "med")
    ),
    "premium": lambda car: car["Buying_Price"] in ("high", "vhigh"),
    "safe_family": lambda car: (
        car["Person_Capacity"] in ("4", "more")
        and car["Safety"] == "high"
    ),
    "low_safety_family": lambda car: (
        car["Person_Capacity"] in ("4", "more")
        and car["Safety"] == "low"
    ),
    "sporty_proxy": lambda car: (
        float(car["Horse_power"]) >= 250 and car["No_of_Doors"] in ("2", "3")
    ),
    "electric_proxy": lambda car: car["Fuel_Type"] == "electric",
}

ACCEPTABILITY_QUESTION = "Is this vehicle configuration acceptable?"


class CarPurchaseScenario(AbstractScenario):
    """Use only the shared held-out partition for reproducible vehicle tests.

    With no arguments the original ten-random-cars behaviour is preserved.
    Named scenarios default to all matching held-out cars; ``sample_count``
    optionally draws a reproducible subset of a specified size.
    """

    def __init__(
        self,
        scenario_type: str = "random",
        sample_count: int | None = None,
        random_state: int = 42,
    ) -> None:
        if scenario_type not in SCENARIO_FILTERS:
            raise ValueError(
                f"Unknown scenario_type {scenario_type!r}. "
                f"Choose from: {', '.join(SCENARIO_FILTERS)}"
            )
        if sample_count is None and scenario_type == "random":
            sample_count = 10  # Preserve the existing default behaviour.
        if sample_count is not None and sample_count <= 0:
            raise ValueError("sample_count must be a positive integer or None.")

        features, targets = load_data()
        _train_idx, _val_idx, test_idx = car_data_split_indices(targets)
        predicate = SCENARIO_FILTERS[scenario_type]
        candidates = [int(i) for i in test_idx if predicate(features[int(i)])]

        if not candidates:
            raise ValueError(f"No held-out cars match scenario {scenario_type!r}.")
        if sample_count is not None:
            if sample_count > len(candidates):
                raise ValueError(
                    f"Scenario {scenario_type!r} contains only {len(candidates)} "
                    f"held-out cars; requested {sample_count}."
                )
            candidates = random.Random(random_state).sample(candidates, sample_count)

        self.scenario_type = scenario_type
        self.random_state = random_state
        self.selected_indices: list[int] = candidates
        self.cars: list[FeatureRow] = [features[i] for i in candidates]
        self.sample_count = len(self.cars)

    @property
    def initial_resources(self) -> dict[type[AbstractAgent], ResourceState]:
        return {}

    @property
    def context(self) -> ResourceState:
        return {"cars": self.cars}

    @property
    def decisions(self) -> list[Decision]:
        decisions = []
        for car_number, car in enumerate(self.cars, start=1):
            car_details = ", ".join(
                f"{column.replace('_', ' ')}: {car[column]}"
                for column in FEATURE_COLUMNS
            )
            decisions.append(
                Decision(
                    text_body=(
                        f"Car {car_number}: {ACCEPTABILITY_QUESTION} "
                        f"(0 = unacceptable, 1 = acceptable, 2 = good)\n{car_details}"
                    ),
                    decision=[0, 1, 2],
                )
            )
        return decisions

    def is_complete(self, state: State) -> bool:
        return state.step >= self.sample_count


# Held-out groups used for the baseline acceptability comparison.
# Random remains a smoke sample and is intentionally excluded.
COMPARISON_SCENARIOS = (
    "full",
    "budget",
    "premium",
    "safe_family",
    "low_safety_family",
)


@dataclass(frozen=True)
class ScenarioPredictionSummary:
    """Three-class scores for one held-out vehicle group.

    Probabilities come from the shared three-class classifier. They are not
    observed purchase rates. Counts are ordered unacceptable, acceptable, good.
    """

    scenario_type: str
    sample_count: int
    class_counts: tuple[int, int, int]
    mean_probabilities: tuple[float, float, float]


def summarize_scenario(
    scenario: CarPurchaseScenario,
    agent: LCAgent,
) -> ScenarioPredictionSummary:
    """Score every selected car once with an already trained three-class agent."""
    if scenario.sample_count == 0:
        raise ValueError("Cannot summarize a scenario with no cars.")

    counts = [0, 0, 0]
    probability_sums = [0.0, 0.0, 0.0]
    for car in scenario.cars:
        prediction, probabilities = score_three_class(
            agent.model,
            features_from_context(car),
        )
        counts[prediction] += 1
        for index, probability in enumerate(probabilities):
            probability_sums[index] += float(probability)

    sample_count = scenario.sample_count
    return ScenarioPredictionSummary(
        scenario_type=scenario.scenario_type,
        sample_count=sample_count,
        class_counts=(counts[0], counts[1], counts[2]),
        mean_probabilities=(
            probability_sums[0] / sample_count,
            probability_sums[1] / sample_count,
            probability_sums[2] / sample_count,
        ),
    )


def compare_held_out_scenarios(
    agent: LCAgent | None = None,
) -> list[ScenarioPredictionSummary]:
    """Evaluate the predefined held-out groups with one three-class model."""
    if agent is None:
        agent = LCAgent()
    return [
        summarize_scenario(CarPurchaseScenario(scenario_type=name), agent)
        for name in COMPARISON_SCENARIOS
    ]


def _share(summary: ScenarioPredictionSummary, class_index: int) -> float:
    return summary.class_counts[class_index] / summary.sample_count


def comparison_deltas(
    summaries: list[ScenarioPredictionSummary],
) -> dict[str, tuple[float, float]]:
    """Return (good-class share difference, mean P(good) difference)."""
    by_name = {summary.scenario_type: summary for summary in summaries}

    def pair(left: str, right: str) -> tuple[float, float]:
        return (
            _share(by_name[left], 2) - _share(by_name[right], 2),
            by_name[left].mean_probabilities[2] - by_name[right].mean_probabilities[2],
        )

    return {
        "budget_minus_premium": pair("budget", "premium"),
        "safe_family_minus_low_safety_family": pair(
            "safe_family",
            "low_safety_family",
        ),
    }


def format_scenario_comparison(
    summaries: list[ScenarioPredictionSummary],
) -> str:
    lines = [
        "Held-out three-class comparison (one model, trained on the training partition)",
        (
            f"{'scenario':<22}{'n':>6}{'unacc':>8}{'acc':>8}{'good':>8}"
            f"{'p_unacc':>10}{'p_acc':>10}{'p_good':>10}"
        ),
    ]
    for summary in summaries:
        lines.append(
            f"{summary.scenario_type:<22}{summary.sample_count:>6}"
            f"{summary.class_counts[0]:>8}{summary.class_counts[1]:>8}"
            f"{summary.class_counts[2]:>8}{summary.mean_probabilities[0]:>10.4f}"
            f"{summary.mean_probabilities[1]:>10.4f}{summary.mean_probabilities[2]:>10.4f}"
        )
    deltas = comparison_deltas(summaries)
    budget_share, budget_probability = deltas["budget_minus_premium"]
    safety_share, safety_probability = deltas["safe_family_minus_low_safety_family"]
    lines.append(
        "budget minus premium, good class: "
        f"share {budget_share:+.4f}, mean_p {budget_probability:+.4f}"
    )
    lines.append(
        "safe_family minus low_safety_family, good class: "
        f"share {safety_share:+.4f}, mean_p {safety_probability:+.4f}"
    )
    lines.append(
        "Scores are predicted acceptability classes, not observed purchases."
    )
    return "\n".join(lines)
