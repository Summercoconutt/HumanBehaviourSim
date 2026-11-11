import random
from collections.abc import Callable
from dataclasses import dataclass

from Agents.LinearClassifierAgent import (
    FEATURE_COLUMNS,
    LCAgent,
    FeatureRow,
    calculate_baseline_vote_probability,
    car_data_split_indices,
    convert_probability_to_binary_vote,
    features_from_context,
    load_data,
)
from Simulator.SimulatorCore import (
    AbstractAgent,
    AbstractScenario,
    Decision,
    ResourceState,
    State,
)


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
}


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
                        f"Car {car_number}: Would you buy this car? "
                        f"(0 = No, 1 = Yes)\n{car_details}"
                    ),
                    decision=[0, 1],
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
    """Acceptability scores for one held-out vehicle group.

    Probabilities come from the shared baseline classifier. They are not
    observed purchase rates.
    """

    scenario_type: str
    sample_count: int
    mean_probability: float
    acceptance_rate: float
    positive_votes: int


def summarize_scenario(
    scenario: CarPurchaseScenario,
    agent: LCAgent,
) -> ScenarioPredictionSummary:
    """Score every selected car once with an already trained agent."""
    if scenario.sample_count == 0:
        raise ValueError("Cannot summarize a scenario with no cars.")

    probabilities: list[float] = []
    votes: list[int] = []
    for car in scenario.cars:
        probability = calculate_baseline_vote_probability(
            agent.classifier,
            agent.vectorizer,
            features_from_context(car),
        )
        probabilities.append(probability)
        votes.append(convert_probability_to_binary_vote(probability))

    sample_count = len(votes)
    return ScenarioPredictionSummary(
        scenario_type=scenario.scenario_type,
        sample_count=sample_count,
        mean_probability=sum(probabilities) / sample_count,
        acceptance_rate=sum(votes) / sample_count,
        positive_votes=sum(votes),
    )


def compare_held_out_scenarios(
    agent: LCAgent | None = None,
) -> list[ScenarioPredictionSummary]:
    """Evaluate the predefined held-out groups with one fixed baseline model."""
    if agent is None:
        agent = LCAgent()
    return [
        summarize_scenario(CarPurchaseScenario(scenario_type=name), agent)
        for name in COMPARISON_SCENARIOS
    ]


def comparison_deltas(
    summaries: list[ScenarioPredictionSummary],
) -> dict[str, tuple[float, float]]:
    """Return (mean probability difference, acceptance-rate difference)."""
    by_name = {summary.scenario_type: summary for summary in summaries}

    def pair(left: str, right: str) -> tuple[float, float]:
        return (
            by_name[left].mean_probability - by_name[right].mean_probability,
            by_name[left].acceptance_rate - by_name[right].acceptance_rate,
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
        "Held-out baseline comparison (one model, trained on the training partition)",
        f"{'scenario':<22}{'n':>6}{'mean_p':>12}{'accept_rate':>14}{'accept_votes':>14}",
    ]
    for summary in summaries:
        lines.append(
            f"{summary.scenario_type:<22}{summary.sample_count:>6}"
            f"{summary.mean_probability:>12.4f}{summary.acceptance_rate:>14.4f}"
            f"{summary.positive_votes:>14}"
        )
    deltas = comparison_deltas(summaries)
    budget_probability, budget_rate = deltas["budget_minus_premium"]
    safety_probability, safety_rate = deltas["safe_family_minus_low_safety_family"]
    lines.append(
        "budget minus premium: "
        f"mean_p {budget_probability:+.4f}, accept_rate {budget_rate:+.4f}"
    )
    lines.append(
        "safe_family minus low_safety_family: "
        f"mean_p {safety_probability:+.4f}, accept_rate {safety_rate:+.4f}"
    )
    lines.append(
        "Scores are predicted acceptability, not observed willingness to buy."
    )
    return "\n".join(lines)
