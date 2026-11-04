import random
from collections.abc import Callable

from Agents.LinearClassifierAgent import (
    FEATURE_COLUMNS,
    FeatureRow,
    car_data_split_indices,
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
                        f"(0 = No, 1 = Yes)
{car_details}"
                    ),
                    decision=[0, 1],
                )
            )
        return decisions

    def is_complete(self, state: State) -> bool:
        return state.step >= self.sample_count
