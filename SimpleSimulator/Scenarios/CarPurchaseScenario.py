import random

from sklearn.model_selection import train_test_split

from Agents.LinearClassifierAgent import FEATURE_COLUMNS, FeatureRow, load_data
from Simulator.SimulatorCore import (
    AbstractAgent,
    AbstractScenario,
    Decision,
    ResourceState,
    State,
)


class CarPurchaseScenario(AbstractScenario):
    sample_count = 10

    def __init__(self) -> None:
        features, targets = load_data()
        _, test_features, _, _ = train_test_split(
            features,
            targets,
            test_size=0.15,
            random_state=42,
            stratify=targets,
        )
        if len(test_features) < self.sample_count:
            raise ValueError(
                f"Car dataset test split must have at least {self.sample_count} rows."
            )
        selected_indices = random.Random(42).sample(
            range(len(test_features)),
            self.sample_count,
        )
        self.cars: list[FeatureRow] = [
            test_features[index] for index in selected_indices
        ]

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
