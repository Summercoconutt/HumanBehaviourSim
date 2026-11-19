"""Illustrative preference adjustments on top of the three-class model.

The weights are chosen by the experimenter. They are not estimated from consumers.
A negative cost weight pulls a high price toward unacceptable.
A positive safety weight pulls high safety toward good.
"""

import math
import random
from dataclasses import dataclass

from Models.data import FeatureRow, features_from_context
from Models.three_class import ThreeClassModel, score_three_class, train_three_class_model
from Simulator.SimulatorCore import AbstractAgent, AgentDecision, Decision, State


PRICE_CODE = {"low": 0.0, "med": 1 / 3, "high": 2 / 3, "vhigh": 1.0}
SAFETY_CODE = {"low": 0.0, "med": 0.5, "high": 1.0}
COST_SENSITIVE_WEIGHTS = {"cost": -6.0, "safety": 0.0}
SAFETY_SEEKING_WEIGHTS = {"cost": 0.0, "safety": 6.0}


def preference_scores(
    probabilities: tuple[float, float, float] | list[float],
    car: FeatureRow,
    weights: dict[str, float],
    noises: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[float]:
    """Adjust the three class log-probabilities. Higher is more favoured."""
    scores = [math.log(max(float(probability), 1e-6)) for probability in probabilities]
    cost = PRICE_CODE[str(car["Buying_Price"])]
    safety = SAFETY_CODE[str(car["Safety"])]
    cost_weight = weights.get("cost", 0.0)
    safety_weight = weights.get("safety", 0.0)
    scores[1] += cost_weight * cost + 0.5 * safety_weight * safety
    scores[2] += cost_weight * cost + safety_weight * safety
    return [score + noise for score, noise in zip(scores, noises)]


@dataclass
class PreferenceProfile:
    name: str
    weights: dict[str, float]
    seed: int = 1
    noise_scale: float = 0.0


class PreferenceAgent(AbstractAgent):
    agent_id = "Preference"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        seed: int = 1,
        noise_scale: float = 0.0,
        model: ThreeClassModel | None = None,
        name: str = "preference",
    ) -> None:
        self.name = name
        self.weights = dict(weights or {})
        self.seed = seed
        self.noise_scale = noise_scale
        self.random = random.Random(seed)
        self.model = model if model is not None else train_three_class_model()

    def decide(self, decision: Decision, state: State) -> AgentDecision:
        if not decision.decision:
            raise ValueError("The current scenario must provide at least one option.")
        feature_context = state.context
        cars = state.context.get("cars")
        if isinstance(cars, list):
            car = cars[state.step]
            if not isinstance(car, dict):
                raise ValueError("Car data must be a mapping.")
            feature_context = car
        features = features_from_context(feature_context)
        _prediction, probabilities = score_three_class(self.model, features)
        noises = (
            tuple(self.random.gauss(0.0, self.noise_scale) for _ in range(3))
            if self.noise_scale
            else (0.0, 0.0, 0.0)
        )
        scores = preference_scores(probabilities, features, self.weights, noises)
        choice = max(range(3), key=lambda index: scores[index])
        if choice not in decision.decision:
            raise ValueError("The scenario choices must include 0, 1, and 2.")
        return AgentDecision(
            agent_id=self.agent_id,
            decision=decision,
            choice=choice,
        )


def repeated_choices(
    profile: PreferenceProfile,
    car: FeatureRow,
    probabilities: tuple[float, float, float],
    repeats: int = 30,
) -> list[int]:
    generator = random.Random(profile.seed)
    choices = []
    for _ in range(repeats):
        noises = (
            tuple(generator.gauss(0.0, profile.noise_scale) for _ in range(3))
            if profile.noise_scale
            else (0.0, 0.0, 0.0)
        )
        scores = preference_scores(probabilities, car, profile.weights, noises)
        choices.append(max(range(3), key=lambda index: scores[index]))
    return choices


def format_preference_demo() -> str:
    car: FeatureRow = {
        "Buying_Price": "vhigh",
        "Maintenance_Price": "med",
        "No_of_Doors": "4",
        "Person_Capacity": "4",
        "Size_of_Luggage": "med",
        "Safety": "high",
        "Fuel_Type": "petrol",
        "Horse_power": 200.0,
    }
    model = train_three_class_model()
    _prediction, probabilities = score_three_class(model, features_from_context(car))
    probability_tuple = tuple(float(value) for value in probabilities)
    profiles = [
        PreferenceProfile("cost_sensitive", dict(COST_SENSITIVE_WEIGHTS), seed=1),
        PreferenceProfile("safety_seeking", dict(SAFETY_SEEKING_WEIGHTS), seed=1),
        PreferenceProfile(
            "no_preference_noise",
            {"cost": 0.0, "safety": 0.0},
            seed=1,
            noise_scale=3.0,
        ),
    ]
    lines = [
        "Illustrative synthetic preferences, not estimated from consumers.",
        "Same constructed car and the same three-class probabilities for every profile.",
        (
            "probabilities "
            f"unacceptable={probability_tuple[0]:.4f}, "
            f"acceptable={probability_tuple[1]:.4f}, "
            f"good={probability_tuple[2]:.4f}; car price=vhigh; safety=high"
        ),
    ]
    for profile in profiles:
        choices = repeated_choices(profile, car, probability_tuple, repeats=30)
        lines.append(
            f"{profile.name}: first_choice={choices[0]} "
            f"classes={sorted(set(choices))} "
            f"noise_scale={profile.noise_scale}"
        )
    return "\n".join(lines)
