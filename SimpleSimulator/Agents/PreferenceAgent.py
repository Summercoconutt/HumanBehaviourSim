"""Illustrative preference adjustments on top of the acceptability baseline.

The weights are chosen by the experimenter. They are not estimated from consumers.
"""

import math
import random
from dataclasses import dataclass

from Agents.LinearClassifierAgent import (
    BaselineModel,
    FeatureRow,
    calculate_baseline_vote_probability,
    features_from_context,
    train_baseline_model,
)
from Simulator.SimulatorCore import AbstractAgent, AgentDecision, Decision, State


PRICE_CODE = {"low": 0.0, "med": 1 / 3, "high": 2 / 3, "vhigh": 1.0}
SAFETY_CODE = {"low": 0.0, "med": 0.5, "high": 1.0}
COST_SENSITIVE_WEIGHTS = {"cost": -6.0, "safety": 0.0}
SAFETY_SEEKING_WEIGHTS = {"cost": 0.0, "safety": 6.0}


def preference_utility(
    probability: float,
    car: FeatureRow,
    weights: dict[str, float],
    noise: float = 0.0,
) -> float:
    bounded = min(max(probability, 1e-6), 1 - 1e-6)
    logit = math.log(bounded / (1 - bounded))
    cost = PRICE_CODE[str(car["Buying_Price"])]
    safety = SAFETY_CODE[str(car["Safety"])]
    return (
        logit
        + weights.get("cost", 0.0) * cost
        + weights.get("safety", 0.0) * safety
        + noise
    )


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
        model: BaselineModel | None = None,
        name: str = "preference",
    ) -> None:
        self.name = name
        self.weights = dict(weights or {})
        self.seed = seed
        self.noise_scale = noise_scale
        self.random = random.Random(seed)
        self.model = model if model is not None else train_baseline_model()

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
        probability = calculate_baseline_vote_probability(
            self.model.classifier,
            self.model.vectorizer,
            features,
        )
        noise = self.random.gauss(0.0, self.noise_scale) if self.noise_scale else 0.0
        utility = preference_utility(probability, features, self.weights, noise)
        choice = int(utility >= 0.0)
        if choice not in decision.decision:
            raise ValueError("The scenario choices must include 0 and 1.")
        return AgentDecision(
            agent_id=self.agent_id,
            decision=decision,
            choice=choice,
        )


def repeated_choices(
    profile: PreferenceProfile,
    car: FeatureRow,
    probability: float,
    repeats: int = 30,
) -> list[int]:
    generator = random.Random(profile.seed)
    choices = []
    for _ in range(repeats):
        noise = generator.gauss(0.0, profile.noise_scale) if profile.noise_scale else 0.0
        utility = preference_utility(probability, car, profile.weights, noise)
        choices.append(int(utility >= 0.0))
    return choices


def format_preference_demo(probability: float = 0.5) -> str:
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
        "Same constructed car and the same baseline probability for every profile.",
        f"baseline probability={probability:.4f}; car price=vhigh; safety=high",
    ]
    for profile in profiles:
        choices = repeated_choices(profile, car, probability, repeats=30)
        lines.append(
            f"{profile.name}: first_choice={choices[0]} "
            f"accept_share={sum(choices) / len(choices):.2f} "
            f"noise_scale={profile.noise_scale}"
        )
    return "\n".join(lines)
