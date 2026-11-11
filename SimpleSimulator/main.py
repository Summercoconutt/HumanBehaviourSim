import argparse

from Parameters.LinearClassifierParameters import LinearClassifierParameters
from Parameters.SimpleParameters1 import SimpleParameters
from Scenarios.CarPurchaseScenario import (
    SCENARIO_FILTERS,
    CarPurchaseScenario,
    compare_held_out_scenarios,
    format_scenario_comparison,
)
from Scenarios.SimpleScenario1 import SimpleScenario1
from Simulator.SimulatorCore import (
    AbstractParameters,
    AbstractScenario,
    SimulatorCore,
)


SCENARIOS: dict[str, type[AbstractScenario]] = {
    "SimpleScenario1": SimpleScenario1,
    "CarPurchaseScenario": CarPurchaseScenario,
}
PARAMETERS: dict[str, type[AbstractParameters]] = {
    "SimpleParameters": SimpleParameters,
    "LinearClassifierParameters": LinearClassifierParameters,
}


def main(
    scenario_name: str = "SimpleScenario1",
    parameters_name: str = "SimpleParameters",
    *,
    scenario_type: str = "random",
    sample_count: int | None = None,
    seed: int = 42,
    compare: bool = False,
) -> None:
    if compare:
        if scenario_name != "CarPurchaseScenario":
            raise ValueError("Scenario comparison requires CarPurchaseScenario.")
        if scenario_type != "random" or sample_count is not None or seed != 42:
            raise ValueError(
                "--compare evaluates the predefined held-out scenarios; "
                "omit --scenario-type, --sample-count, and --seed."
            )
        print(format_scenario_comparison(compare_held_out_scenarios()))
        return

    if scenario_name == "CarPurchaseScenario":
        scenario = CarPurchaseScenario(
            scenario_type=scenario_type,
            sample_count=sample_count,
            random_state=seed,
        )
    else:
        if scenario_type != "random" or sample_count is not None or seed != 42:
            raise ValueError("Car selection options require CarPurchaseScenario.")
        scenario = SCENARIOS[scenario_name]()

    parameters = PARAMETERS[parameters_name]()
    simulator = SimulatorCore()
    result = simulator(scenario, parameters)
    simulator.show_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a human behaviour scenario.")
    parser.add_argument(
        "scenario",
        nargs="?",
        choices=SCENARIOS,
        default="SimpleScenario1",
        help="scenario class name (default: SimpleScenario1)",
    )
    parser.add_argument(
        "parameters",
        nargs="?",
        choices=PARAMETERS,
        default="SimpleParameters",
    )
    parser.add_argument(
        "--scenario-type",
        choices=list(SCENARIO_FILTERS),
        default="random",
        help="vehicle filter for CarPurchaseScenario (default: random)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="number of held-out cars (default: 10 for random, all for named filters)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used when sampling vehicles (default: 42)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "compare full, budget, premium, safe_family, and "
            "low_safety_family on one trained baseline model"
        ),
    )
    arguments = parser.parse_args()
    main(
        arguments.scenario,
        arguments.parameters,
        scenario_type=arguments.scenario_type,
        sample_count=arguments.sample_count,
        seed=arguments.seed,
        compare=arguments.compare,
    )
