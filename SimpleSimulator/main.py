import argparse

from Parameters.SimpleParameters1 import SimpleParameters
from Scenarios.SimpleScenario1 import SimpleScenario1
from Simulator.SimulatorCore import (
    AbstractParameters,
    AbstractScenario,
    SimulatorCore,
)


SCENARIOS: dict[str, type[AbstractScenario]] = {
    "SimpleScenario1": SimpleScenario1,
}
PARAMETERS: dict[str, type[AbstractParameters]] = {
    "SimpleParameters": SimpleParameters,
}


def main(
    scenario_name: str = "SimpleScenario1",
    parameters_name: str = "SimpleParameters",
) -> None:
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
        help="parameters class name (default: SimpleParameters)",
    )
    arguments = parser.parse_args()
    main(arguments.scenario, arguments.parameters)
