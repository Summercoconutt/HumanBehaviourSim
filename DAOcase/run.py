import bootstrap
from parameters import DaoParameters
from scenario import UniswapGovernanceScenario
from Simulator.SimulatorCore import SimulatorCore


def main() -> None:
    scenario = UniswapGovernanceScenario()
    parameters = DaoParameters()
    simulator = SimulatorCore()
    result = simulator(scenario, parameters)
    simulator.show_result(result)

    print("Integrated proposals:")
    for proposal in scenario.proposals:
        print(
            f"{proposal.proposal_id} | options={list(proposal.choices)} "
            f"| votes_by_choice={proposal.votes_by_choice} "
            f"| {proposal.title}"
        )


if __name__ == "__main__":
    main()
