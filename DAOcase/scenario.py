import bootstrap
from Agents.SimpleAgent1 import SimpleAgent1
from Simulator.SimulatorCore import (
    AbstractAgent,
    AbstractScenario,
    Decision,
    ResourceState,
    State,
)
from data import PROPOSAL_IDS, SNAPSHOT_DIR, load_proposal


class UniswapGovernanceScenario(AbstractScenario):
    """Three Uniswap Snapshot proposals, one decision per proposal."""

    def __init__(
        self,
        data_dir: str = SNAPSHOT_DIR,
        proposal_ids: tuple[str, ...] = PROPOSAL_IDS,
    ) -> None:
        if len(proposal_ids) != 3:
            raise ValueError("This scenario uses exactly three proposals.")
        self.proposals = [
            load_proposal(data_dir, proposal_id) for proposal_id in proposal_ids
        ]

    @property
    def initial_resources(self) -> dict[type[AbstractAgent], ResourceState]:
        return {SimpleAgent1: {}}

    @property
    def context(self) -> ResourceState:
        return {
            "space": "uniswapgovernance.eth",
            "rounds": len(self.proposals),
            "proposals": [
                {
                    "proposal_id": proposal.proposal_id,
                    "title": proposal.title,
                    "created": proposal.created,
                    "choices": list(proposal.choices),
                    "vote_count": proposal.vote_count,
                    "votes_by_choice": dict(proposal.votes_by_choice),
                    "voting_power_by_choice": dict(proposal.voting_power_by_choice),
                }
                for proposal in self.proposals
            ],
        }

    @property
    def decisions(self) -> list[Decision]:
        # Choice 1 is first because that is the option SimpleAgent1 returns.
        return [
            Decision(text_body=proposal.title, decision=list(proposal.choices))
            for proposal in self.proposals
        ]

    def is_complete(self, state: State) -> bool:
        return state.step >= len(self.proposals)
