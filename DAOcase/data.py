import os
from collections import Counter
from dataclasses import dataclass

import pyarrow.parquet as pq


SNAPSHOT_DIR = r"D:\snapshot_votes_441\spaces\space=uniswapgovernance.eth"

# Single-choice Snapshot ballots. Choice is an integer id, and these files
# contain every id from 1 through N, including 1. SimpleAgent1 always returns
# the first option, so the option list is sorted and starts at 1.
PROPOSAL_IDS = (
    # 15,212 votes. Options 1 and 2 both received votes; option 1 holds most voting power.
    "0x5ad7208d9ccad93dbc96abc12bfeffcfe83e96b9782e5135727cdaaf494d70f3",
    # 2,452 votes. Options 1 and 2; option 2 won, and option 1 is still a recorded vote.
    "0x3d8d2f44677c19f49a5bbe4466e4b50068b1a704861801d95384c898b1cbde54",
    # 335 votes. Options 1, 2, and 3 all received votes.
    "0x6175261050aeccdd3d541cb4eb2b58c9fde747bfda509896586df5a62e1295d4",
)


@dataclass(frozen=True)
class ProposalRecord:
    proposal_id: str
    title: str
    body: str
    created: str
    choices: tuple[int, ...]
    votes_by_choice: dict[int, int]
    voting_power_by_choice: dict[int, float]

    @property
    def vote_count(self) -> int:
        return sum(self.votes_by_choice.values())


def _proposal_path(data_dir: str, proposal_id: str) -> str:
    prefix = f"proposal={proposal_id}_"
    matches = [
        name
        for name in os.listdir(data_dir)
        if name.startswith(prefix) and name.endswith(".parquet")
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one parquet file for {proposal_id} in {data_dir}, "
            f"found {len(matches)}."
        )
    return os.path.join(data_dir, matches[0])


def load_proposal(data_dir: str, proposal_id: str) -> ProposalRecord:
    """Load one proposal and reject ballots this simulator cannot score."""
    table = pq.read_table(
        _proposal_path(data_dir, proposal_id),
        columns=[
            "Proposal ID",
            "Proposal Title",
            "Proposal Body",
            "Created Time",
            "Choice",
            "Voting Power",
        ],
    )
    if table.num_rows == 0:
        raise ValueError(f"{proposal_id} has no votes.")

    stored_id = table.column("Proposal ID")[0].as_py()
    if stored_id != proposal_id:
        raise ValueError(
            f"File for {proposal_id} contains proposal id {stored_id}."
        )

    title = (table.column("Proposal Title")[0].as_py() or "").strip()
    body = (table.column("Proposal Body")[0].as_py() or "").strip()
    created = table.column("Created Time")[0].as_py() or ""
    if not title:
        raise ValueError(f"{proposal_id} has an empty title.")
    if not body:
        raise ValueError(f"{proposal_id} has an empty proposal body.")

    choices = table.column("Choice").to_pylist()
    powers = table.column("Voting Power").to_pylist()
    if any(not isinstance(choice, int) for choice in choices):
        raise ValueError(
            f"{proposal_id} is not a single-choice ballot. "
            "Weighted or approval votes are stored as text and cannot be "
            "one option in this simulator."
        )

    unique = tuple(sorted(set(choices)))
    expected = tuple(range(1, unique[-1] + 1))
    if unique != expected or unique[0] != 1:
        raise ValueError(
            f"{proposal_id} choices {unique} are not the contiguous ids 1..N. "
            "The agent selects the first option, which must be choice 1."
        )

    votes_by_choice = dict(Counter(choices))
    power_by_choice: dict[int, float] = {choice: 0.0 for choice in unique}
    for choice, power in zip(choices, powers):
        power_by_choice[choice] += float(power or 0)

    return ProposalRecord(
        proposal_id=proposal_id,
        title=title,
        body=body,
        created=created,
        choices=unique,
        votes_by_choice=votes_by_choice,
        voting_power_by_choice=power_by_choice,
    )
