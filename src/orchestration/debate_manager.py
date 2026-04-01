"""
Debate manager — runs structured N-round debates across a list of agents.

Each round: every agent speaks in order, seeing all prior statements.
After all rounds: each agent casts a vote (direction from heuristics, reasoning from LLM).

Returns the updated ParliamentState with debate_history and votes populated.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent, VoteRecord

from heuristics.vote_calculator import calculate_vote_probability


def run_debate(
    agents: list["BaseAgent"],
    bill_title: str,
    bill_summary: str,
    bill_analysis: dict,
    n_rounds: int = 3,
    on_statement=None,
) -> tuple[list[dict], list["VoteRecord"]]:
    """
    Run a full multi-round debate and collect votes.

    agents:       List of BaseAgent instances for this chamber
    bill_analysis: Output of bill_analyzer.analyze_bill()
    n_rounds:     Number of debate rounds before voting
    on_statement: Optional callback(agent_name, round_num, statement) for streaming output

    Returns:
      debate_history: list of {agent, round, statement} dicts
      vote_records:   list of VoteRecord (one per agent)

    TODO: Add a "persuasion check" between rounds — if an agent's debate statement
          contained the opponent's pressure_point, apply a small modifier to their
          vote_probability in the next round. This models actual debate persuasion.

    TODO: For large chambers (100 senators), run agents in parallel using asyncio
          within each round to avoid O(N×rounds) sequential LLM calls.
          Agents in the same round don't depend on each other until the round is logged.
          Use asyncio.gather with async LLM calls.
    """
    debate_history: list[dict] = []

    # ── Debate Rounds ────────────────────────────────────────────────────────
    for round_num in range(1, n_rounds + 1):
        print(f"\n[DEBATE ROUND {round_num}/{n_rounds}]")
        for agent in agents:
            statement = agent.debate(bill_title, bill_summary, debate_history, round_num)
            entry = {
                "agent": agent.name,
                "party": agent.profile.get("party", "?"),
                "round": round_num,
                "statement": statement,
            }
            debate_history.append(entry)
            print(f"  {agent.name}: {statement[:80]}...")
            if on_statement:
                on_statement(agent.name, round_num, statement)

    # ── Voting ───────────────────────────────────────────────────────────────
    print("\n[VOTING]")
    vote_records: list["VoteRecord"] = []

    for agent in agents:
        # Heuristic engine determines direction
        vote_result = calculate_vote_probability(agent.profile, bill_analysis)

        # LLM writes in-character reasoning for that direction
        record = agent.vote(bill_title, bill_summary, debate_history, vote_result)
        vote_records.append(record)
        print(f"  {agent.name}: {record.vote} (p={record.vote_probability:.2f})")

    return debate_history, vote_records


def build_debate_summary(debate_history: list[dict]) -> str:
    """
    Compress the full debate transcript into a summary string for use in
    executive review or SCOTUS context (where the full transcript is too long).

    TODO: Use an LLM summarizer (haiku) for this rather than raw concatenation.
          The summary should capture: key arguments made, points of contention,
          any concessions or amendments proposed during debate.
    """
    lines = []
    for entry in debate_history:
        lines.append(f"[Round {entry['round']}] {entry['agent']} ({entry['party']}): {entry['statement']}")
    return "\n".join(lines)
