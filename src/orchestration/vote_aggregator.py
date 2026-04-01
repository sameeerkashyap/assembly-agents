"""
Vote aggregator — tallies VoteRecords per chamber using the correct threshold.

Thresholds:
  house:         218/435 (simple majority of full chamber, but we simulate a subset)
  senate:        51/100 for passage; 60/100 for cloture (end debate)
  senate_cloture: 60/100
  executive:     President SIGN/VETO (single decider)
  scotus:        5/9 justices to uphold
  override:      2/3 of both chambers to override a veto
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base_agent import VoteRecord


# Fraction of agents needed to pass (applied to however many agents are in the simulation)
# For a subset simulation, we use proportional majority rather than absolute seat counts.
CHAMBER_THRESHOLDS: dict[str, float] = {
    "house": 0.5,           # simple majority of simulated agents
    "senate": 0.5,          # simple majority (51/100 scaled to subset)
    "senate_cloture": 0.6,  # 60/100 scaled to subset
    "executive": 1.0,       # President alone — handled separately
    "scotus": 5 / 9,        # 5 of 9 justices
    "override": 2 / 3,      # 2/3 of simulated agents
}


@dataclass
class ChamberResult:
    chamber: str
    passed: bool
    yes_count: int
    no_count: int
    total: int
    threshold: float
    required_yes: int         # how many YES votes were needed
    vote_breakdown: list[dict]  # [{agent, vote, probability, party}]
    transcript_summary: str


def tally_votes(
    vote_records: list["VoteRecord"],
    chamber: str,
    chamber_size: int | None = None,
) -> ChamberResult:
    """
    Tally votes and determine if the bill passes this chamber.

    vote_records:  List of VoteRecord from run_debate()
    chamber:       One of CHAMBER_THRESHOLDS keys
    chamber_size:  If provided, scale threshold against full chamber size
                   (e.g., 10 simulated senators out of 100 → scale proportionally).
                   If None, use proportion of simulated agents.

    TODO: Implement chamber_size scaling so a 10-agent Senate subset correctly
          represents proportional majority rather than absolute vote counts.

    TODO: Add filibuster check for Senate — even if 51/100 vote YES, if there aren't
          60 votes for cloture, the bill is blocked. Implement as a pre-check before
          the final passage tally.
    """
    threshold_frac = CHAMBER_THRESHOLDS.get(chamber, 0.5)
    total = len(vote_records)

    yes_count = sum(1 for r in vote_records if r.vote == "YES")
    no_count = total - yes_count

    if chamber_size is not None and chamber_size > total:
        # Scale: required_yes = threshold_frac × chamber_size, then compare to yes_count
        # proportionally: yes_count/total ≥ threshold_frac
        required_yes = int(threshold_frac * chamber_size)
        # Estimate full-chamber yes from our sample
        estimated_yes = int((yes_count / total) * chamber_size)
        passed = estimated_yes >= required_yes
    else:
        required_yes = int(threshold_frac * total) + (1 if threshold_frac * total % 1 == 0 else 0)
        passed = yes_count / total > threshold_frac

    vote_breakdown = [
        {
            "agent": r.agent_name,
            "vote": r.vote,
            "probability": r.vote_probability,
        }
        for r in vote_records
    ]

    return ChamberResult(
        chamber=chamber,
        passed=passed,
        yes_count=yes_count,
        no_count=no_count,
        total=total,
        threshold=threshold_frac,
        required_yes=required_yes,
        vote_breakdown=vote_breakdown,
        transcript_summary="",  # Filled in by parliament_graph after debate summary
    )


def format_chamber_result(result: ChamberResult) -> str:
    """Human-readable result string for logging and output files."""
    status = "PASSED" if result.passed else "FAILED"
    return (
        f"[{result.chamber.upper()}] {status} — "
        f"{result.yes_count} YES / {result.no_count} NO "
        f"(needed {result.required_yes}/{result.total}, threshold {result.threshold:.0%})"
    )
