"""
Heuristic vote probability engine.

Given an agent profile (from ai-gov-simulator schema) and a bill analysis,
computes a vote_probability in [0, 1] where > 0.5 means the agent leans YES.

This keeps vote direction deterministic and grounded in real political data.
The LLM is only called afterward to write in-character reasoning for the
predetermined direction — it never gets to override the heuristic outcome.
"""

from __future__ import annotations
from typing import TypedDict


class VoteProbabilityResult(TypedDict):
    vote_probability: float      # 0.0–1.0, >0.5 = leans YES
    vote_direction: str          # "YES" | "NO"
    issue_alignment: float       # raw weighted agreement before modifiers
    modifiers: dict              # breakdown of each modifier applied


def calculate_vote_probability(
    profile: dict,
    bill_analysis: dict,
) -> VoteProbabilityResult:
    """
    Compute vote probability for an agent on a bill.

    profile:       Senate/executive profile dict from ProfileStore
    bill_analysis: Output of bill_analyzer.analyze_bill() — must contain
                   issueWeights (dict[str, float]) and issuePositions (dict[str, float])

    Returns VoteProbabilityResult with probability and breakdown.

    TODO: Tune modifier weights after testing against historical votes.
          Goal: ~85% accuracy vs. real roll-call votes on landmark legislation.
    """
    issue_weights: dict[str, float] = bill_analysis.get("issueWeights", {})
    issue_positions: dict[str, float] = bill_analysis.get("issuePositions", {})
    agent_issues: dict[str, float] = profile.get("issues", {})
    behavior: dict[str, float] = profile.get("behavior", {})
    electoral: dict = profile.get("electoral", {})
    bill_party_support: str = bill_analysis.get("partySupport", "bipartisan")  # "R" | "D" | "bipartisan"
    agent_party: str = profile.get("party", "")

    # ── 1. Base Issue Alignment ───────────────────────────────────────────────
    total_weight = sum(issue_weights.values())
    if total_weight == 0:
        issue_alignment = 0.5  # No issue data → assume neutral
    else:
        weighted_sum = 0.0
        for issue, weight in issue_weights.items():
            if weight <= 0:
                continue
            agent_pos = agent_issues.get(issue, 0.5)
            bill_pos = issue_positions.get(issue, 0.5)
            # Agreement: 1.0 = perfectly aligned, 0.0 = polar opposite
            agreement = 1.0 - abs(agent_pos - bill_pos)
            weighted_sum += agreement * weight
        issue_alignment = weighted_sum / total_weight

    # ── 2. Party Loyalty Modifier ─────────────────────────────────────────────
    # TODO: Expand party alignment check — currently binary (same party or not).
    #       Consider "bipartisan" bills as partial credit.
    party_loyalty = behavior.get("party_loyalty", 0.5)
    party_modifier = 0.0
    if bill_party_support != "bipartisan":
        if bill_party_support == agent_party:
            party_modifier = party_loyalty * 0.15   # Pull toward YES
        else:
            party_modifier = -party_loyalty * 0.15  # Pull toward NO

    # ── 3. Lobby Susceptibility Modifier ─────────────────────────────────────
    # TODO: Implement full industry matching.
    #       Compare bill_analysis["affectedIndustries"] against profile["lobbying"]["top_industries"].
    #       If overlap: susceptibility pulls toward bill position alignment.
    lobby_susceptibility = behavior.get("lobby_susceptibility", 0.5)
    affected_industries: list[str] = bill_analysis.get("affectedIndustries", [])
    top_industries: list[str] = profile.get("lobbying", {}).get("top_industries", [])
    lobby_modifier = _calculate_lobby_modifier(
        affected_industries, top_industries, issue_alignment, lobby_susceptibility
    )

    # ── 4. Electoral Risk Modifier ────────────────────────────────────────────
    # Agents in tight races or primary-vulnerable become more risk-averse.
    # They vote closer to their base (stronger party alignment effect).
    seat_safety = electoral.get("seat_safety", "safe")
    primary_vulnerable = electoral.get("primary_vulnerable", False)
    electoral_modifier = 0.0
    if seat_safety == "toss-up" or primary_vulnerable:
        # Amplify party loyalty (they dare not cross their base)
        electoral_modifier = party_modifier * 0.3

    # ── 5. Media Sensitivity Modifier ────────────────────────────────────────
    # High-controversy bills are pulled toward popular opinion.
    # TODO: Integrate a "public_opinion" field on the bill (would require external data).
    #       For now, media_sensitivity × controversy_level nudges toward 0.5 (safe center).
    media_sensitivity = behavior.get("media_sensitivity", 0.5)
    controversy = bill_analysis.get("controversy_level", 0.5)
    # High controversy → high-media-sensitivity agents regress toward 0.5
    media_modifier = media_sensitivity * controversy * (0.5 - issue_alignment) * 0.1

    # ── 6. Combine ───────────────────────────────────────────────────────────
    raw = issue_alignment + party_modifier + lobby_modifier + electoral_modifier + media_modifier
    vote_probability = max(0.0, min(1.0, raw))

    return VoteProbabilityResult(
        vote_probability=round(vote_probability, 4),
        vote_direction="YES" if vote_probability > 0.5 else "NO",
        issue_alignment=round(issue_alignment, 4),
        modifiers={
            "party": round(party_modifier, 4),
            "lobby": round(lobby_modifier, 4),
            "electoral": round(electoral_modifier, 4),
            "media": round(media_modifier, 4),
        },
    )


def _calculate_lobby_modifier(
    affected_industries: list[str],
    top_industries: list[str],
    issue_alignment: float,
    susceptibility: float,
) -> float:
    """
    If the bill affects industries that are top donors to this agent,
    and the agent is susceptible to lobbying, pull their vote toward
    whatever direction that industry would prefer.

    TODO: Determine industry preference direction (pro/anti bill) from
          bill_analysis["factions"]["supporters"] / ["opponents"] text.
          For now, if overlap exists and alignment > 0.5, amplify YES.
          If overlap exists and alignment < 0.5, amplify NO.
    """
    if not affected_industries or not top_industries:
        return 0.0

    # Normalize for substring matching (top_industries may have "$amount" suffix)
    top_lower = [t.lower() for t in top_industries]
    overlap_count = sum(
        1 for ind in affected_industries
        if any(ind.lower() in t for t in top_lower)
    )

    if overlap_count == 0:
        return 0.0

    overlap_ratio = min(overlap_count / max(len(affected_industries), 1), 1.0)
    direction = 1.0 if issue_alignment >= 0.5 else -1.0
    return direction * susceptibility * overlap_ratio * 0.10
