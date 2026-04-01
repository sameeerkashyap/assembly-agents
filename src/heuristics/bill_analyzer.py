"""
Bill analyzer — converts raw bill text into structured BillAnalysis.

Uses the same schema as ai-gov-simulator's analyze-bill API endpoint
(app/api/analyze-bill/route.js) so output feeds directly into vote_calculator.

Output schema:
{
  "name": str,
  "summary": str,
  "issueWeights": dict[str, float],    # 0–1, how much each issue is touched
  "issuePositions": dict[str, float],  # 0–1, what stance the bill takes
  "partySupport": "R" | "D" | "bipartisan",
  "affectedIndustries": list[str],
  "constitutionalIssues": dict[str, float],
  "constitutionalPosition": dict[str, float],
  "committees": list[str],
  "controversy_level": float,
  "startChamber": "hou" | "sen",
  "factions": {"supporters": str, "opponents": str}
}
"""

from __future__ import annotations
import json
from typing import Optional

# Matches ai-gov-simulator's BILL_SYSTEM_PROMPT (app/api/analyze-bill/route.js)
BILL_ANALYSIS_SYSTEM = """You are a congressional policy analyst. Given a bill description, return ONLY a JSON object (no markdown, no backticks) with this exact schema:

{
  "name": "Short official-sounding bill title",
  "summary": "One sentence description",
  "issueWeights": {
    "immigration": 0.0, "taxes_spending": 0.0, "healthcare": 0.0,
    "gun_rights": 0.0, "climate_energy": 0.0, "defense_military": 0.0,
    "education": 0.0, "tech_regulation": 0.0, "criminal_justice": 0.0,
    "trade_tariffs": 0.0, "abortion_social": 0.0, "government_spending": 0.0,
    "foreign_policy_hawks": 0.0, "civil_liberties": 0.0, "labor_unions": 0.0
  },
  "issuePositions": {},
  "partySupport": "R or D or bipartisan",
  "affectedIndustries": ["industry1", "industry2"],
  "constitutionalIssues": {},
  "constitutionalPosition": {},
  "committees": ["Committee1"],
  "controversy_level": 0.0,
  "startChamber": "hou or sen",
  "factions": {
    "supporters": "Who supports this and why (one sentence)",
    "opponents": "Who opposes this and why (one sentence)"
  }
}

issueWeights: how much each issue matters to this bill (0.0=irrelevant, 1.0=core issue).
issuePositions: what POSITION this bill takes on each relevant issue (0.0=most liberal, 1.0=most conservative). Only include issues where weight > 0.
controversy_level: 0.0=routine, 1.0=maximally divisive.
Only return JSON, nothing else."""


def analyze_bill(bill_text: str, llm=None) -> dict:
    """
    Convert raw bill text into a structured BillAnalysis dict.

    llm: a LangChain BaseChatModel (e.g. ChatAnthropic with haiku model).
         If None, returns a stub for testing.

    TODO: Wire in the actual LLM once agents/base_agent.py sets up the LLM client.
          Use claude-haiku-4-5 for speed (bill analysis is a one-time call per session).

    TODO: Add caching — same bill text should not re-call the API.
          Cache key = sha256(bill_text), store in data/bill_cache.json.
    """
    if llm is None:
        # Stub for testing without an LLM
        return _stub_analysis(bill_text)

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=BILL_ANALYSIS_SYSTEM),
        HumanMessage(content=bill_text),
    ]

    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    # Strip markdown fences if present (model sometimes adds them despite instructions)
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        # TODO: Add retry logic here — re-call LLM with stricter prompt on parse failure
        raise ValueError(f"Bill analysis returned invalid JSON: {e}\nRaw: {raw[:200]}")


def _stub_analysis(bill_text: str) -> dict:
    """
    Returns a neutral stub BillAnalysis for unit testing without an LLM.
    All issue weights at 0.3, positions at 0.5 (centrist / ambiguous).

    TODO: Remove once real LLM integration is in place.
    """
    from .profiles import ISSUE_KEYS
    return {
        "name": "Test Bill",
        "summary": "A test bill for simulation development.",
        "issueWeights": {k: 0.3 for k in ISSUE_KEYS},
        "issuePositions": {k: 0.5 for k in ISSUE_KEYS},
        "partySupport": "bipartisan",
        "affectedIndustries": [],
        "constitutionalIssues": {},
        "constitutionalPosition": {},
        "committees": ["Finance"],
        "controversy_level": 0.5,
        "startChamber": "sen",
        "factions": {
            "supporters": "Moderate members of both parties.",
            "opponents": "Ideological extremes on both sides.",
        },
    }
