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
from langchain_core.messages import HumanMessage, SystemMessage

# Matches ai-gov-simulator's BILL_SYSTEM_PROMPT (app/api/analyze-bill/route.js)
BILL_ANALYSIS_SYSTEM = """You are a congressional policy analyst. Return ONLY a compact JSON object — no markdown, no backticks, no extra whitespace.

Rules to keep output short:
- issueWeights: include ALL 15 keys, but set irrelevant ones to 0.0
- issuePositions: ONLY include keys where issueWeight > 0 (omit zeros)
- constitutionalIssues: ONLY include if bill raises a constitutional question (omit if empty)
- constitutionalPosition: same — omit if empty
- affectedIndustries: max 3 items, each under 20 chars
- committees: max 2 items
- factions.supporters and factions.opponents: max 8 words each
- summary: max 15 words

Schema (fill in values, omit empty optional fields):
{"name":"<title>","summary":"<15 words max>","issueWeights":{"immigration":0.0,"taxes_spending":0.0,"healthcare":0.0,"gun_rights":0.0,"climate_energy":0.0,"defense_military":0.0,"education":0.0,"tech_regulation":0.0,"criminal_justice":0.0,"trade_tariffs":0.0,"abortion_social":0.0,"government_spending":0.0,"foreign_policy_hawks":0.0,"civil_liberties":0.0,"labor_unions":0.0},"issuePositions":{},"partySupport":"R|D|bipartisan","affectedIndustries":[],"committees":[],"controversy_level":0.0,"startChamber":"hou|sen","factions":{"supporters":"<8 words>","opponents":"<8 words>"}}

Only return JSON."""

def analyze_bill(bill_text: str, llm=None , context: str = "", ) -> dict:
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

    content = f"{bill_text} \n\nBackground research: {context}"

    messages = [
        SystemMessage(content=BILL_ANALYSIS_SYSTEM),
        HumanMessage(content=content),
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
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(clean)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"[bill_analyzer] Bill analysis failed — LLM returned invalid JSON.\n"
            f"This likely means max_new_tokens is too low and the response was truncated.\n"
            f"Raw output ({len(raw)} chars): {raw[:300]}"
        )


def _repair_truncated_json(s: str) -> str:
    """
    Best-effort repair of a truncated JSON string.
    Closes any open string, then appends enough closing braces/brackets
    to make the structure valid.
    """
    s = s.rstrip()
    # If ends mid-string, close the string
    # Count unescaped quotes to determine if we're inside a string
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_string:
            i += 2
            continue
        if c == '"':
            in_string = not in_string
        i += 1
    if in_string:
        s += '"'
    # Remove trailing comma before closing
    s = s.rstrip().rstrip(',')
    # Count unclosed braces and brackets
    depth_brace = s.count('{') - s.count('}')
    depth_bracket = s.count('[') - s.count(']')
    s += ']' * max(depth_bracket, 0)
    s += '}' * max(depth_brace, 0)
    return s


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
