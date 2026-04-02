"""
LangGraph implementation for Democratic Agents — Multi-Chamber Legislative Simulation.

Orchestrates the full bill lifecycle:
  bill_analysis → [house] → [senate] → [executive] → [scotus] → final_summary

Each chamber node runs run_debate() from orchestration.debate_manager, then
tally_votes() from orchestration.vote_aggregator to determine pass/fail.

Agent profiles are loaded from ProfileStore (data/*.json — generated from ai-gov-simulator).

TODO before this graph runs end-to-end:
  1. Generate data/senate_profiles.json — run: python src/heuristics/generate_profiles.py
  2. Set ANTHROPIC_API_KEY in environment
  3. Install deps: pip install -r requirements.txt
"""
from typing import TypedDict, List, Dict, Annotated, Literal, Optional
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.executive_agent import ExecutiveAgent, make_president_decision
from agents.scotus_agent import SCOTUSAgent
from agents.senator_agent import SenatorAgent
from heuristics.bill_analyzer import analyze_bill
from heuristics.profiles import ProfileStore
from heuristics.vote_calculator import calculate_vote_probability
from orchestration.debate_manager import run_debate, build_debate_summary
from orchestration.vote_aggregator import tally_votes, format_chamber_result
from llm_factory import get_llm, get_llm_for_votes
from langchain_community.tools import DuckDuckGoSearchRun


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ParliamentState(TypedDict):
    """Shared state that flows through all LangGraph nodes."""

    # Bill
    bill_id: str
    bill_title: str
    bill_text: str
    bill_analysis: Dict          # issueWeights, issuePositions, factions — from bill_analyzer

    # Research (Speaker node)
    speaker_research: str
    web_sources: List[str]

    # Simulation config (passed in at start)
    chamber: str                 # "senate" | "house" | "executive" | "scotus" | "all"
    agent_names: Optional[List[str]]  # explicit agent list, or None to use n_agents from profile store
    n_agents: int                # how many agents to load if agent_names is None
    max_rounds: int

    # Per-chamber results (accumulated as bill moves through chambers)
    chamber_results: Dict        # chamber_name → ChamberResult dict

    # Active debate state (reset per chamber)
    current_chamber: str
    debate_round: int
    debate_history: Annotated[List[Dict], add_messages]
    active_agent_names: List[str]

    # Voting
    votes: Dict                  # agent_name → VoteRecord dict
    vote_result: str             # "PASSED" | "REJECTED" | "SIGNED" | "VETOED"
    vote_tally: Dict             # {"YES": count, "NO": count}

    # Session
    session_id: str
    proceedings_summary: str
    timestamp: str


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def bill_analysis_node(state: ParliamentState) -> dict:
    """
    Node 1: Convert bill text into structured BillAnalysis.

    Calls bill_analyzer.analyze_bill() which uses an LLM to extract:
      issueWeights, issuePositions, partySupport, affectedIndustries,
      constitutionalIssues, controversy_level, factions.
    """
    search = DuckDuckGoSearchRun()
    web_context = search.invoke(f"{state['bill_title']} legislation summary")

    print(f"\n[BILL ANALYSIS] Analyzing: {state['bill_title']}")

    bill_analysis = analyze_bill(state["bill_text"], llm=get_llm(max_new_tokens=5000), context=web_context)
    print(f"  Party support: {bill_analysis.get('partySupport')}")
    print(f"  Controversy: {bill_analysis.get('controversy_level', 0.0):.2f}")
    print(f"  Top issues: {[k for k, v in bill_analysis.get('issueWeights', {}).items() if v > 0.4]}")

    return {
        "bill_analysis": bill_analysis,
        "timestamp": datetime.now().isoformat(),
    }


def senate_chamber_node(state: ParliamentState) -> dict:
    """
    Node 2: Run Senate debate and vote.

    Loads SenatorAgent instances from ProfileStore, runs run_debate(),
    then tallies with tally_votes().

    TODO: Wire in:
      1. ProfileStore().get_roster("senate", n=state["n_agents"])
      2. Build SenatorAgent instances with real LLM
      3. Call run_debate() with those agents
      4. Call tally_votes() with chamber="senate"
      5. Return updated chamber_results + debate_history + votes

    TODO: Add SQLite memory retrieval before debate — each senator queries
          their past votes on similar topics from data/parliament.db.
          Use LangGraph's SqliteSaver checkpointer for this.
    """
    print(f"\n[SENATE] Starting Senate debate on: {state['bill_title']}")
    print(f"  Rounds: {state['max_rounds']}")

    store = ProfileStore()
    profiles = store.get_roster("senate", n=state["n_agents"])
    debate_llm = get_llm(max_new_tokens=256, temperature=0.8)
    vote_llm = get_llm_for_votes(max_new_tokens=512, temperature=0.4)
    agents = [SenatorAgent(p, llm=debate_llm, vote_llm=vote_llm) for p in profiles]
    bill_summary = state["bill_analysis"].get("summary", state["bill_title"])
    debate_history, vote_records = run_debate(
        agents, state["bill_title"], bill_summary,
        state["bill_analysis"], n_rounds=state["max_rounds"]
    )
    result = tally_votes(vote_records, chamber="senate")
    result.transcript_summary = build_debate_summary(debate_history)

    print(format_chamber_result(result))
    return {
        "current_chamber": "senate",
        "vote_result": "PASSED" if result.passed else "REJECTED",
        "vote_tally": {"YES": result.yes_count, "NO": result.no_count},
        "votes": {
            r.agent_name: {
                "vote": r.vote,
                "probability": r.vote_probability,
                "reasoning": r.reasoning,
                "chamber": "senate",
            }
            for r in vote_records
        },
        "active_agent_names": [a.name for a in agents],
        "chamber_results": {**state.get("chamber_results", {}), "senate": result.__dict__},
    }


def executive_chamber_node(state: ParliamentState) -> dict:
    """
    Node 3: Cabinet deliberation + Presidential SIGN/VETO decision.

    TODO: Wire in:
      1. ProfileStore().get_roster("executive") for Cabinet agents
      2. Run 1-round Cabinet deliberation via run_debate()
      3. Collect Cabinet advice (SIGN/VETO recommendations)
      4. Call make_president_decision() from executive_agent.py
      5. Return vote_result = "SIGNED" | "VETOED"

    TODO: If VETOED, trigger veto_override_node which checks if
          Senate + House can muster 2/3 majority using override threshold.
    """
    print(f"\n[EXECUTIVE] Cabinet deliberating on: {state['bill_title']}")

    store = ProfileStore()
    all_profiles = store.get_roster("executive")
    president_profile = next((p for p in all_profiles if p.get("role") == "President"), None)
    if president_profile is None:
        raise ValueError("No President profile found in executive roster")
    cabinet_profiles = [p for p in all_profiles if p.get("role") != "President"]

    senate_result = state.get("chamber_results", {}).get("senate", {})
    debate_summary = senate_result.get("transcript_summary", "No prior congressional debate available.")
    bill_summary = state["bill_analysis"].get("summary", state["bill_title"])

    llm = get_llm(max_new_tokens=256, temperature=0.7)
    cabinet_advice = {}
    for profile in cabinet_profiles:
        agent = ExecutiveAgent(profile, llm=llm)
        cabinet_advice[agent.name] = agent.advise(state["bill_title"], bill_summary, debate_summary)

    exec_decision = make_president_decision(
        president_profile,
        state["bill_analysis"],
        cabinet_advice,
        llm=llm,
    )

    decision_label = "SIGNED" if exec_decision.decision == "SIGN" else "VETOED"
    print(f"  Presidential decision: {decision_label} (veto_probability={exec_decision.veto_probability:.2f})")

    return {
        "current_chamber": "executive",
        "vote_result": decision_label,
        "chamber_results": {
            **state.get("chamber_results", {}),
            "executive": {
                "passed": exec_decision.decision == "SIGN",
                "decision": decision_label,
                "reasoning": exec_decision.reasoning,
                "veto_probability": exec_decision.veto_probability,
                "cabinet_advice": exec_decision.cabinet_advice,
            },
        },
    }


def scotus_chamber_node(state: ParliamentState) -> dict:
    """
    Node 4: SCOTUS constitutional review (optional — triggered by challenge).

    TODO: Wire in:
      1. ProfileStore().get_roster("scotus") for all 9 justices
      2. Build SCOTUSAgent instances
      3. Run 2-round conference deliberation
      4. Use constitutional vote calculator (TODO in scotus_agent.py)
      5. Return "UPHELD" | "STRUCK_DOWN" with opinion types (majority/dissent/concurrence)

    TODO: Determine which constitutional issues the bill triggers from bill_analysis
          and pass those to each justice's deliberate() call.
    """
    print(f"\n[SCOTUS] Constitutional review of: {state['bill_title']}")
    from heuristics.vote_calculator import calculate_vote_probability

    store = ProfileStore()
    profiles = store.get_roster("scotus")  # always all 9 justices
    constitutional_issues = state["bill_analysis"].get("constitutionalIssues", {})

    deliberation_llm = get_llm(max_new_tokens=256, temperature=0.7)
    ruling_llm = get_llm_for_votes(max_new_tokens=512, temperature=0.3)
    agents = [SCOTUSAgent(p, llm=deliberation_llm, vote_llm=ruling_llm) for p in profiles]

    # 2-round conference deliberation
    deliberation_history: list[dict] = []
    for round_num in range(1, 3):
        print(f"\n[SCOTUS DELIBERATION ROUND {round_num}/2]")
        for agent in agents:
            statement = agent.deliberate(state["bill_title"], constitutional_issues, deliberation_history, round_num)
            deliberation_history.append({"agent": agent.name, "round": round_num, "statement": statement})
            print(f"  {agent.name}: {statement[:80]}...")

    # Vote — use vote_calculator as uphold_probability proxy
    print("\n[SCOTUS VOTE]")
    rulings = []
    for agent in agents:
        result = calculate_vote_probability(agent.profile, state["bill_analysis"])
        ruling = agent.rule(state["bill_title"], constitutional_issues, deliberation_history, result["vote_probability"])
        rulings.append(ruling)
        print(f"  {agent.name}: {ruling.ruling} (p={ruling.vote_probability:.2f})")

    uphold_count = sum(1 for r in rulings if r.ruling == "UPHOLD")
    strike_count = len(rulings) - uphold_count
    final_ruling = "UPHELD" if uphold_count >= 5 else "STRUCK_DOWN"
    print(f"\n[SCOTUS] {final_ruling} — {uphold_count} uphold / {strike_count} strike")

    return {
        "current_chamber": "scotus",
        "vote_result": final_ruling,
        "vote_tally": {"UPHOLD": uphold_count, "STRIKE_DOWN": strike_count},
        "chamber_results": {
            **state.get("chamber_results", {}),
            "scotus": {
                "passed": final_ruling == "UPHELD",
                "ruling": final_ruling,
                "uphold_count": uphold_count,
                "strike_count": strike_count,
                "vote_breakdown": [
                    {"justice": r.justice_name, "ruling": r.ruling, "basis": r.constitutional_basis}
                    for r in rulings
                ],
            },
        },
    }


def final_summary_node(state: ParliamentState) -> dict:
    """
    Node 5: Generate full proceedings summary across all chambers.

    TODO: Use an LLM (sonnet) to write a professional parliamentary record:
      - Bill title and session ID
      - Summary of each chamber's debate (key arguments, party positions)
      - Vote tallies per chamber
      - Final disposition (passed/signed/upheld/struck down)
      - Notable moments (unanimous votes, surprising crossovers, etc.)

    TODO: Write output to output/transcripts/{session_id}.md and
          output/votes/{session_id}.json for the record.
    """
    print(f"\n[SUMMARY] Generating proceedings summary...")
    chamber_results = state.get("chamber_results", {})

    lines = [
        "=" * 72,
        "DEMOCRATIC AGENTS — SESSION RECORD",
        f"Session ID : {state['session_id']}",
        f"Bill       : {state['bill_title']}",
        f"Date       : {state['timestamp']}",
        "=" * 72,
        "",
    ]
    for chamber, result in chamber_results.items():
        if chamber == "senate":
            status = "PASSED" if result.get("passed") else "FAILED"
            lines.append(f"[SENATE]    {status} — {result.get('yes_count', '?')} YES / {result.get('no_count', '?')} NO")
        elif chamber == "house":
            status = "PASSED" if result.get("passed") else "FAILED"
            lines.append(f"[HOUSE]     {status} — {result.get('yes_count', '?')} YES / {result.get('no_count', '?')} NO")
        elif chamber == "executive":
            lines.append(f"[EXECUTIVE] {result.get('decision', '?')} (veto_probability={result.get('veto_probability', '?')})")
            if result.get("reasoning"):
                lines.append(f"            {result['reasoning']}")
        elif chamber == "scotus":
            lines.append(f"[SCOTUS]    {result.get('ruling', '?')} — {result.get('uphold_count', '?')} uphold / {result.get('strike_count', '?')} strike")

    lines += ["", f"FINAL RESULT: {state.get('vote_result', 'N/A')}", "=" * 72]
    summary = "\n".join(lines)
    print(summary)

    import json, os
    os.makedirs("output/transcripts", exist_ok=True)
    os.makedirs("output/votes", exist_ok=True)
    sid = state["session_id"]
    with open(f"output/transcripts/{sid}.md", "w") as f:
        f.write(summary)
    with open(f"output/votes/{sid}.json", "w") as f:
        json.dump({
            "session_id": sid,
            "bill_title": state["bill_title"],
            "final_result": state.get("vote_result"),
            "chamber_results": chamber_results,
            "votes": state.get("votes", {}),
        }, f, indent=2, default=str)

    return {"proceedings_summary": summary}


def save_to_database_node(state: ParliamentState) -> dict:
    """
    Node 6: Persist session to SQLite via LangGraph checkpointer.

    TODO: Write explicit records to data/parliament.db:
      - sessions table: session_id, bill_title, bill_text, final_result, timestamp
      - agent_decisions: one row per agent vote (agent_name, vote, probability, reasoning, pros, cons)
      - debate_transcript: one row per debate statement
      These records are used by agents in future sessions to retrieve memory
      of how they voted on similar bills.
    """
    import sqlite3, json, os

    print(f"\n[DATABASE] Saving session {state['session_id']}...")
    os.makedirs("data", exist_ok=True)
    db_path = "data/parliament.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                bill_title   TEXT,
                bill_text    TEXT,
                final_result TEXT,
                timestamp    TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT,
                chamber      TEXT,
                agent_name   TEXT,
                vote         TEXT,
                probability  REAL,
                reasoning    TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS debate_transcript (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                chamber    TEXT,
                round_num  INTEGER,
                agent_name TEXT,
                statement  TEXT
            )
        """)

        # sessions row
        conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?)",
            (state["session_id"], state["bill_title"], state["bill_text"],
             state.get("vote_result"), state["timestamp"]),
        )

        # agent_decisions — one row per vote stored in state["votes"]
        for agent_name, v in state.get("votes", {}).items():
            conn.execute(
                "INSERT INTO agent_decisions (session_id, chamber, agent_name, vote, probability, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
                (state["session_id"], v.get("chamber", ""), agent_name,
                 v.get("vote"), v.get("probability"), v.get("reasoning", "")),
            )

        # debate_transcript — stored inside chamber_results transcript_summary is a
        # flat string; we persist it as a single row per chamber for memory retrieval
        for chamber, result in state.get("chamber_results", {}).items():
            transcript = result.get("transcript_summary", "")
            if transcript:
                conn.execute(
                    "INSERT INTO debate_transcript (session_id, chamber, round_num, agent_name, statement) VALUES (?, ?, ?, ?, ?)",
                    (state["session_id"], chamber, 0, "FULL_TRANSCRIPT", transcript),
                )

        conn.commit()

    print(f"  Saved to {db_path}")
    return {}


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def route_after_senate(state: ParliamentState) -> Literal["executive", "summarize"]:
    """Route: if Senate passed, move to executive. If failed, go to summary."""
    senate_result = state.get("chamber_results", {}).get("senate", {})
    if senate_result.get("passed", False):
        return "executive"
    state["vote_result"] = "REJECTED"
    return "summarize"


def route_after_executive(state: ParliamentState) -> Literal["scotus", "summarize"]:
    """Route: if signed and bill raises constitutional issues, go to SCOTUS. Otherwise summarize."""
    exec_result = state.get("chamber_results", {}).get("executive", {})
    if exec_result.get("decision") == "SIGNED":
        const_issues = state.get("bill_analysis", {}).get("constitutionalIssues", {})
        if any(v > 0.2 for v in const_issues.values()):
            return "scotus"
    return "summarize"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_parliament_graph(checkpoint_path: str = "data/parliament.db"):
    """
    Build and compile the LangGraph state machine.

    Current graph: bill_analysis → senate → executive → scotus → summary → save_db

    TODO: Add house_chamber_node and route bill through house first (per Article I).
    TODO: Add veto_override_node between executive and summarize.
    TODO: Make SCOTUS review conditional on constitutional challenge flag in bill_analysis.
    """
    workflow = StateGraph(ParliamentState)

    # Nodes
    workflow.add_node("bill_analysis", bill_analysis_node)
    workflow.add_node("senate", senate_chamber_node)
    workflow.add_node("executive", executive_chamber_node)
    workflow.add_node("scotus", scotus_chamber_node)
    workflow.add_node("summarize", final_summary_node)
    workflow.add_node("save_db", save_to_database_node)

    # Edges
    workflow.set_entry_point("bill_analysis")
    workflow.add_edge("bill_analysis", "senate")
    workflow.add_conditional_edges("senate", route_after_senate, {"executive": "executive", "summarize": "summarize"})
    workflow.add_conditional_edges("executive", route_after_executive, {"scotus": "scotus", "summarize": "summarize"})
    workflow.add_edge("scotus", "summarize")
    workflow.add_edge("summarize", "save_db")
    workflow.add_edge("save_db", END)

    # Checkpointing for session persistence and agent memory
    import sqlite3
    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    app = workflow.compile(checkpointer=memory)

    return app


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_parliamentary_session(
    bill_title: str,
    bill_text: str,
    chamber: str = "senate",
    agent_names: Optional[List[str]] = None,
    n_agents: int = 10,
    max_debate_rounds: int = 3,
    session_id: str = None,
) -> dict:
    """
    Execute a complete multi-chamber session using LangGraph.

    chamber:     "senate" | "house" | "executive" | "scotus" | "all"
    agent_names: explicit list of senator/justice names, or None to load from profiles
    n_agents:    how many agents to auto-select from profiles when agent_names is None
    """
    app = create_parliament_graph()
    sid = session_id or f"session_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"

    initial_state = {
        "bill_id": f"BILL_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}",
        "bill_title": bill_title,
        "bill_text": bill_text,
        "bill_analysis": {},
        "speaker_research": "",
        "web_sources": [],
        "chamber": chamber,
        "agent_names": agent_names,
        "n_agents": n_agents,
        "max_rounds": max_debate_rounds,
        "chamber_results": {},
        "current_chamber": "",
        "debate_round": 1,
        "debate_history": [],
        "active_agent_names": [],
        "votes": {},
        "vote_result": "",
        "vote_tally": {},
        "session_id": sid,
        "proceedings_summary": "",
        "timestamp": datetime.now().isoformat(),
    }

    config = {"configurable": {"thread_id": sid}}
    print("=" * 80)
    print(f"DEMOCRATIC AGENTS — SESSION START")
    print(f"Bill: {bill_title} | Chamber: {chamber} | Session: {sid}")
    print("=" * 80)

    result = app.invoke(initial_state, config)

    print("\n" + "=" * 80)
    print("SESSION COMPLETE")
    print("=" * 80)
    print(result.get("proceedings_summary", ""))

    return result


if __name__ == "__main__":
    with open("bills/climate_action_act_2026.txt") as f:
        bill = f.read()

    result = run_parliamentary_session(
        bill_title="Climate Action Act 2026",
        bill_text=bill,
        chamber="senate",
        n_agents=10,
        max_debate_rounds=3,
    )
