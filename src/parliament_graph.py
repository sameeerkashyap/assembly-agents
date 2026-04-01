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

    TODO: Wire in the LLM client here. Use ChatAnthropic with haiku model.
          from langchain_anthropic import ChatAnthropic
          llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1500)

    TODO: Add speaker research via Tavily/web search for additional context.
          The Speaker agent in ai-gov-simulator scrapes web for bill background.
          Use langchain_community.tools.TavilySearchResults.
    """
    from heuristics.bill_analyzer import analyze_bill
    print(f"\n[BILL ANALYSIS] Analyzing: {state['bill_title']}")

    # TODO: Pass real LLM here instead of None
    bill_analysis = analyze_bill(state["bill_text"], llm=None)
    print(f"  Party support: {bill_analysis.get('partySupport')}")
    print(f"  Controversy: {bill_analysis.get('controversy_level'):.2f}")
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

    # TODO: Replace stub with real agent + debate execution
    # from heuristics.profiles import ProfileStore
    # from agents.senator_agent import SenatorAgent
    # from orchestration.debate_manager import run_debate
    # from orchestration.vote_aggregator import tally_votes, format_chamber_result
    #
    # store = ProfileStore()
    # profiles = store.get_roster("senate", n=state["n_agents"])
    # agents = [SenatorAgent(p, llm=_get_llm()) for p in profiles]
    # debate_history, vote_records = run_debate(
    #     agents, state["bill_title"], state["bill_analysis"]["summary"],
    #     state["bill_analysis"], n_rounds=state["max_rounds"]
    # )
    # result = tally_votes(vote_records, chamber="senate")
    # print(format_chamber_result(result))
    # return {"chamber_results": {**state.get("chamber_results", {}), "senate": result.__dict__}, ...}

    return {
        "current_chamber": "senate",
        "vote_result": "PASSED",   # TODO: Replace with real result
        "vote_tally": {"YES": 6, "NO": 4},  # TODO: Replace
        "chamber_results": {"senate": {"passed": True, "yes": 6, "no": 4}},
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

    # TODO: Replace stub with real executive decision logic
    return {
        "current_chamber": "executive",
        "vote_result": "SIGNED",   # TODO: Replace with make_president_decision()
        "chamber_results": {
            **state.get("chamber_results", {}),
            "executive": {"decision": "SIGNED"},
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

    # TODO: Replace stub
    return {
        "current_chamber": "scotus",
        "vote_result": "UPHELD",   # TODO: Replace
        "chamber_results": {
            **state.get("chamber_results", {}),
            "scotus": {"ruling": "UPHELD"},
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

    # Build summary from chamber results
    lines = [
        f"DEMOCRATIC AGENTS — SESSION RECORD",
        f"Session ID: {state['session_id']}",
        f"Bill: {state['bill_title']}",
        f"Date: {state['timestamp']}",
        "",
    ]
    for chamber, result in chamber_results.items():
        lines.append(f"[{chamber.upper()}] {result}")

    lines.append(f"\nFINAL RESULT: {state.get('vote_result', 'N/A')}")
    summary = "\n".join(lines)
    print(summary)
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
    print(f"\n[DATABASE] Saving session {state['session_id']}...")
    # TODO: Implement explicit DB writes (LangGraph checkpointer handles graph state,
    #       but we need custom tables for agent memory retrieval)
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
    """Route: if signed, optionally go to SCOTUS. If vetoed, go to summary.
    TODO: Add override_node between executive and summarize for veto override path.
    TODO: Make SCOTUS review optional (only when constitutional challenge is flagged).
    """
    exec_result = state.get("chamber_results", {}).get("executive", {})
    if exec_result.get("decision") == "SIGNED":
        return "scotus"   # TODO: Make conditional on whether SCOTUS review is warranted
    state["vote_result"] = "VETOED"
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
    memory = SqliteSaver.from_conn_string(checkpoint_path)
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
