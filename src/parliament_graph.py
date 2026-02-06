"""
LangGraph implementation for Parliamentary Multi-Agent Simulation
"""
from typing import TypedDict, List, Dict, Annotated, Literal
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ParliamentState(TypedDict):
    """Shared state that flows through all nodes in the graph"""
    
    # Bill information
    bill_id: str
    bill_title: str
    bill_text: str
    
    # Research data (from Speaker)
    speaker_research: str
    web_sources: List[str]
    
    # Debate tracking
    debate_round: int
    max_rounds: int
    debate_history: Annotated[List[Dict], add_messages]  # Auto-appends messages
    
    # Agent states
    agent_memories: Dict[str, List[Dict]]  # agent_id -> list of past decisions
    agent_positions: Dict[str, str]
    
    # Voting
    votes: Dict[str, Dict]  # agent_id -> {vote, reasoning, pros, cons}
    vote_result: str  # "PASSED" or "REJECTED"
    vote_tally: Dict[str, int]  # {"YES": count, "NO": count}
    
    # Final outputs
    session_id: str
    proceedings_summary: str
    timestamp: str


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def research_bill_node(state: ParliamentState) -> ParliamentState:
    """
    Node 1: Speaker agent researches the bill topic using web scraping
    """
    print(f"\n[RESEARCH] Speaker researching: {state['bill_title']}")
    
    # TODO: Implement actual web scraping and LLM-based research
    # For now, mock the research
    research_summary = f"""
    Research Summary for {state['bill_title']}:
    - This bill addresses key policy concerns
    - Historical context shows mixed results from similar legislation
    - Current data suggests potential economic impacts
    - Expert opinions vary across the political spectrum
    """
    
    sources = [
        "https://example.com/policy-analysis",
        "https://example.com/economic-impact-study"
    ]
    
    return {
        "speaker_research": research_summary,
        "web_sources": sources,
        "timestamp": datetime.now().isoformat()
    }


def retrieve_memories_node(state: ParliamentState) -> ParliamentState:
    """
    Node 2: Each agent queries their memory database for relevant past decisions
    """
    print(f"\n[MEMORY] Retrieving agent memories...")
    
    # TODO: Implement actual database query
    # For now, mock the memory retrieval
    agent_memories = {}
    
    # Mock: Each agent has some past memories
    for agent_id in ["conservative_1", "liberal_1", "centrist_1"]:
        agent_memories[agent_id] = [
            {
                "topic": "Climate Policy",
                "vote": "YES" if "liberal" in agent_id else "NO",
                "reasoning": "Mock past reasoning",
                "session_id": "past_session_001"
            }
        ]
    
    return {"agent_memories": agent_memories}


def debate_node(state: ParliamentState) -> ParliamentState:
    """
    Node 3: One round of debate where all agents present their positions
    This node can be called multiple times (cyclic)
    """
    round_num = state["debate_round"]
    print(f"\n[DEBATE ROUND {round_num}] Agents debating...")
    
    # TODO: Implement actual agent debate using LLM
    # For now, mock debate statements
    statements = []
    
    agents_config = [
        ("conservative_1", "Conservative Representative"),
        ("liberal_1", "Liberal Representative"),
        ("centrist_1", "Centrist Representative")
    ]
    
    for agent_id, role in agents_config:
        statement = f"[{role}] Statement for round {round_num} on {state['bill_title']}"
        
        statements.append({
            "agent_id": agent_id,
            "role": role,
            "round": round_num,
            "statement": statement,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"  - {role}: {statement[:50]}...")
    
    return {
        "debate_history": statements,  # Automatically appended due to add_messages
        "debate_round": round_num + 1
    }


def voting_node(state: ParliamentState) -> ParliamentState:
    """
    Node 4: Each agent casts their vote with full reasoning (pros/cons/data)
    """
    print(f"\n[VOTING] Agents casting votes...")
    
    # TODO: Implement actual voting logic with LLM
    # For now, mock the votes
    votes = {}
    
    mock_votes = [
        ("conservative_1", "NO", "Conservative reasoning against"),
        ("liberal_1", "YES", "Liberal reasoning in favor"),
        ("centrist_1", "YES", "Centrist reasoning in favor")
    ]
    
    for agent_id, vote_choice, reasoning in mock_votes:
        votes[agent_id] = {
            "vote": vote_choice,
            "reasoning": reasoning,
            "pros": ["Pro point 1", "Pro point 2"],
            "cons": ["Con point 1", "Con point 2"],
            "data_sources": state["web_sources"]
        }
        
        print(f"  - {agent_id}: {vote_choice}")
    
    # Tally the votes
    yes_count = sum(1 for v in votes.values() if v["vote"] == "YES")
    no_count = len(votes) - yes_count
    result = "PASSED" if yes_count > no_count else "REJECTED"
    
    print(f"\n[RESULT] Bill {result} ({yes_count} YES, {no_count} NO)")
    
    return {
        "votes": votes,
        "vote_result": result,
        "vote_tally": {"YES": yes_count, "NO": no_count}
    }


def speaker_summarize_node(state: ParliamentState) -> ParliamentState:
    """
    Node 5: Speaker summarizes the entire parliamentary session
    """
    print(f"\n[SUMMARY] Speaker preparing proceedings summary...")
    
    # TODO: Implement actual LLM-based summarization
    summary = f"""
    PARLIAMENTARY PROCEEDINGS SUMMARY
    Session ID: {state['session_id']}
    Bill: {state['bill_title']}
    
    DEBATE:
    - {len(state['debate_history'])} total statements across {state['debate_round'] - 1} rounds
    
    VOTING:
    - Result: {state['vote_result']}
    - Tally: {state['vote_tally']['YES']} YES, {state['vote_tally']['NO']} NO
    
    The bill has been {state['vote_result']}.
    """
    
    return {"proceedings_summary": summary}


def save_to_database_node(state: ParliamentState) -> ParliamentState:
    """
    Node 6: Persist all session data to database
    """
    print(f"\n[DATABASE] Saving session {state['session_id']} to database...")
    
    # TODO: Implement actual database save
    # For now, just log
    print(f"  - Bill: {state['bill_title']}")
    print(f"  - Votes: {len(state['votes'])}")
    print(f"  - Debate statements: {len(state['debate_history'])}")
    
    return {}


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_continue_debate(state: ParliamentState) -> Literal["debate", "vote"]:
    """
    Conditional edge: Determine if we need more debate rounds or move to voting
    """
    if state["debate_round"] > state["max_rounds"]:
        return "vote"
    else:
        return "debate"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_parliament_graph(checkpoint_path: str = "data/parliament.db"):
    """
    Build and compile the LangGraph state machine for parliamentary simulation
    """
    
    # Create the graph
    workflow = StateGraph(ParliamentState)
    
    # Add all nodes
    workflow.add_node("research", research_bill_node)
    workflow.add_node("retrieve_memory", retrieve_memories_node)
    workflow.add_node("debate", debate_node)
    workflow.add_node("vote", voting_node)
    workflow.add_node("summarize", speaker_summarize_node)
    workflow.add_node("save_db", save_to_database_node)
    
    # Define the flow
    workflow.set_entry_point("research")
    workflow.add_edge("research", "retrieve_memory")
    workflow.add_edge("retrieve_memory", "debate")
    
    # Conditional edge: cycle back to debate or move to vote
    workflow.add_conditional_edges(
        "debate",
        should_continue_debate,
        {
            "debate": "debate",  # Loop back for another round
            "vote": "vote"        # Proceed to voting
        }
    )
    
    workflow.add_edge("vote", "summarize")
    workflow.add_edge("summarize", "save_db")
    workflow.add_edge("save_db", END)
    
    # Compile with checkpointing for persistence
    memory = SqliteSaver.from_conn_string(checkpoint_path)
    app = workflow.compile(checkpointer=memory)
    
    return app


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_parliamentary_session(
    bill_title: str,
    bill_text: str,
    max_debate_rounds: int = 3,
    session_id: str = None
):
    """
    Execute a complete parliamentary session using LangGraph
    """
    
    # Create the graph
    app = create_parliament_graph()
    
    # Initialize state
    initial_state = {
        "bill_id": f"BILL_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}",
        "bill_title": bill_title,
        "bill_text": bill_text,
        "speaker_research": "",
        "web_sources": [],
        "debate_round": 1,
        "max_rounds": max_debate_rounds,
        "debate_history": [],
        "agent_memories": {},
        "agent_positions": {},
        "votes": {},
        "vote_result": "",
        "vote_tally": {},
        "session_id": session_id or str(uuid.uuid4()),
        "proceedings_summary": "",
        "timestamp": datetime.now().isoformat()
    }
    
    # Run the graph
    config = {"configurable": {"thread_id": initial_state["session_id"]}}
    
    print("=" * 80)
    print(f"STARTING PARLIAMENTARY SESSION")
    print(f"Bill: {bill_title}")
    print("=" * 80)
    
    result = app.invoke(initial_state, config)
    
    print("\n" + "=" * 80)
    print("SESSION COMPLETE")
    print("=" * 80)
    print(result["proceedings_summary"])
    
    return result


if __name__ == "__main__":
    # Example usage
    bill = """
    Climate Action and Renewable Energy Transition Act 2026
    
    This bill proposes:
    1. 50% reduction in carbon emissions by 2030
    2. $100B investment in renewable energy infrastructure
    3. Carbon tax on corporations exceeding emission limits
    4. Green jobs training programs for displaced workers
    """
    
    result = run_parliamentary_session(
        bill_title="Climate Action Act 2026",
        bill_text=bill,
        max_debate_rounds=3
    )
