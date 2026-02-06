# LangGraph Architecture for Parliament Simulation

## Overview

LangGraph orchestrates the multi-agent parliamentary simulation as a **state graph** where each node represents a phase of the legislative process.

## Graph Structure

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Load Bill &    │  Node: Speaker researches bill
│  Research       │  Output: Bill + research data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieve Memory │  Node: Each agent queries past decisions
│                 │  Output: Relevant historical context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Debate Round 1  │  Node: Agents present initial positions
│                 │  Output: Statements from all agents
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Debate Round 2  │  Node: Agents respond to each other
│                 │  Output: Rebuttals and counter-arguments
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Debate Round 3  │  Node: Final arguments
│                 │  Output: Final statements
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vote & Tally   │  Node: Each agent votes with reasoning
│                 │  Output: Vote results + justifications
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Speaker Summary │  Node: Speaker summarizes entire session
│                 │  Output: Proceedings summary + final result
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save to DB     │  Node: Persist all data
│                 │  Output: Session ID
└────────┬────────┘
         │
         ▼
┌─────────────┐
│     END     │
└─────────────┘
```

## State Definition

The shared state flows through all nodes:

```python
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import add_messages

class ParliamentState(TypedDict):
    # Bill information
    bill_id: str
    bill_title: str
    bill_text: str
    
    # Research data
    speaker_research: str
    web_sources: List[str]
    
    # Debate tracking
    debate_round: int
    max_rounds: int
    debate_history: Annotated[List[Dict], add_messages]  # Auto-appends
    
    # Agent states
    agent_memories: Dict[str, List[Dict]]  # agent_id -> past decisions
    agent_positions: Dict[str, str]
    
    # Voting
    votes: Dict[str, Dict]  # agent_id -> {vote, reasoning, pros, cons}
    vote_result: str  # "PASSED" or "REJECTED"
    
    # Final outputs
    session_id: str
    proceedings_summary: str
    timestamp: str
```

## Key Nodes

### 1. Research Node (Speaker)
```python
def research_bill_node(state: ParliamentState) -> ParliamentState:
    """Speaker agent researches the bill topic"""
    speaker_agent = get_speaker_agent()
    
    research_result = speaker_agent.research(
        bill_title=state["bill_title"],
        bill_text=state["bill_text"]
    )
    
    return {
        "speaker_research": research_result["summary"],
        "web_sources": research_result["sources"]
    }
```

### 2. Memory Retrieval Node
```python
def retrieve_memories_node(state: ParliamentState) -> ParliamentState:
    """Each agent retrieves relevant past decisions"""
    agent_memories = {}
    
    for agent in get_all_agents():
        memories = agent.query_memory(
            topic=state["bill_title"],
            limit=5
        )
        agent_memories[agent.id] = memories
    
    return {"agent_memories": agent_memories}
```

### 3. Debate Node (Cyclic)
```python
def debate_node(state: ParliamentState) -> ParliamentState:
    """One round of debate - all agents contribute"""
    round_num = state["debate_round"]
    statements = []
    
    for agent in get_all_agents():
        statement = agent.debate(
            bill=state["bill_text"],
            research=state["speaker_research"],
            memories=state["agent_memories"][agent.id],
            previous_statements=state["debate_history"],
            round_number=round_num
        )
        
        statements.append({
            "agent_id": agent.id,
            "role": agent.role,
            "round": round_num,
            "statement": statement
        })
    
    return {
        "debate_history": statements,  # Auto-appends due to Annotated
        "debate_round": round_num + 1
    }
```

### 4. Voting Node
```python
def voting_node(state: ParliamentState) -> ParliamentState:
    """Each agent casts their vote with full reasoning"""
    votes = {}
    
    for agent in get_all_agents():
        vote_decision = agent.vote(
            bill=state["bill_text"],
            debate_history=state["debate_history"],
            memories=state["agent_memories"][agent.id]
        )
        
        votes[agent.id] = {
            "vote": vote_decision["vote"],  # "YES" or "NO"
            "reasoning": vote_decision["reasoning"],
            "pros": vote_decision["pros"],
            "cons": vote_decision["cons"],
            "data_sources": vote_decision["sources"]
        }
    
    # Tally votes
    yes_count = sum(1 for v in votes.values() if v["vote"] == "YES")
    no_count = len(votes) - yes_count
    result = "PASSED" if yes_count > no_count else "REJECTED"
    
    return {
        "votes": votes,
        "vote_result": result
    }
```

## Conditional Edges

### Continue Debate or Move to Voting?
```python
def should_continue_debate(state: ParliamentState) -> str:
    """Decide if we need more debate rounds"""
    if state["debate_round"] >= state["max_rounds"]:
        return "vote"
    else:
        return "debate"
```

## Graph Construction

```python
from langgraph.graph import StateGraph, END

# Create the graph
workflow = StateGraph(ParliamentState)

# Add nodes
workflow.add_node("research", research_bill_node)
workflow.add_node("retrieve_memory", retrieve_memories_node)
workflow.add_node("debate", debate_node)
workflow.add_node("vote", voting_node)
workflow.add_node("summarize", speaker_summarize_node)
workflow.add_node("save_db", save_to_database_node)

# Add edges
workflow.set_entry_point("research")
workflow.add_edge("research", "retrieve_memory")
workflow.add_edge("retrieve_memory", "debate")

# Conditional edge: continue debate or vote?
workflow.add_conditional_edges(
    "debate",
    should_continue_debate,
    {
        "debate": "debate",  # Loop back for another round
        "vote": "vote"       # Move to voting
    }
)

workflow.add_edge("vote", "summarize")
workflow.add_edge("summarize", "save_db")
workflow.add_edge("save_db", END)

# Compile with checkpointing for memory
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("data/parliament.db")
app = workflow.compile(checkpointer=memory)
```

## Running a Session

```python
# Initialize state
initial_state = {
    "bill_id": "BILL_2026_001",
    "bill_title": "Climate Action Act 2026",
    "bill_text": load_bill("bills/climate_bill.txt"),
    "debate_round": 1,
    "max_rounds": 3,
    "debate_history": [],
    "agent_memories": {},
    "votes": {},
    "session_id": generate_uuid()
}

# Run the graph
config = {"configurable": {"thread_id": "session_001"}}
result = app.invoke(initial_state, config)

print(f"Bill: {result['bill_title']}")
print(f"Result: {result['vote_result']}")
print(f"\nSummary:\n{result['proceedings_summary']}")
```

## Advantages of LangGraph

1. **Built-in Cycles**: Debate rounds are natural loops
2. **State Persistence**: Automatic checkpointing to SQLite
3. **Conditional Logic**: Easy routing based on state
4. **Streaming**: Can stream agent responses in real-time
5. **Debugging**: Built-in visualization tools
6. **Human-in-the-Loop**: Can add approval nodes if needed

## Agent Inter-communication

Agents naturally see each other's statements through `state["debate_history"]`:

```python
def debate(self, bill, research, memories, previous_statements, round_number):
    # previous_statements contains all prior debate contributions
    context = f"""
    Bill: {bill}
    
    Research Summary: {research}
    
    Previous Debate Points:
    {format_debate_history(previous_statements)}
    
    Your past positions on similar topics:
    {format_memories(memories)}
    
    Based on the above, provide your position for Round {round_number}.
    Address specific points made by other representatives.
    """
    
    return self.llm.invoke(context)
```

## Next Steps

1. Implement each node function
2. Create agent factory from config
3. Set up database schema
4. Build CLI interface
5. Add visualization of graph execution
