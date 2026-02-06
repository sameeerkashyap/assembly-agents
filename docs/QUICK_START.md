# LangGraph Multi-Agent Orchestration Guide

## 🎯 Quick Answer: How to Use LangGraph

**LangGraph orchestrates your multi-agent parliament through a state graph where:**

1. **State flows through nodes** (research → debate → vote → summarize)
2. **Each node modifies the state** (adds debate statements, votes, etc.)
3. **Edges control flow** (conditional loops for debate rounds)
4. **Checkpointing persists state** (agents remember past sessions)

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestration                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │Research │ => │ Memory  │ => │ Debate  │ => │  Vote   │  │
│  │  Node   │    │  Node   │    │  Node   │    │  Node   │  │
│  └─────────┘    └─────────┘    └────┬────┘    └─────────┘  │
│                                      │                       │
│                                      └─ Loop if more         │
│                                         rounds needed        │
└──────────────────────────────────────────────────────────────┘
                          ▲
                          │
                    Shared State
                  (ParliamentState)
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  Config-Based Agents                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Conservative Rep    Liberal Rep    Centrist Rep    Speaker │
│  ┌─────────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐│
│  │ Config: ✓   │   │ Config: ✓│   │ Config: ✓│   │Config:✓││
│  │ Prompt: ✓   │   │ Prompt: ✓│   │ Prompt: ✓│   │Prompt:✓││
│  │ Memory: ✓   │   │ Memory: ✓│   │ Memory: ✓│   │Memory:✓││
│  └─────────────┘   └──────────┘   └──────────┘   └────────┘│
│                                                              │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    SQLite Database                           │
│  • Session history  • Agent decisions  • Debate transcripts  │
└──────────────────────────────────────────────────────────────┘
```

## 🔑 Key Concepts

### 1. **State** (ParliamentState)
The shared data structure that flows through all nodes:

```python
class ParliamentState(TypedDict):
    bill_text: str              # Input
    debate_history: List[Dict]  # Accumulates over time
    votes: Dict[str, Dict]      # Final votes
    vote_result: str            # PASSED/REJECTED
    # ... and more
```

### 2. **Nodes** (Functions)
Each node is a Python function that:
- Receives current state
- Performs some action (research, debate, vote)
- Returns updated state

```python
def debate_node(state: ParliamentState) -> ParliamentState:
    # Agents debate based on current state
    statements = generate_debate_statements(state)
    return {"debate_history": statements}  # Updates state
```

### 3. **Edges** (Flow Control)
Edges connect nodes and can be:
- **Static**: Always go from A → B
- **Conditional**: Choose path based on state

```python
# Conditional: loop back to debate or move to vote?
def should_continue_debate(state):
    return "debate" if state["debate_round"] < 3 else "vote"
```

### 4. **Checkpointing** (Memory)
LangGraph automatically saves state to SQLite:
- Each session has a unique thread_id
- State is persisted between runs
- Agents can query past sessions

## 🚀 How It Works for Your Use Case

### Step 1: Define Your State
```python
# In parliament_graph.py - already created!
class ParliamentState(TypedDict):
    bill_text: str
    debate_history: Annotated[List[Dict], add_messages]  # Auto-appends!
    votes: Dict[str, Dict]
    # etc.
```

### Step 2: Create Node Functions
```python
# Each phase is a node
def research_bill_node(state):
    # Speaker researches bill
    research = speaker_agent.research(state["bill_text"])
    return {"speaker_research": research}

def debate_node(state):
    # All agents debate
    statements = []
    for agent in agents:
        stmt = agent.debate(state)
        statements.append(stmt)
    return {"debate_history": statements}  # Automatically appends!

def voting_node(state):
    # All agents vote
    votes = {agent.id: agent.vote(state) for agent in agents}
    return {"votes": votes}
```

### Step 3: Build the Graph
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(ParliamentState)

# Add nodes
workflow.add_node("research", research_bill_node)
workflow.add_node("debate", debate_node)
workflow.add_node("vote", voting_node)

# Add edges
workflow.set_entry_point("research")
workflow.add_edge("research", "debate")

# Conditional edge for debate loops
workflow.add_conditional_edges(
    "debate",
    should_continue_debate,  # Function that returns "debate" or "vote"
    {
        "debate": "debate",   # Loop back
        "vote": "vote"        # Move to voting
    }
)

workflow.add_edge("vote", END)
```

### Step 4: Compile with Checkpointing
```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("data/parliament.db")
app = workflow.compile(checkpointer=memory)
```

### Step 5: Run a Session
```python
# Initialize state
initial_state = {
    "bill_text": load_bill("bills/climate_bill.txt"),
    "debate_round": 1,
    "max_rounds": 3,
    "debate_history": [],
    "votes": {},
    # etc.
}

# Run with a unique session ID
config = {"configurable": {"thread_id": "session_001"}}
result = app.invoke(initial_state, config)

print(result["vote_result"])  # "PASSED" or "REJECTED"
```

## 🔄 Agent Inter-Communication

Agents communicate through the **shared state**:

```python
def debate_node(state):
    for agent in agents:
        # Each agent sees all previous statements
        previous_statements = state["debate_history"]
        
        # Agent crafts response addressing others
        statement = agent.debate(
            bill=state["bill_text"],
            others_said=previous_statements  # See what others said!
        )
        
        # Add to debate history
        state["debate_history"].append({
            "agent": agent.id,
            "statement": statement
        })
    
    return state
```

**Example flow:**
1. Conservative Rep 1 speaks: "This bill costs too much"
2. Liberal Rep 1 sees that in `state["debate_history"]`
3. Liberal Rep 1 responds: "Re: Conservative Rep's cost concerns, studies show..."
4. Centrist sees both, synthesizes: "Both make valid points, but data suggests..."

## 💾 Memory System

### Two Levels of Memory:

1. **Session Memory** (within one debate)
   - Maintained by LangGraph state
   - Debate history flows through nodes
   - All agents see entire conversation

2. **Long-Term Memory** (across sessions)
   - Stored in SQLite database
   - Agents query before voting: "What did I say about climate bills before?"
   - Retrieved in `retrieve_memories_node`

```python
def retrieve_memories_node(state):
    for agent in agents:
        # Query database for this agent's past decisions
        past_decisions = db.query(
            "SELECT * FROM agent_decisions WHERE agent_id=? AND topic LIKE ?",
            agent.id, 
            f"%{state['bill_title']}%"
        )
        
        state["agent_memories"][agent.id] = past_decisions
    
    return state
```

Agents use this when voting:
```python
def vote(self, state):
    my_memories = state["agent_memories"][self.id]
    
    prompt = f"""
    Bill: {state["bill_text"]}
    
    Your past relevant decisions:
    {my_memories}
    
    Debate so far:
    {state["debate_history"]}
    
    How do you vote and why?
    """
    # Agent generates vote with reasoning
```

## 📊 Advantages for Your Project

| Feature | How LangGraph Helps |
|---------|-------------------|
| **Multiple debate rounds** | Conditional edges create natural loops |
| **Agent communication** | Shared state = all agents see same context |
| **Memory persistence** | Built-in SQLite checkpointing |
| **Complex flow control** | Conditional routing based on state |
| **Debugging** | Can visualize graph, inspect state at each node |
| **Streaming** | Can stream agent responses in real-time |
| **Recovery** | Checkpoint allows resuming failed sessions |

## 🎨 Config-Based Approach

**Your agents are config-driven, LangGraph orchestrates them:**

```yaml
# config/agents.yaml
agents:
  - id: conservative_1
    role: Conservative Representative
    system_prompt_key: conservative
```

```python
# At runtime:
configs = load_agent_configs()  # From YAML
prompts = load_system_prompts()  # From YAML

# Create agents dynamically
agents = [
    BaseAgent(config, prompts[config.system_prompt_key])
    for config in configs
]

# Use them in LangGraph nodes
def debate_node(state):
    for agent in agents:  # Dynamically loaded!
        statement = agent.debate(state)
        # ...
```

This means:
- ✅ Add new agents by editing YAML, no code changes
- ✅ Modify personalities by changing prompts
- ✅ Different configs for different simulations

## 🔧 Next Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the skeleton:**
   ```bash
   python src/parliament_graph.py
   ```

3. **Implement real LLM calls** (replace mocks in nodes)

4. **Set up database schema** for long-term memory

5. **Add web scraping** for Speaker research

6. **Build CLI** for running different bills

## 📚 Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Multi-Agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [Checkpointing Guide](https://langchain-ai.github.io/langgraph/how-tos/persistence/)

---

**Bottom Line:** LangGraph gives you a **state machine** where agents naturally communicate through shared state, debate rounds loop automatically, and everything persists to a database. Perfect for your parliamentary simulation! 🏛️
