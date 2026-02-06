# AssemblyBots: Implementation Summary

## 🎯 Architecture Decision: Config-Based + LangGraph

**You asked:** Should I define business logic in code or use config files?

**Answer:** **Use config files!** ✅

### Why Config-Based Approach?

1. **Flexibility**: Add new agents by editing YAML, no code changes needed
2. **Experimentation**: Test different personas, voting biases, debate styles
3. **Separation of Concerns**: Logic (Python) separate from behavior (YAML)
4. **Version Control**: Track agent evolution through config commits
5. **Reusability**: Same `BaseAgent` class works for all roles

## 📁 What You Now Have

```
assemblybots/
├── config/
│   ├── agents.yaml              ✅ Agent definitions (roles, biases)
│   ├── system_prompts.yaml      ✅ Personality prompts per role
│   └── simulation.yaml          ✅ Sim settings (rounds, LLM config)
│
├── src/
│   ├── parliament_graph.py      ✅ LangGraph orchestration (skeleton)
│   ├── config_loader.py         ✅ Load configs with defaults
│   └── main.py                  ⏳ Entry point (next step)
│
├── LANGGRAPH_ARCHITECTURE.md    ✅ LangGraph deep dive
├── QUICK_START.md               ✅ How to use LangGraph
└── requirements.txt             ✅ Updated with LangGraph deps
```

## 🔄 How LangGraph Orchestrates Agents

### The Flow:

```
1. RESEARCH NODE
   └─ Speaker: Scrapes web for bill context
   └─ Output: Research summary + sources

2. MEMORY RETRIEVAL NODE
   └─ All Agents: Query DB for past decisions on similar topics
   └─ Output: Relevant historical context per agent

3. DEBATE NODE (Cyclic - Runs 3 times)
   └─ Round 1: Each agent states initial position
   └─ Round 2: Agents respond to each other's points
   └─ Round 3: Final arguments
   └─ Output: Full debate transcript
   
   [Conditional Edge: More rounds needed? Loop back : Move to vote]

4. VOTING NODE
   └─ Each agent votes with:
      - YES/NO
      - Full reasoning
      - Pros/Cons list
      - Data sources backing their position
   └─ Output: Vote tally + detailed justifications

5. SUMMARIZE NODE
   └─ Speaker: Creates proceedings summary
   └─ Output: Professional parliamentary record

6. SAVE TO DATABASE NODE
   └─ Persist: Session, votes, debate, reasoning
   └─ Output: Session ID for future reference
```

### Agent Inter-Communication

**Agents "talk" through shared state:**

```python
# State flows through all nodes
state = {
    "bill_text": "...",
    "debate_history": [
        {"agent": "conservative_1", "statement": "This costs too much"},
        {"agent": "liberal_1", "statement": "Re: conservative's cost concern..."},
        {"agent": "centrist_1", "statement": "Both make valid points..."}
    ],
    "votes": {...},
    # etc.
}

# Each agent sees what others said
def debate(self, state):
    others_said = state["debate_history"]
    my_response = f"Responding to {others_said[-1]['agent']}: ..."
    return my_response
```

## 💾 Memory System (Two Levels)

### 1. Short-Term (Within One Session)
- **Managed by:** LangGraph state
- **Contains:** Current debate history, bill info, votes
- **Used for:** Agents responding to each other in real-time

### 2. Long-Term (Across Sessions)
- **Managed by:** SQLite database
- **Contains:** Historical votes, past reasoning, outcomes
- **Used for:** "I voted YES on similar climate bills before because..."

**Example:**
```sql
SELECT * FROM agent_decisions 
WHERE agent_id = 'conservative_1' 
  AND topic LIKE '%climate%'
ORDER BY timestamp DESC 
LIMIT 5;
```

Agent retrieves their past 5 climate-related votes and uses that context when voting on a new climate bill.

## 🎨 Generic Agent Design

**Single agent class, behavior from config:**

```python
class BaseAgent:
    def __init__(self, config: AgentConfig, system_prompt: str, llm, db):
        self.id = config.id
        self.role = config.role
        self.party = config.party
        self.prompt = system_prompt  # From config/system_prompts.yaml
        self.bias = config.voting_bias
        self.llm = llm
        self.memory = db
    
    def debate(self, state):
        # Generic debate logic
        prompt = f"{self.prompt}\n\nBill: {state['bill_text']}\n..."
        return self.llm.invoke(prompt)
    
    def vote(self, state):
        # Generic voting logic
        prompt = f"{self.prompt}\n\nVote on: {state['bill_text']}\n..."
        return self.llm.invoke(prompt)
```

**Create agents from config:**
```python
# Load configs
loader = ConfigLoader()
agent_configs = loader.load_agents()  # From agents.yaml
prompts = loader.load_system_prompts()  # From system_prompts.yaml

# Create agents dynamically
agents = [
    BaseAgent(
        config=cfg,
        system_prompt=prompts[cfg.system_prompt_key],
        llm=initialize_llm(),
        db=database
    )
    for cfg in agent_configs
]
```

## 🎯 Next Implementation Steps

### Phase 1: Basic Setup (Start Here)
- [ ] Set up database schema (SQLite)
- [ ] Implement `BaseAgent` class
- [ ] Connect LLM (OpenAI/Anthropic API)
- [ ] Test single agent debate on a simple bill

### Phase 2: Multi-Agent Orchestration
- [ ] Integrate agents with LangGraph nodes
- [ ] Implement debate rounds (agent-to-agent communication)
- [ ] Add voting logic with pros/cons
- [ ] Test full simulation end-to-end

### Phase 3: Memory & Research
- [ ] Build database query functions for memory retrieval
- [ ] Implement web scraping for Speaker research
- [ ] Add memory context to agent prompts
- [ ] Store session results to DB

### Phase 4: Polish & Features
- [ ] CLI interface for running different bills
- [ ] Output formatting (markdown summaries)
- [ ] Visualization of debate flow
- [ ] Fact-checking (future enhancement)

## 🚀 Running Your First Simulation

Once implemented, usage will be:

```bash
# Basic run
python src/main.py --bill bills/climate_bill.txt

# Custom config
python src/main.py \
  --bill bills/healthcare_bill.txt \
  --agents config/agents.yaml \
  --max-rounds 5 \
  --session-id "session_001"

# View past session
python src/main.py --view-session "session_001"
```

## 🔑 Key Advantages of This Architecture

| Feature | Implementation | Benefit |
|---------|---------------|---------|
| **Config-driven** | YAML files define agents | Add new agents without code |
| **Generic agents** | Single `BaseAgent` class | Reusable, maintainable |
| **LangGraph orchestration** | State machine with loops | Natural debate flow |
| **Shared state** | `ParliamentState` dict | Agents naturally communicate |
| **Memory persistence** | SQLite + checkpointing | Agents learn from past |
| **Speaker agent** | Special capabilities | Web research + summarization |
| **Modular design** | Clear separation | Easy to test/extend |

## 📊 Database Schema

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    bill_title TEXT,
    bill_text TEXT,
    final_vote_result TEXT,  -- 'PASSED' or 'REJECTED'
    vote_count_yes INTEGER,
    vote_count_no INTEGER,
    timestamp DATETIME
);

CREATE TABLE agent_decisions (
    decision_id INTEGER PRIMARY KEY,
    session_id TEXT,
    agent_id TEXT,
    vote TEXT,  -- 'YES' or 'NO'
    reasoning TEXT,
    pros TEXT,  -- JSON
    cons TEXT,  -- JSON
    data_sources TEXT,  -- JSON
    timestamp DATETIME,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE debate_transcript (
    transcript_id INTEGER PRIMARY KEY,
    session_id TEXT,
    round_number INTEGER,
    agent_id TEXT,
    statement TEXT,
    timestamp DATETIME,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
```

## 💡 Example: How a Bill Flows Through the System

**Input:** `bills/climate_bill.txt`

```
1. Speaker researches climate policy (web scraping)
2. Agents retrieve their past climate votes from DB
3. DEBATE ROUND 1:
   - Conservative: "Too expensive, hurt economy"
   - Liberal: "Climate crisis urgent, investment needed"
   - Centrist: "Need cost-benefit analysis"
4. DEBATE ROUND 2:
   - Conservative responds to Liberal's urgency point
   - Liberal addresses Conservative's cost concern
   - Centrist synthesizes both views
5. DEBATE ROUND 3:
   - Final arguments from all
6. VOTING:
   - Conservative: NO (reasoning: fiscal impact)
   - Liberal: YES (reasoning: environmental necessity)
   - Centrist: YES (reasoning: balanced approach)
7. RESULT: PASSED (2 YES, 1 NO)
8. Speaker summarizes entire session
9. Everything saved to DB with session_id
```

**Output:** Full proceedings summary + database record

---

## 🎓 Learning Resources

- **LangGraph:** Read `QUICK_START.md` for detailed examples
- **Architecture:** See `LANGGRAPH_ARCHITECTURE.md` for deep dive
- **Config Format:** Check `config/*.yaml` files for examples

## 🤝 Ready to Build?

You now have:
✅ Architecture decided (config-based + LangGraph)  
✅ Project structure set up  
✅ Config files with agent definitions  
✅ LangGraph skeleton code  
✅ Clear implementation roadmap  

**Next Step:** Implement the database schema and `BaseAgent` class!

Let me know if you want help with any specific component! 🚀
