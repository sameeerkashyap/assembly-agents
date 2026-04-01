# Democratic Agents — Architecture & Execution Plan

## Overview

Democratic Agents is a **LangGraph-powered multi-chamber legislative simulation**. Real congressional profiles from `ai-gov-simulator` are loaded as heuristic data and injected into LangChain agent instances. Each agent debates and votes in character based on their actual issue scores, behavioral traits, and electoral context.

---

## Source of Truth: ai-gov-simulator Heuristics

Profiles are ported from `ai-gov-simulator/data/government-profiles.json` (generated via its prompts). Each profile type provides:

### Senate / House Agent Heuristics
| Field | Description |
|---|---|
| `issues` | 15-dimension score (0=far left, 1=far right): `immigration`, `healthcare`, `gun_rights`, `climate_energy`, `defense_military`, `education`, `tech_regulation`, `criminal_justice`, `trade_tariffs`, `abortion_social`, `government_spending`, `foreign_policy_hawks`, `civil_liberties`, `labor_unions`, `taxes_spending` |
| `behavior.party_loyalty` | 0–1, how reliably they vote party line |
| `behavior.bipartisan_index` | 0–1, tendency to cross the aisle |
| `behavior.lobby_susceptibility` | 0–1, how much donor pressure moves them |
| `behavior.media_sensitivity` | 0–1, how much public opinion shapes votes |
| `behavior.deal_maker` | 0–1, willingness to trade concessions |
| `behavior.ideological_rigidity` | 0–1, resistance to changing position mid-debate |
| `electoral.seat_safety` | `safe` / `lean` / `toss-up` — affects risk tolerance |
| `electoral.primary_vulnerable` | Whether a primary challenge makes them more extreme |
| `personality.archetype` | `hawk`, `moderate`, `progressive`, `libertarian`, etc. |
| `personality.pressure_point` | The framing that could actually move their vote |
| `personality.dealbreaker` | The argument that makes them dig in harder |
| `state_context.hot_button` | Local issue that can override national ideology |
| `lobbying.top_industries` | Industries with financial leverage |

### Executive Branch Heuristics
Same `issues` + `executive_behavior` (influence_on_president, congressional_relations, etc.) + `veto_factors` (issues that trigger veto vs sign recommendation).

### SCOTUS Heuristics
`constitutional_issues` (14 dimensions) + `judicial_behavior` (deference_to_precedent, willingness_to_overturn, etc.) + `voting_patterns` (agrees/disagrees most with which justice).

---

## Vote Calculation (Heuristic Engine)

For each agent voting on a bill:

```
1. Bill Analysis → issueWeights (how much bill touches each issue) + issuePositions (what stance it takes, 0-1)

2. Agreement Score:
   agreement = Σ(issueWeight[i] × (1 − |agentIssue[i] − billPosition[i]|)) / Σ(issueWeight[i])
   → Raw alignment between agent's ideology and bill's stance

3. Behavioral Modifiers:
   + party_loyalty modifier  → if bill aligns with agent's party: +loyalty×0.15, else −loyalty×0.15
   + lobby modifier          → if bill helps top industries: +susceptibility×0.10
   + electoral modifier      → toss-up seat or primary_vulnerable: push toward base, reduce deal-making
   + media modifier          → if bill is high controversy: media_sensitivity pulls toward popular opinion

4. Final vote_probability = clamp(agreement + modifiers, 0.0, 1.0)
   → probability > 0.5 → YES (LLM writes in-character reasoning)
   → probability ≤ 0.5 → NO  (LLM writes in-character reasoning)
```

The LLM is given the vote direction and asked to write *in-character reasoning* using the agent's personality, pressure_point, and dealbreaker.

---

## System Architecture

```
democratic-agents/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py          # BaseAgent: heuristics → LangChain agent
│   │   ├── senator_agent.py       # Senate-specific behavior + filibuster
│   │   ├── executive_agent.py     # Cabinet deliberation + president veto
│   │   └── scotus_agent.py        # Justice constitutional review
│   │
│   ├── heuristics/
│   │   ├── __init__.py
│   │   ├── profiles.py            # Load + validate profiles from JSON
│   │   ├── vote_calculator.py     # Heuristic vote probability engine
│   │   └── bill_analyzer.py       # LLM: bill text → issueWeights + issuePositions
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── debate_manager.py      # Structured N-round debate runner
│   │   └── vote_aggregator.py     # Chamber-level tallying + threshold logic
│   │
│   ├── parliament_graph.py        # LangGraph state machine (multi-chamber)
│   ├── config_loader.py           # Load YAML configs
│   └── main.py                    # CLI entry point
│
├── config/
│   ├── agents.yaml                # Override/subset of agents to include
│   ├── simulation.yaml            # Debate rounds, thresholds, LLM settings
│   └── system_prompts.yaml        # Prompt templates per agent type
│
├── data/
│   ├── senate_profiles.json       # Ported from ai-gov-simulator (generated)
│   ├── executive_profiles.json    # Cabinet + President profiles
│   └── scotus_profiles.json       # SCOTUS justice profiles
│
├── bills/                         # Input bill text files
├── output/                        # Session transcripts, vote records, summaries
└── ai-gov-simulator/              # Source repo (read-only reference)
```

---

## LangGraph Multi-Chamber State Machine

```
                         ┌─────────────┐
                         │  bill_input │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │bill_analysis│  LLM → issueWeights + issuePositions
                         └──────┬──────┘
                                │
               ┌────────────────▼─────────────────┐
               │         HOUSE CHAMBER             │
               │  debate_node (N rounds)           │
               │  → vote_node (majority: 218/435)  │
               └────────────┬─────────┬────────────┘
                      PASSED│         │FAILED → END
                            │
               ┌────────────▼─────────────────────┐
               │         SENATE CHAMBER            │
               │  debate_node (N rounds)           │
               │  → vote_node (simple majority 51) │
               └────────────┬─────────┬────────────┘
                      PASSED│         │FAILED → END
                            │
               ┌────────────▼─────────────────────┐
               │       EXECUTIVE REVIEW            │
               │  cabinet_deliberation_node        │
               │  → president_decision_node        │
               │     (SIGN / VETO based on         │
               │      veto_factors heuristics)     │
               └────────────┬─────────┬────────────┘
                      SIGNED│         │VETOED
                            │         └──────► override_node (2/3 majority)
               ┌────────────▼─────────────────────┐
               │     SCOTUS REVIEW (optional)      │
               │  constitutional_analysis_node     │
               │  → conference_vote_node (5/9)     │
               └────────────┬─────────────────────┘
                            │
                     ┌──────▼──────┐
                     │final_summary│
                     └─────────────┘
```

### State Schema

```python
class ParliamentState(TypedDict):
    # Bill
    bill_id: str
    bill_title: str
    bill_text: str
    bill_analysis: dict          # issueWeights, issuePositions, factions, partySupport

    # Research
    speaker_research: str
    web_sources: List[str]

    # Per-chamber tracking
    current_chamber: str                       # "house" | "senate" | "executive" | "scotus"
    chamber_results: Dict[str, dict]           # chamber → {passed, tally, transcript}

    # Active debate
    debate_round: int
    max_rounds: int
    debate_history: Annotated[List[dict], add_messages]
    active_agents: List[str]                   # profile names in current chamber

    # Voting
    votes: Dict[str, dict]                     # agent_name → {vote, probability, reasoning, pros, cons}
    vote_result: str                           # "PASSED" | "REJECTED" | "SIGNED" | "VETOED"
    vote_tally: Dict[str, int]

    # Session
    session_id: str
    timestamp: str
    proceedings_summary: str
```

---

## Debate Structure (Per Chamber)

Each chamber runs `N` configurable rounds (default: 3 for Senate, 2 for House, 1 for Cabinet).

**Per round, for each agent in turn:**
1. Agent receives: bill text + bill_analysis + their heuristic profile summary + full debate_history so far
2. Agent generates an in-character statement (~150 tokens, haiku model for speed)
3. Statement appended to `debate_history` — subsequent agents in the same round see it

**Vote turn (after final debate round):**
1. Heuristic engine computes `vote_probability` from issue alignment + behavioral modifiers
2. Vote direction (YES/NO) is determined by `probability > 0.5`
3. LLM writes in-character reasoning for that direction (sonnet model for quality)

**Agent system prompt (built dynamically from profile):**
```
You are {name}, {role} ({party}-{state}).
Ideology: {archetype}, {temperament} temperament.
Known for: {known_for}

Issue positions (0=far left, 1=far right):
  climate_energy: {score}  |  healthcare: {score}  |  gun_rights: {score} ...

What can move you: {pressure_point}
What makes you dig in: {dealbreaker}
State priority: {hot_button}
Electoral situation: {seat_safety} seat, next election {next_election}

Stay in character. Be specific. Reference real policy and data.
```

---

## Implementation Phases

### Phase 1 — Heuristics Layer
- [ ] `heuristics/profiles.py`: load `data/senate_profiles.json`, validate schema, expose `get_profile(name)`, `get_roster(chamber, n)`
- [ ] `heuristics/vote_calculator.py`: `calculate_vote_probability(profile, bill_analysis)` with party + lobby + electoral modifiers
- [ ] `heuristics/bill_analyzer.py`: LLM call (haiku) converting bill text → `BillAnalysis` matching ai-gov-simulator schema
- [ ] Generate `data/*.json` profiles — port the Anthropic API calls from ai-gov-simulator's `lib/generate.js` to Python

### Phase 2 — Agent Layer
- [ ] `agents/base_agent.py`: `BaseAgent(profile, llm)` — builds system prompt, exposes `debate(state) -> str` and `vote(state, direction) -> VoteRecord`
- [ ] `agents/senator_agent.py`: extends BaseAgent — adds filibuster framing, committee expertise context
- [ ] `agents/executive_agent.py`: cabinet agents advise; president decides via `veto_factors` heuristics
- [ ] `agents/scotus_agent.py`: uses `judicial_philosophy` + `constitutional_issues` instead of `issues`

### Phase 3 — Orchestration Layer
- [ ] `orchestration/debate_manager.py`: `run_debate(agents, state, n_rounds) -> updated_state`
- [ ] `orchestration/vote_aggregator.py`: `tally_votes(votes, chamber) -> ChamberResult` using correct thresholds (51 Senate, 218 House, 5/9 SCOTUS, 2/3 override)

### Phase 4 — LangGraph Integration
- [ ] Update `parliament_graph.py` — replace mock nodes with real BaseAgent calls
- [ ] Add multi-chamber conditional routing
- [ ] SQLite checkpointing for session persistence + replay

### Phase 5 — CLI & Output
- [ ] `main.py`: argparse CLI (`--bill`, `--chamber`, `--agents`, `--rounds`, `--session-id`, `--view-session`)
- [ ] Markdown transcript + JSON vote record output per session

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Profile source | Port from ai-gov-simulator JSON | Already generated with real DW-NOMINATE + interest group data |
| Vote direction | Heuristic engine, LLM writes reasoning | Prevents LLM from hallucinating ideology; agents are deterministic |
| Debate model | `claude-haiku-4-5` | Speed + cost across N×M agent turns per round |
| Vote reasoning model | `claude-sonnet-4-6` | Quality matters for the official record |
| Chamber subset | Default 10 senators (configurable) | Full 100-senator runs are expensive; subset captures political spread |
| Memory | SQLite via LangGraph checkpointer | Agents reference past votes on similar bills |
| Bill analysis | Same schema as ai-gov-simulator | `issueWeights` + `issuePositions` directly feeds vote calculator |

