# Democratic Agents

A multi-agent legislative simulation powered by LangGraph. Real congressional profiles from `ai-gov-simulator` are loaded as heuristic data and injected into LLM agent instances. Each agent debates and votes in character based on their actual issue scores, behavioral traits, and electoral context.

![Democratic Agents](./images/cover.png)

---

## How It Works

### 1. Profile-Grounded Agents

Profiles are sourced from `ai-gov-simulator/data/government-profiles.json` — 100 Senators, 436 House members, 25 Executive branch members, and 9 SCOTUS justices. Each profile encodes:

- **15-dimension issue scores** (0 = far left, 1 = far right): immigration, healthcare, climate_energy, gun_rights, taxes_spending, etc.
- **Behavioral traits**: party_loyalty, bipartisan_index, lobby_susceptibility, media_sensitivity, deal_maker, ideological_rigidity
- **Electoral context**: seat_safety (safe / lean / toss-up), primary_vulnerable
- **Personality**: archetype, pressure_point (what moves them), dealbreaker (what hardens them)
- **State context**: local hot_button issue that can override national ideology

### 2. Heuristic Vote Engine

Vote direction is determined by the heuristic engine before the LLM is invoked. This keeps votes anchored in real political data — the LLM writes in-character reasoning for a predetermined direction, never overriding the outcome.

```
agreement = Σ(issueWeight[i] × (1 − |agentIssue[i] − billPosition[i]|)) / Σ(issueWeight[i])

Modifiers applied:
  + party_loyalty    → ±0.15 if bill aligns/opposes agent's party
  + lobby            → ±0.10 if bill affects agent's top donor industries
  + electoral        → amplifies party modifier if toss-up seat or primary-vulnerable
  + media_sensitivity → regresses toward 0.5 on high-controversy bills

vote_probability = clamp(agreement + modifiers, 0.0, 1.0)
vote = YES if probability > 0.5
```

### 3. LangGraph State Machine

The full bill lifecycle runs as a LangGraph directed graph:

```
bill_analysis → senate → executive → scotus* → summarize → save_db

* SCOTUS only triggered if bill raises constitutional issues (constitutionalIssues score > 0.2)
```

Each chamber node:
1. Loads profiles from `ProfileStore` (balanced R/D subset for legislative chambers)
2. Instantiates agents with a debate LLM (fast) and vote LLM (quality)
3. Runs N rounds of structured debate via `run_debate()`
4. Tallies votes via `tally_votes()` with chamber-appropriate thresholds
5. Routes to the next chamber based on outcome

---

## Architecture

```
assemblybots/
├── src/
│   ├── parliament_graph.py        # LangGraph state machine — entry point
│   ├── llm_factory.py             # LLM provider (Anthropic API or local HF on MPS)
│   ├── main.py                    # CLI
│   │
│   ├── agents/
│   │   ├── base_agent.py          # BaseAgent: profile → system prompt, debate(), vote()
│   │   ├── senator_agent.py       # Senate framing: committees, cloture, seniority
│   │   ├── executive_agent.py     # Cabinet advise() + make_president_decision()
│   │   └── scotus_agent.py        # Justice deliberate() + rule() with constitutional issues
│   │
│   ├── heuristics/
│   │   ├── profiles.py            # ProfileStore: loads all 4 chambers from government-profiles.json
│   │   ├── vote_calculator.py     # Heuristic vote probability engine
│   │   └── bill_analyzer.py       # LLM: bill text → issueWeights + issuePositions
│   │
│   └── orchestration/
│       ├── debate_manager.py      # run_debate(): N-round structured debate runner
│       └── vote_aggregator.py     # tally_votes(): chamber thresholds + ChamberResult
│
├── config/
│   ├── simulation.yaml            # Rounds, agent counts, LLM settings, thresholds
│   └── rosters.yaml               # 119th Congress rosters with full metadata
│
├── data/
│   └── government-profiles.json   # Source of truth — all profiles (from ai-gov-simulator)
│
├── bills/                         # Input bill text files
├── output/
│   ├── transcripts/               # Per-session Markdown records
│   └── votes/                     # Per-session JSON vote records
└── ai-gov-simulator/              # Source repo (profile generation)
```

### Chamber Thresholds

| Chamber | Threshold | Notes |
|---|---|---|
| Senate | >50% of simulated subset | Proportional to 51/100 |
| House | >50% of simulated subset | Proportional to 218/435 |
| Executive | Heuristic veto_probability | veto_factors + cabinet advice |
| SCOTUS | 5/9 justices | Always all 9 justices |
| Veto override | >66% | Not yet implemented |

### LLM Configuration

Two LLM roles per session:

| Role | Anthropic | Local (MPS) |
|---|---|---|
| Debate turns | `claude-haiku-4-5` | configured model, 256 tokens, temp 0.8 |
| Vote reasoning + judicial opinions | `claude-sonnet-4-6` | same model, 512 tokens, temp 0.4 |

Set `ANTHROPIC_API_KEY` to use the API. Without it, the local HuggingFace model runs on Apple Silicon MPS automatically (`torch.float16`, ~3.5GB RAM).

---

## Installation

```bash
git clone <repository-url>
cd assemblybots
uv sync
```

## Usage

```bash
# Run a Senate simulation on the sample bill
export ANTHROPIC_API_KEY=your_key   # optional — falls back to local model
uv run python src/main.py --bill bills/climate_action_act_2026.txt --chamber senate --n-agents 4 --rounds 2

# View a past session
uv run python src/main.py --view-session session_20260401_e10e20
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--bill` | required | Path to bill text file |
| `--chamber` | `senate` | `senate`, `house`, `executive`, `scotus` |
| `--n-agents` | `10` | Agents to load (balanced R/D; use 4 for a quick test) |
| `--rounds` | `3` | Debate rounds per chamber |
| `--session-id` | auto | Set to replay or name a session |
| `--view-session` | — | Print transcript + votes from a past session ID |

---

## Sample Sessions — Climate Action Act 2026

Two sessions were run on the same bill with different agent pool sizes, producing opposite outcomes.

---

### Session 1 — REJECTED (`session_20260401_a57c0d`)

**15 senators (7R, 6D, 2 R-moderates) | 1 debate round | Final: REJECTED 5–10**

| Senator | Party | Vote | Probability |
|---|---|---|---|
| John Barrasso | R | YES | 0.895 |
| Mitch McConnell | R | YES | 0.814 |
| John Thune | R | YES | 0.782 |
| John Cornyn | R | YES | 0.774 |
| Chuck Grassley | R | YES | 0.725 |
| Susan Collins | R | NO | 0.414 |
| Lisa Murkowski | R | NO | 0.354 |
| Amy Klobuchar | D | NO | 0.230 |
| Dick Durbin | D | NO | 0.211 |
| Chuck Schumer | D | NO | 0.194 |
| Ron Wyden | D | NO | 0.154 |
| Bernie Sanders | I | NO | 0.117 |
| Maria Cantwell | D | NO | 0.107 |
| Patty Murray | D | NO | 0.126 |
| Sheldon Whitehouse | D | NO | 0.067 |

A counterintuitive partisan split — 5 Republicans voted YES while all 8 Democrats and 1 Independent voted NO. The heuristic engine reflects the bill's actual `issuePositions` from the LLM analysis: the bill was scored as taking moderate-right positions on climate (market-based carbon pricing, rural energy grants, emphasis on energy security), which aligned better with centrist Republicans than with progressive Democrats whose issue scores sit far left on climate_energy (0.1–0.2) and thus registered low agreement with the bill's centrist framing.

Notable individual votes:
- **Collins (R-ME) and Murkowski (R-AK)** broke from their party — both cited specific industry concerns (Maine's fishing industry, Alaska's oil sector) that the heuristic captured via `state_context.hot_button` pulling their probability below 0.5
- **Whitehouse (D-RI)** had the lowest probability of all at 0.067 — his profile is strongly progressive on climate and the bill's market-based approach registered as insufficient
- **Sanders (I-VT)** voted NO at 0.117 — his debate statement supported the bill but his heuristic scored the bill's conservative market mechanisms as misaligned with his position
- **Schumer (D-NY)** voted NO despite being Senate leader, citing the bill's regional carbon pricing as undermining New York's own stricter standards

---

### Session 2 — SIGNED (`session_20260401_e10e20`)

**4 senators (2R, 2D) | 2 debate rounds | Final: SIGNED**

| Senator | Party | Vote | Probability |
|---|---|---|---|
| John Thune | R | YES | 0.722 |
| John Barrasso | R | YES | 0.688 |
| Chuck Schumer | D | YES | 0.720 |
| Dick Durbin | D | YES | 0.710 |

Senate passed 4–0. Despite Thune and Barrasso arguing strongly against the bill in debate (citing agricultural costs and energy independence), their underlying issue alignment scores pushed them to YES — illustrating the key design principle: **debate rhetoric and vote direction are computed independently**.

### Executive Result: SIGNED (veto_probability = 0.28)

Cabinet vote was 14 SIGN / 9 VETO. Notable splits:
- **SIGN:** Rubio (national security / counter-China framing), Rollins/Agriculture (farmer transition support), Ratcliffe/CIA (counter-China energy strategy)
- **VETO:** Bessent/Treasury (market instability), Hegseth/Defense (military energy costs), Wright/Energy (undermines fossil fuel dominance), Waltz/UN (cedes ground to international agreements), Wiles/Chief of Staff (political risk to heartland states)

The presidential heuristic found weak overlap between the bill's `issueWeights` and the president's `veto_factors`, producing a 0.28 veto probability → SIGN. The LLM generated a bipartisan statement framing the bill as an economic opportunity.

### SCOTUS: Not triggered

`constitutionalIssues` scores were below the 0.2 threshold in both sessions — no constitutional challenge flagged.

---

## Drawbacks

**1. Heuristic vote ≠ real roll-call accuracy**
The issue alignment formula is a reasonable approximation but hasn't been calibrated against historical votes. Agents like Thune (R-SD) voting YES on a climate bill reflects profile scores, not real-world likelihood. Modifier weights (party loyalty ±0.15, lobby ±0.10) were set by intuition, not regression.

**2. Small agent subsets distort outcomes**
With `--n-agents 4`, the R/D split is 2/2 and there are no moderates, independents, or regional outliers. A 4-senator simulation is useful for development but not politically representative.

**3. No cross-round persuasion**
Agents generate debate statements that are contextually aware (they see the full prior transcript), but the vote probability is computed from static heuristics — nothing a senator says in debate can actually change another senator's vote direction.

**4. Local model quality gap**
Using `Llama-3.2-3B-Instruct` locally produces grammatically correct but shallow debate. Senators cite real policy (RFS, carbon pricing, Wyoming coal) but arguments are repetitive across rounds. `claude-haiku-4-5` produces substantially more varied and specific statements.

**5. SCOTUS vote calculator is a proxy**
SCOTUS justices use the same `calculate_vote_probability()` as legislators. This maps issue alignment to uphold probability, which is a rough proxy. A proper constitutional vote calculator would use `constitutional_issues` scores and `judicial_behavior` modifiers (deference_to_precedent, willingness_to_overturn) instead.

**6. House chamber not yet implemented**
The graph routes `bill_analysis → senate`, skipping the House entirely. Per Article I, bills must pass the House first.

---

## Planned Improvements

- **House chamber node** — add `house_chamber_node` before Senate in the graph
- **Filibuster / cloture** — Senate bills should require 60 votes to end debate, not just 51 to pass
- **Veto override node** — route VETOED bills to a 2/3 override vote rather than directly to summary
- **Constitutional vote calculator** — SCOTUS-specific scoring using `constitutional_issues` + `judicial_behavior` modifiers
- **Persuasion modeling** — if a debate statement contains an opponent's `pressure_point`, apply a small modifier to their next-round vote probability
- **Per-agent memory** — before debate, each senator queries `data/parliament.db` for their 2–3 most recent votes on similar bills and references them in their system prompt
- **Modifier calibration** — run against historical roll-call votes (DW-NOMINATE dataset) and tune weights to maximize accuracy
- **Async debate** — run all agents in a round concurrently with `asyncio.gather` to cut wall-clock time for large chambers
