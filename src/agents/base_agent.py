"""
BaseAgent — wraps an ai-gov-simulator profile into a debating, voting LangChain agent.

All chamber-specific agents (SenatorAgent, ExecutiveAgent, SCOTUSAgent) extend this.

Key responsibilities:
  1. Build a system prompt that encodes the agent's heuristic profile
  2. Generate debate statements in-character given the current state
  3. Cast a vote (direction from heuristics, reasoning from LLM)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class VoteRecord:
    agent_name: str
    vote: str                  # "YES" | "NO"
    vote_probability: float    # from heuristic engine
    reasoning: str             # LLM-generated, in-character
    pros: list[str]
    cons: list[str]
    issue_alignment: float
    modifiers: dict


class BaseAgent:
    """
    A legislative agent whose ideology is anchored by ai-gov-simulator heuristics.

    profile: dict loaded from ProfileStore (senate/executive/scotus schema)
    llm:     LangChain BaseChatModel — use haiku for debate, sonnet for vote reasoning

    TODO: Accept separate debate_llm and vote_llm once cost tuning is done.
    """

    def __init__(self, profile: dict, llm=None):
        self.profile = profile
        self.llm = llm
        self.name: str = profile.get("name", "Unknown")
        self._system_prompt: Optional[str] = None

    # ── System Prompt ────────────────────────────────────────────────────────

    def build_system_prompt(self) -> str:
        """
        Construct the agent's character prompt from their heuristic profile.
        Called once and cached.

        TODO: Add biography.notable_story for richer characterization.
        TODO: Add lobbying.lobbying_vulnerability so agents reference donor pressure naturally.
        TODO: Tune token length — target ~300 tokens for system prompt to leave room for debate context.
        """
        p = self.profile
        personality = p.get("personality", {})
        behavior = p.get("behavior", {})
        electoral = p.get("electoral", {})
        state_ctx = p.get("state_context", {})
        issues = p.get("issues", {})

        issue_lines = "\n".join(
            f"  {k}: {v:.2f}" for k, v in sorted(issues.items(), key=lambda x: x[0])
        )

        return f"""You are {p.get('name')}, {p.get('role', p.get('party', ''))} \
({p.get('party', '')}{('-' + p.get('state', '')) if p.get('state') else ''}).

Ideology: {personality.get('archetype', 'moderate')}, {personality.get('temperament', 'measured')} temperament.
Known for: {personality.get('known_for', 'principled positions')}

Issue positions (0.00=far left, 1.00=far right):
{issue_lines}

Behavioral traits:
  Party loyalty:        {behavior.get('party_loyalty', 0.5):.2f}
  Bipartisan tendency:  {behavior.get('bipartisan_index', 0.5):.2f}
  Deal-making:          {behavior.get('deal_maker', 0.5):.2f}
  Ideological rigidity: {behavior.get('ideological_rigidity', 0.5):.2f}

What can move you: {personality.get('pressure_point', 'strong constituent interest')}
What makes you dig in: {personality.get('dealbreaker', 'violations of core principles')}

Electoral: {electoral.get('seat_safety', 'safe')} seat, next election {electoral.get('next_election', 'N/A')}
State priority: {state_ctx.get('hot_button', 'constituent welfare')}

You are participating in a legislative debate simulation. Stay in character at all times.
Speak as this person would — reference your actual positions, state interests, and political history.
Be specific. Cite real policy consequences. Do not break character."""

    def get_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self.build_system_prompt()
        return self._system_prompt

    # ── Debate ───────────────────────────────────────────────────────────────

    def debate(self, bill_title: str, bill_summary: str, debate_history: list[dict], round_num: int) -> str:
        """
        Generate one in-character debate statement.

        debate_history: list of {"agent": name, "statement": text, "round": int}
        Returns the statement string.

        Uses haiku model (fast + cheap for N×M turns across all agents).

        TODO: Cap debate_history context at last ~2000 tokens to avoid hitting context limits
              in later rounds of a large chamber debate.
        """
        if self.llm is None:
            # TODO: Remove stub once LLM is wired in
            return f"[{self.name} - STUB] My position on {bill_title} reflects my constituents' interests."

        history_text = self._format_debate_history(debate_history)
        prompt = f"""Bill under debate: {bill_title}
Summary: {bill_summary}

Debate transcript so far (Round {round_num}):
{history_text if history_text else '(You are the first to speak this round.)'}

It is your turn to speak. Give your position on this bill in 2-3 sentences. \
Be specific to the bill's provisions. Reference your state's interests or your record if relevant."""

        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    # ── Voting ───────────────────────────────────────────────────────────────

    def vote(
        self,
        bill_title: str,
        bill_summary: str,
        debate_history: list[dict],
        vote_result: "VoteProbabilityResult",  # from vote_calculator
    ) -> VoteRecord:
        """
        Cast a vote with in-character reasoning.

        vote_result: output of calculate_vote_probability() — provides direction + probability.
        The LLM writes reasoning for the predetermined direction; it does NOT choose the direction.

        Uses sonnet model (quality matters for the official record).

        TODO: Add pros/cons extraction — prompt the LLM to list 2 pros and 2 cons
              even when voting YES (acknowledges complexity) or NO (acknowledges merits).
        """
        direction = vote_result["vote_direction"]
        probability = vote_result["vote_probability"]

        if self.llm is None:
            # TODO: Remove stub
            return VoteRecord(
                agent_name=self.name,
                vote=direction,
                vote_probability=probability,
                reasoning=f"[STUB] {self.name} votes {direction} based on issue alignment {probability:.2f}.",
                pros=["Stub pro"],
                cons=["Stub con"],
                issue_alignment=vote_result["issue_alignment"],
                modifiers=vote_result["modifiers"],
            )

        history_text = self._format_debate_history(debate_history)
        prompt = f"""Bill: {bill_title}
Summary: {bill_summary}

Full debate transcript:
{history_text}

Based on this debate and your positions, you are voting {direction} on this bill.
(Your heuristic issue alignment score: {probability:.2f})

In 2-3 sentences, explain your vote in character. Be specific about which bill provisions \
drive your decision. Then provide:
PROS (2 bullet points — even if voting NO, acknowledge what the bill does well):
CONS (2 bullet points — even if voting YES, acknowledge real concerns):"""

        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)

        reasoning, pros, cons = self._parse_vote_response(raw)

        return VoteRecord(
            agent_name=self.name,
            vote=direction,
            vote_probability=probability,
            reasoning=reasoning,
            pros=pros,
            cons=cons,
            issue_alignment=vote_result["issue_alignment"],
            modifiers=vote_result["modifiers"],
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _format_debate_history(self, history: list[dict]) -> str:
        if not history:
            return ""
        lines = []
        for entry in history:
            lines.append(f"[{entry.get('agent', '?')} - Round {entry.get('round', '?')}]")
            lines.append(entry.get("statement", ""))
            lines.append("")
        return "\n".join(lines)

    def _parse_vote_response(self, raw: str) -> tuple[str, list[str], list[str]]:
        """
        Parse LLM vote response into (reasoning, pros, cons).

        TODO: Make this more robust — use structured output (json_mode) on the vote prompt
              so we get clean fields instead of parsing free text.
        """
        lines = raw.strip().split("\n")
        reasoning_lines = []
        pros = []
        cons = []

        mode = "reasoning"
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.upper().startswith("PROS"):
                mode = "pros"
                continue
            if stripped.upper().startswith("CONS"):
                mode = "cons"
                continue

            if mode == "reasoning":
                reasoning_lines.append(stripped)
            elif mode == "pros" and stripped.startswith(("-", "•", "*")):
                pros.append(stripped.lstrip("-•* "))
            elif mode == "cons" and stripped.startswith(("-", "•", "*")):
                cons.append(stripped.lstrip("-•* "))

        reasoning = " ".join(reasoning_lines) if reasoning_lines else raw[:300]
        return reasoning, pros[:2], cons[:2]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} party={self.profile.get('party', '?')!r}>"
