"""
SCOTUSAgent — Supreme Court justice constitutional review.

Differs from legislative agents in key ways:
  - Uses constitutional_issues scores (not policy issues)
  - Judicial philosophy drives framing (originalist, textualist, living constitution, etc.)
  - No party loyalty modifier — judicial independence framing
  - Oral argument style shapes how the agent asks questions during deliberation
  - Vote threshold: 5/9 majority

Deliberation structure:
  - Round 1: Each justice states constitutional concerns (not a yes/no vote yet)
  - Round 2: Justices respond to each other's constitutional arguments
  - Vote: Each justice rules UPHOLD or STRIKE DOWN with legal reasoning
"""

from __future__ import annotations
from dataclasses import dataclass
from .base_agent import BaseAgent


@dataclass
class ConstitutionalRuling:
    justice_name: str
    ruling: str                    # "UPHOLD" | "STRIKE_DOWN"
    vote_probability: float        # >0.5 = leans uphold
    legal_reasoning: str           # in-character judicial opinion
    constitutional_basis: str      # which clause/doctrine drives the ruling
    opinion_type: str              # "majority" | "concurrence" | "dissent" (set post-aggregation)


class SCOTUSAgent(BaseAgent):

    def build_system_prompt(self) -> str:
        """
        SCOTUS framing: judicial philosophy, constitutional approach, oral argument style.

        TODO: Add voting_patterns context — if this justice typically agrees with
              another justice who has already spoken, note that alignment.
        TODO: Add landmark opinions this justice has written — makes reasoning more authentic.
        """
        p = self.profile
        philosophy = p.get("judicial_philosophy", {})
        judicial_behavior = p.get("judicial_behavior", {})
        personality = p.get("personality", {})
        voting_patterns = p.get("voting_patterns", {})
        const_issues = p.get("constitutional_issues", {})

        const_lines = "\n".join(
            f"  {k}: {v:.2f}" for k, v in sorted(const_issues.items(), key=lambda x: x[0])
        )

        return f"""You are Justice {p.get('name')}, {p.get('role', 'Associate Justice')} \
of the United States Supreme Court.

Judicial philosophy: {philosophy.get('primary', 'pragmatist')} \
(secondary: {philosophy.get('secondary', 'none')})
{philosophy.get('description', '')}

Oral argument style: {personality.get('oral_argument_style', 'methodical')}
Known for: {personality.get('known_for', 'careful jurisprudence')}

Constitutional issue positions (0.00=most liberal/expansive, 1.00=most conservative/restrictive):
{const_lines}

Judicial behavior:
  Deference to precedent: {judicial_behavior.get('deference_to_precedent', 0.5):.2f}
  Willingness to overturn: {judicial_behavior.get('willingness_to_overturn', 0.5):.2f}
  Swing vote frequency:   {judicial_behavior.get('swing_vote_frequency', 0.5):.2f}
  Coalition builder:      {judicial_behavior.get('coalition_builder', 0.5):.2f}

Tends to uphold: {personality.get('likely_to_uphold', 'legislation with strong precedential support')}
Tends to strike: {personality.get('likely_to_strike_down', 'legislation that overreaches constitutional authority')}

Most aligned with: Justice {voting_patterns.get('agrees_most_with', 'N/A')}
Most divergent from: Justice {voting_patterns.get('disagrees_most_with', 'N/A')}

You are participating in Supreme Court conference deliberation. Write as a jurist, not a politician.
Ground your analysis in constitutional text, precedent, and your judicial philosophy.
Do not refer to political parties or election outcomes."""

    def deliberate(self, bill_title: str, constitutional_issues: dict, debate_history: list[dict], round_num: int) -> str:
        """
        Generate a justice's constitutional analysis during conference.

        constitutional_issues: from bill_analysis — which constitutional dimensions the bill touches
        Round 1: State constitutional concerns
        Round 2: Respond to colleagues' constitutional arguments

        TODO: In round 2, extract specific arguments from debate_history and have justices
              respond to them by name (e.g., "I agree with Justice X's point on the Commerce Clause").
        """
        return super().debate(bill_title, str(constitutional_issues), debate_history, round_num)

    def rule(
        self,
        bill_title: str,
        constitutional_issues: dict,
        debate_history: list[dict],
        uphold_probability: float,
    ) -> ConstitutionalRuling:
        """
        Issue a constitutional ruling (UPHOLD or STRIKE_DOWN).

        uphold_probability: from constitutional vote calculator (see TODO below).
        LLM writes the legal opinion for the predetermined ruling direction.

        TODO: Implement constitutional vote calculator — mirror vote_calculator.py but
              using constitutional_issues scores and judicial_behavior modifiers instead
              of policy issues and party/lobby modifiers.
        """
        ruling = "UPHOLD" if uphold_probability > 0.5 else "STRIKE_DOWN"

        if self.vote_llm is None:
            raise RuntimeError(f"Justice '{self.name}' has no vote LLM. Pass vote_llm= at instantiation.")

        from langchain_core.messages import HumanMessage, SystemMessage
        history_text = self._format_debate_history(debate_history)
        active_issues = ", ".join(
            k for k, v in constitutional_issues.items() if v > 0.2
        ) or "general constitutional principles"
        prompt = f"""Bill under review: {bill_title}
Constitutional issues at stake: {active_issues}

Conference deliberation:
{history_text}

You are voting to {ruling} this legislation. In 2-3 sentences, write your judicial \
opinion in character. Ground it in constitutional text, precedent, or your judicial \
philosophy. End with one sentence naming the primary constitutional basis for your ruling."""

        response = self.vote_llm.invoke([
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ])
        raw = response.content if hasattr(response, "content") else str(response)

        # Last sentence is the constitutional basis
        sentences = [s.strip() for s in raw.replace("\n", " ").split(".") if s.strip()]
        constitutional_basis = sentences[-1] if sentences else "Constitutional principles"
        legal_reasoning = raw.strip()

        return ConstitutionalRuling(
            justice_name=self.name,
            ruling=ruling,
            vote_probability=uphold_probability,
            legal_reasoning=legal_reasoning,
            constitutional_basis=constitutional_basis,
            opinion_type="majority",  # set post-aggregation by scotus_chamber_node
        )
