"""
SenatorAgent — Senate-specific debate and voting behavior.

Extends BaseAgent with:
  - Committee expertise context injected into prompts
  - Filibuster / cloture awareness (60-vote threshold)
  - State constituent pressure framing
"""

from __future__ import annotations
from .base_agent import BaseAgent


class SenatorAgent(BaseAgent):

    def build_system_prompt(self) -> str:
        """
        Extends base system prompt with Senate-specific framing.

        TODO: Add committee expertise — if the bill touches issues in the senator's
              committees, inject relevant domain knowledge into the prompt.
              e.g., Finance committee member on a tax bill → reference markup experience.

        TODO: Add recent voting record context — pull 2-3 related past votes from
              SQLite memory and mention them so the agent maintains consistency.
        """
        base = super().build_system_prompt()
        p = self.profile
        committees = p.get("committees", [])
        seniority = p.get("seniority", 0)
        leadership = p.get("leadership")

        senate_addendum = f"""
Senate context:
  Committees: {', '.join(committees) if committees else 'None listed'}
  Seniority: {seniority} years
  {f'Leadership role: {leadership}' if leadership else ''}

You are debating on the Senate floor. A 60-vote threshold is required for cloture \
(to end debate and proceed to a final vote). A simple majority (51) is needed to pass.
Reference your committee jurisdiction and seniority where relevant."""

        return base + senate_addendum

    def debate(self, bill_title: str, bill_summary: str, debate_history: list[dict], round_num: int) -> str:
        """
        TODO: In later rounds, have senators respond directly to the previous speaker
              by name — this creates more realistic back-and-forth dynamics.
              Detect the last speaker from debate_history and frame the response accordingly.
        """
        return super().debate(bill_title, bill_summary, debate_history, round_num)
