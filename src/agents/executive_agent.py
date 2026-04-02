"""
ExecutiveAgent — Cabinet member and President behavior.

Two modes:
  1. Cabinet deliberation: each secretary advises on bill within their department domain
  2. President decision: uses veto_factors heuristics to decide SIGN vs VETO

The President's final decision is:
  - Driven by veto_factors.issues_that_trigger_veto_recommendation
  - Weighted by Cabinet influence_on_president scores
  - Expressed as a signed executive statement or veto message
"""

from __future__ import annotations
from dataclasses import dataclass
from .base_agent import BaseAgent


@dataclass
class ExecutiveDecision:
    decision: str           # "SIGN" | "VETO"
    reasoning: str          # President's public statement
    cabinet_advice: dict    # name → recommendation string (SIGN/VETO + reason)
    veto_probability: float # 0–1, higher = more likely to veto


class ExecutiveAgent(BaseAgent):

    def build_system_prompt(self) -> str:
        """
        Executive-branch framing: department mission, veto factors, influence dynamics.

        TODO: For Cabinet members (non-President), add department_interests context
              so secretaries argue from their institutional role, not just personal ideology.
        """
        base = super().build_system_prompt()
        p = self.profile
        dept = p.get("department", "")
        dept_interests = p.get("department_interests", {})
        exec_behavior = p.get("executive_behavior", {})

        exec_addendum = f"""
Executive branch context:
  Department: {dept}
  Primary mission: {dept_interests.get('primary_mission', 'N/A')}
  Budget priority: {dept_interests.get('budget_priority', 'N/A')}
  Regulatory stance: {dept_interests.get('regulatory_stance', 'N/A')}
  Influence on President: {exec_behavior.get('influence_on_president', 0.5):.2f}
  Congressional relations: {exec_behavior.get('congressional_relations', 0.5):.2f}

You are advising the President on whether to sign or veto this legislation.
Frame your recommendation from your department's perspective and your policy expertise."""

        return base + exec_addendum

    def advise(self, bill_title: str, bill_summary: str, debate_summary: str) -> str:
        """
        Cabinet member provides a SIGN or VETO recommendation with reasoning.

        TODO: Weight the advice by influence_on_president when aggregating in
              president_decision(). High-influence Cabinet members (e.g., Chief of Staff)
              should count more than low-influence ones.
        """
        if self.llm is None:
            raise RuntimeError(f"Agent '{self.name}' has no LLM. Pass llm= at instantiation.")

        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = f"""Bill: {bill_title}
Summary: {bill_summary}

Congressional debate summary:
{debate_summary}

As {self.profile.get('role', 'Cabinet Secretary')}, provide a brief (2-3 sentence) \
recommendation to the President: should they SIGN or VETO this bill?
Start your response with SIGN or VETO."""

        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)


def make_president_decision(
    president_profile: dict,
    bill_analysis: dict,
    cabinet_advice: dict[str, str],
    llm=None,
) -> ExecutiveDecision:
    """
    Determine presidential SIGN vs VETO using veto_factors heuristics + cabinet advice.

    Algorithm:
      1. Check veto_factors.issues_that_trigger_veto_recommendation against bill's issueWeights
      2. If any trigger issues have weight > 0.4 in the bill → lean VETO
      3. Weight cabinet advice by executive_behavior.influence_on_president
      4. Combine into veto_probability; > 0.5 → VETO, else → SIGN
      5. LLM writes the presidential statement

    TODO: Implement full weighted cabinet aggregation.
    TODO: Add political capital consideration — high controversy_level bills cost more
          political capital, making the President more cautious.
    """
    veto_factors = president_profile.get("veto_factors", {})
    veto_trigger_issues: list[str] = veto_factors.get("issues_that_trigger_veto_recommendation", [])
    sign_trigger_issues: list[str] = veto_factors.get("issues_that_trigger_sign_recommendation", [])
    issue_weights: dict[str, float] = bill_analysis.get("issueWeights", {})

    # Heuristic: sum weight of veto-trigger issues touched by bill
    veto_weight = sum(issue_weights.get(issue, 0.0) for issue in veto_trigger_issues)
    sign_weight = sum(issue_weights.get(issue, 0.0) for issue in sign_trigger_issues)
    total = veto_weight + sign_weight

    if total == 0:
        # No strong triggers → fall back to issue alignment via base vote calculator
        from heuristics.vote_calculator import calculate_vote_probability
        result = calculate_vote_probability(president_profile, bill_analysis)
        veto_probability = 1.0 - result["vote_probability"]
    else:
        veto_probability = veto_weight / total

    # TODO: Adjust veto_probability by weighted cabinet sign/veto ratio

    decision = "VETO" if veto_probability > 0.5 else "SIGN"

    # Generate presidential statement
    reasoning = f"The President has decided to {decision} this legislation."
    if llm is not None:
        from langchain_core.messages import HumanMessage, SystemMessage
        name = president_profile.get("name", "The President")
        party = president_profile.get("party", "")
        cabinet_summary = "\n".join(
            f"  {adv_name}: {str(advice)[:120]}" for adv_name, advice in cabinet_advice.items()
        )
        prompt = f"""Cabinet advice received:
{cabinet_summary}

Heuristic analysis: veto probability = {veto_probability:.2f}

You have decided to {decision} this bill. Write a 2-3 sentence presidential statement \
in your voice. Reference the bill's policy impact. Do not start with "I"."""
        response = llm.invoke([
            SystemMessage(content=f"You are {name}, President of the United States ({party} party). Speak with presidential authority and gravity."),
            HumanMessage(content=prompt),
        ])
        reasoning = response.content if hasattr(response, "content") else str(response)

    return ExecutiveDecision(
        decision=decision,
        reasoning=reasoning,
        cabinet_advice=cabinet_advice,
        veto_probability=round(veto_probability, 4),
    )
