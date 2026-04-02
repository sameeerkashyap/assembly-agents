"""
Generate agent profiles from ai-gov-simulator prompts and save to data/*.json.

Ported from:
  ai-gov-simulator/lib/rosters.js  — SENATE_ROSTER, EXECUTIVE_ROSTER, SCOTUS_ROSTER
  ai-gov-simulator/lib/prompts.js  — SENATE_SYSTEM/PROMPT, EXEC_SYSTEM/PROMPT, SCOTUS_SYSTEM/PROMPT
  ai-gov-simulator/lib/generate.js — callAPI, generateBranch, validateProfile

Run once to populate data/*.json before running the simulation:
    python src/heuristics/generate_profiles.py --branch senate
    python src/heuristcs/generate_profiles.py --branch executive
    python src/heuristics/generate_profiles.py --branch scotus
    python src/heuristics/generate_profiles.py --branch all

Requires: pip install transformers torch accelerate langchain-huggingface
          Model is loaded from config/simulation.yaml (llm.model).
          Override with: HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3 python ...
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_factory import get_raw_pipeline

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
ROSTERS_FILE = CONFIG_DIR / "rosters.yaml"


# ══════════════════════════════════════════════════════════════════════════════
# ROSTER LOADER
# Rosters live in config/rosters.yaml — edit there to add/remove members.
# ══════════════════════════════════════════════════════════════════════════════

def _load_rosters() -> dict:
    with open(ROSTERS_FILE) as f:
        return yaml.safe_load(f)

_rosters = _load_rosters()
SENATE_ROSTER: list[dict] = _rosters["senate"]
EXECUTIVE_ROSTER: list[dict] = _rosters["executive"]
SCOTUS_ROSTER: list[dict] = _rosters["scotus"]



# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS (ported from ai-gov-simulator/lib/prompts.js)
# ══════════════════════════════════════════════════════════════════════════════

SENATE_SYSTEM = """You are a nonpartisan congressional research analyst with encyclopedic knowledge of every sitting U.S. Senator. You have access to:
- DW-NOMINATE scores (first and second dimension) from voteview.com
- Interest group ratings: NRA (guns), LCV (environment), ACLU (civil liberties), Heritage Action (conservative), AFL-CIO (labor), NumbersUSA (immigration restriction), FreedomWorks (fiscal conservative), Planned Parenthood (reproductive rights), ADA (liberal)
- Complete roll-call voting records from the 116th-119th Congress
- Campaign finance data from OpenSecrets.org — top donors, industry contributions, PAC support, total raised
- Lobbying disclosure data — which industries and organizations lobby each senator most aggressively
- Biographical information: birth, education, career history, family, military service, religion
- State demographics, key industries, and electoral history

ACCURACY IS PARAMOUNT. These profiles drive a realistic policy simulation AND a debate system where agents argue in character. The biography must be detailed and real enough that the senator's dialogue feels authentic.

RULES:
1. Base ALL issue scores on ACTUAL voting records and interest group ratings, not vibes or party averages
2. Every senator is unique — Rand Paul ≠ Ted Cruz ≠ Mike Lee even though all are "conservative"
3. State context matters enormously — a Republican from Maine ≠ a Republican from Alabama
4. Capture known idiosyncrasies: Paul is libertarian on foreign policy and civil liberties; Collins breaks on social issues; Fetterman is progressive on economics but hawkish on Israel
5. For behavioral scores: party_loyalty should reflect actual party-line voting percentage; bipartisan_index should reflect Lugar Center scores or similar
6. BIOGRAPHY must include real facts — actual hometown, actual schools, actual career before politics. Do NOT invent details. If unsure, omit rather than fabricate.
7. Senators who resigned or lost (Rubio, Bob Casey, Debbie Stabenow) — skip them, return a stub with {"name": "...", "status": "no_longer_serving"}

Return ONLY a valid JSON array. No markdown fences, no commentary, no preamble."""

EXEC_SYSTEM = """You are a nonpartisan political analyst specializing in the U.S. Executive Branch during the second Trump administration (2025-2026). You have deep knowledge of every Cabinet member's policy positions, public statements, management record, congressional testimony, and the departments they oversee.

ACCURACY IS PARAMOUNT. These profiles determine how the simulation handles veto decisions and executive policy reactions.

RULES:
1. The President's issue scores are the most important — they determine veto/sign. Base them on actual executive orders signed, bills signed/vetoed, and stated positions in 2025-2026
2. Cabinet members influence the President within their domain — capture both their personal views AND their department's institutional interests
3. Some officials are heterodox: RFK Jr. is liberal on environment but populist-right on pharma/vaccines; Gabbard is non-interventionist but authoritarian on domestic security; Lori Chavez-DeRemer was notably pro-union for a Republican
4. Capture how much real influence each person has on the President vs. being a figurehead
5. Include their relationship with Congress — some Cabinet members have good Hill relationships (former members) while others are antagonistic

Return ONLY a valid JSON array. No markdown, no commentary."""

SCOTUS_SYSTEM = """You are a Supreme Court legal analyst with expert knowledge of every sitting justice's complete body of opinions, dissents, concurrences, oral argument style, and judicial philosophy. You are familiar with the October 2024-2025 term decisions and the current composition of voting blocs.

ACCURACY IS PARAMOUNT. These profiles determine how the simulation handles constitutional review of legislation.

RULES:
1. Base scores on ACTUAL opinions and voting patterns, not media narratives
2. Roberts is NOT simply "conservative" — he's an institutionalist who will side with liberals to preserve Court legitimacy
3. Gorsuch's textualism sometimes produces outcomes coded as "liberal" (Bostock, McGirt, tribal sovereignty cases) — capture this
4. Thomas and Alito are both very conservative but differ: Thomas is willing to overturn almost anything; Alito is more strategic and incrementalist
5. Kavanaugh is the current median justice — his incrementalist concurrences often define the actual holding
6. Barrett has shown more independence than expected on standing, procedure, and some regulatory questions
7. The three liberals differ internally: Sotomayor is the most passionate and furthest left; Kagan is the most strategic and best coalition-builder; Jackson is bold but still establishing her approach

Return ONLY a valid JSON array. No markdown, no commentary."""


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS (ported from ai-gov-simulator/lib/prompts.js)
# ══════════════════════════════════════════════════════════════════════════════

def build_senate_prompt(batch: list[dict]) -> str:
    lines = []
    for m in batch:
        parts = [f"- {m['name']} ({m['party']}-{m['state']})"]
        if m.get("leadership"):
            parts.append(f", {m['leadership']}")
        parts.append(f", committees: [{', '.join(m.get('committees', []))}]")
        parts.append(f", {m['seniority']}yr seniority, Class {m['classNum']}")
        if m.get("note"):
            parts.append(f" — NOTE: {m['note']}")
        lines.append("".join(parts))

    schema = """{
  "name": "Full Name",
  "state": "XX",
  "party": "R|D|I",
  "committees": ["Committee1"],
  "seniority": 0,
  "leadership": null,
  "class": 1,
  "issues": {
    "immigration": 0.00, "taxes_spending": 0.00, "healthcare": 0.00,
    "gun_rights": 0.00, "climate_energy": 0.00, "defense_military": 0.00,
    "education": 0.00, "tech_regulation": 0.00, "criminal_justice": 0.00,
    "trade_tariffs": 0.00, "abortion_social": 0.00, "government_spending": 0.00,
    "foreign_policy_hawks": 0.00, "civil_liberties": 0.00, "labor_unions": 0.00
  },
  "behavior": {
    "party_loyalty": 0.00, "bipartisan_index": 0.00, "lobby_susceptibility": 0.00,
    "media_sensitivity": 0.00, "deal_maker": 0.00, "ideological_rigidity": 0.00
  },
  "electoral": {
    "seat_safety": "safe|lean|toss-up", "last_margin": 0,
    "next_election": 2026, "primary_vulnerable": false
  },
  "personality": {
    "archetype": "hawk|establishment|moderate|populist|progressive|libertarian|centrist",
    "temperament": "combative|measured|folksy|academic|fiery|reserved",
    "known_for": "One sentence",
    "pressure_point": "The argument that could actually move them",
    "dealbreaker": "The argument that makes them dig in harder"
  },
  "interests": ["industry1"],
  "state_context": {"key_industries": ["industry1"], "hot_button": "Top local issue"},
  "biography": {
    "born": "City, State, Year", "age": 0,
    "education": "Degrees and institutions",
    "career_before_politics": "Career history",
    "family": "Spouse, children",
    "military_service": null,
    "religion": "Affiliation",
    "personal_style": "How they present",
    "notable_story": "Defining anecdote",
    "hobbies_interests": "Outside politics"
  },
  "lobbying": {
    "top_industries": ["industry with $amount"],
    "top_donors": ["Organization"],
    "total_raised_last_cycle": "$X million",
    "pac_support": ["PAC"],
    "notable_donor_conflicts": "Known conflicts",
    "lobbying_vulnerability": "Which industries have leverage and why"
  }
}

SCORING GUIDE (use two decimal places):
0.00-0.10: Far left (Sanders on most issues)
0.10-0.25: Strong liberal (Warren, Markey)
0.25-0.40: Solid liberal (most Democrats)
0.40-0.50: Center-left (moderate Democrats like Kelly, Rosen)
0.50-0.60: True centrist (Collins on some issues)
0.60-0.75: Center-right to conservative (most Republicans)
0.75-0.90: Solidly conservative (Cotton, Hawley)
0.90-1.00: Far right (Cruz on immigration, Lee on spending)

CRITICAL: A single senator WILL have different scores across issues."""

    return (
        f"Generate profiles for these {len(batch)} senators:\n"
        + "\n".join(lines)
        + "\n\nReturn a JSON array using this exact schema:\n"
        + schema
        + "\n\nFor any member who has left the Senate, return: {\"name\": \"...\", \"status\": \"no_longer_serving\"}"
    )


def build_exec_prompt(batch: list[dict]) -> str:
    lines = [f"- {m['name']}, {m['role']} ({m['department']}): {m['background']}" for m in batch]
    return (
        f"Generate profiles for these {len(batch)} Executive Branch officials:\n"
        + "\n".join(lines)
        + "\n\nReturn a JSON array. Each object must include: name, role, department, issues (same 15 keys as senate), "
        + "executive_behavior (influence_on_president, congressional_relations, media_presence, ideological_rigidity, "
        + "institutional_loyalty, policy_independence — all 0.00-1.00), "
        + "veto_factors (issues_that_trigger_veto_recommendation: list, issues_that_trigger_sign_recommendation: list, threshold: str), "
        + "personality (archetype, temperament, known_for, management_style, pressure_point, dealbreaker), "
        + "department_interests (primary_mission, budget_priority, regulatory_stance), biography, lobbying."
    )


def build_scotus_prompt(batch: list[dict]) -> str:
    lines = [
        f"- {j['name']}, {j['role']}, appointed by {j['appointedBy']} ({j['year']}). Prior: {j['priorRole']}. {j['background']}"
        for j in batch
    ]
    return (
        f"Generate profiles for all {len(batch)} Supreme Court justices:\n"
        + "\n".join(lines)
        + "\n\nReturn a JSON array. Each object must include: name, role, appointed_by, year_appointed, "
        + "judicial_philosophy (primary, secondary, description), "
        + "constitutional_issues (executive_power, individual_rights_vs_government, federal_vs_state_power, "
        + "regulatory_authority_admin_state, criminal_defendant_rights, free_speech_1A, gun_rights_2A, "
        + "religious_liberty, abortion_reproductive_rights, commerce_clause_scope, equal_protection_discrimination, "
        + "voting_rights, immigration_executive_authority, environmental_regulation — all 0.00-1.00), "
        + "judicial_behavior (deference_to_precedent, deference_to_legislature, willingness_to_overturn, "
        + "solo_concurrence_tendency, swing_vote_frequency, opinion_writing_influence, coalition_builder — all 0.00-1.00), "
        + "personality (temperament, oral_argument_style, known_for, likely_to_strike_down, likely_to_uphold), "
        + "voting_patterns (agrees_most_with, disagrees_most_with, majority_rate, dissent_rate), biography."
        + "\n\nScoring for constitutional_issues: 0.00=most liberal/expansive, 1.00=most conservative/restrictive."
    )


# ══════════════════════════════════════════════════════════════════════════════
# API + GENERATION (ported from ai-gov-simulator/lib/generate.js)
# ══════════════════════════════════════════════════════════════════════════════

def call_api(system: str, prompt: str, retries: int = 3) -> list[dict]:
    """Call local transformers model and return parsed JSON list."""
    pipe, max_new_tokens, temperature = get_raw_pipeline()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(1, retries + 1):
        try:
            output = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
            # pipeline with return_full_text=False returns the new tokens only
            generated = output[0]["generated_text"]
            if isinstance(generated, list):
                # Some pipelines return the full message list; grab last assistant turn
                text = generated[-1]["content"]
            else:
                text = generated

            clean = text.replace("```json", "").replace("```", "").strip()
            # Find the JSON array bounds
            start = clean.find("[")
            end = clean.rfind("]") + 1
            if start == -1 or end == 0:
                print(f"\n  [DEBUG] Raw model output ({len(text)} chars): {repr(text[:500])}")
                raise json.JSONDecodeError("No JSON array found in response", clean, 0)
            return json.loads(clean[start:end])
        except json.JSONDecodeError as e:
            if attempt < retries:
                print(f"  JSON parse failed (attempt {attempt}): {e}. Retrying...")
                time.sleep(2)
            else:
                raise
        except Exception as e:
            if attempt < retries:
                print(f"  Model error (attempt {attempt}): {e}. Retrying...")
                time.sleep(2)
            else:
                raise


def generate_branch(
    roster: list[dict],
    system: str,
    prompt_fn,
    batch_size: int = 5,
) -> tuple[list[dict], list[dict]]:
    """
    Generate profiles for all roster members in batches.
    Returns (profiles, errors).
    """
    profiles = []
    errors = []
    batches = [roster[i : i + batch_size] for i in range(0, len(roster), batch_size)]

    for i, batch in enumerate(batches):
        names = ", ".join(m["name"].split()[-1] for m in batch)
        print(f"  Batch {i+1}/{len(batches)} [{names}]...", end=" ", flush=True)
        try:
            result = call_api(system, prompt_fn(batch))
            if isinstance(result, list):
                active = [p for p in result if p.get("status") != "no_longer_serving"]
                skipped = len(result) - len(active)
                profiles.extend(active)
                print(f"OK ({len(active)} profiles{', ' + str(skipped) + ' skipped' if skipped else ''})")
            else:
                errors.append({"batch": i, "error": "Response was not a list"})
                print("ERROR: not a list")
        except Exception as e:
            errors.append({"batch": i, "members": [m["name"] for m in batch], "error": str(e)})
            print(f"ERROR: {e}")

        if i < len(batches) - 1:
            time.sleep(2)  # Rate limit buffer between batches

    return profiles, errors


def validate_profiles(profiles: list[dict], ptype: str) -> list[str]:
    """Basic validation — mirrors generate.js validateProfile()."""
    warnings = []
    for p in profiles:
        name = p.get("name", "UNKNOWN")
        if ptype in ("senate", "executive"):
            if not p.get("issues"):
                warnings.append(f"{name}: missing issues")
            else:
                vals = list(p["issues"].values())
                if any(v < 0 or v > 1 for v in vals):
                    warnings.append(f"{name}: issue score out of range")
                if len(set(round(v, 2) for v in vals)) == 1:
                    warnings.append(f"{name}: all issue scores identical")
                avg = sum(vals) / len(vals)
                if p.get("party") == "D" and avg > 0.65:
                    warnings.append(f"{name}: Democrat with high conservative avg ({avg:.2f})")
                if p.get("party") == "R" and avg < 0.35:
                    warnings.append(f"{name}: Republican with low conservative avg ({avg:.2f})")
            if not p.get("behavior"):
                warnings.append(f"{name}: missing behavior")
            if not p.get("personality"):
                warnings.append(f"{name}: missing personality")
        if ptype == "scotus":
            if not p.get("constitutional_issues"):
                warnings.append(f"{name}: missing constitutional_issues")
            if not p.get("judicial_behavior"):
                warnings.append(f"{name}: missing judicial_behavior")
        if ptype == "executive":
            if not p.get("veto_factors"):
                warnings.append(f"{name}: missing veto_factors")
    return warnings


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate Democratic Agent profiles using local LLM"
    )
    parser.add_argument(
        "--branch",
        choices=["senate", "executive", "scotus", "all"],
        default="senate",
        help="Which branch to generate profiles for",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Senators per API call (default: 5). Lower = less likely to hit token limits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing profile files (default: skip if file exists)",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)

    branches = ["senate", "executive", "scotus"] if args.branch == "all" else [args.branch]

    for branch in branches:
        out_path = DATA_DIR / f"{branch}_profiles.json"

        if out_path.exists() and not args.overwrite:
            print(f"[{branch.upper()}] {out_path} already exists — skipping (use --overwrite to regenerate)")
            continue

        print(f"\n[{branch.upper()}] Generating profiles...")

        if branch == "senate":
            roster = SENATE_ROSTER
            system = SENATE_SYSTEM
            prompt_fn = build_senate_prompt
        elif branch == "executive":
            roster = EXECUTIVE_ROSTER
            system = EXEC_SYSTEM
            prompt_fn = build_exec_prompt
        else:  # scotus
            roster = SCOTUS_ROSTER
            system = SCOTUS_SYSTEM
            prompt_fn = build_scotus_prompt

        print(f"  Roster size: {len(roster)} members, batch size: {args.batch_size}")
        profiles, errors = generate_branch(roster, system, prompt_fn, batch_size=args.batch_size)

        # Validate
        warnings = validate_profiles(profiles, branch)
        if warnings:
            print(f"  Warnings ({len(warnings)}):")
            for w in warnings:
                print(f"    - {w}")

        if errors:
            print(f"  Errors ({len(errors)}):")
            for e in errors:
                print(f"    - Batch {e['batch']}: {e['error']}")

        # Save
        with open(out_path, "w") as f:
            json.dump(profiles, f, indent=2)

        party_counts = {}
        for p in profiles:
            party = p.get("party", "?")
            party_counts[party] = party_counts.get(party, 0) + 1

        print(f"  Saved {len(profiles)} profiles to {out_path}")
        print(f"  Party breakdown: {party_counts}")

    print("\nDone. Run the simulation with:")
    print("  python src/main.py --bill bills/climate_action_act_2026.txt --chamber senate")


if __name__ == "__main__":
    main()
