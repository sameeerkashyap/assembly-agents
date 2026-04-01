#!/usr/bin/env python3
"""
Democratic Agents - Parliamentary Multi-Agent Simulation
Main entry point for running parliamentary sessions
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from parliament_graph import run_parliamentary_session


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run a parliamentary simulation with multi-agent debate and voting"
    )

    # Bill input
    parser.add_argument(
        "--bill",
        type=str,
        required=False,
        help="Path to bill text file",
    )

    # Chamber selection
    parser.add_argument(
        "--chamber",
        type=str,
        default="senate",
        choices=["senate", "house", "executive", "scotus", "all"],
        help="Which chamber(s) to simulate (default: senate)",
    )

    # Agent selection
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help='Comma-separated list of agent names to include, e.g. "Bernie Sanders,Ted Cruz"',
    )

    parser.add_argument(
        "--n-agents",
        type=int,
        default=10,
        help="Number of agents to load from profiles when --agents not specified (default: 10)",
    )

    # Simulation settings
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of debate rounds per chamber (default: 3)",
    )

    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Optional session ID (auto-generated if not provided)",
    )

    # View past session
    parser.add_argument(
        "--view-session",
        type=str,
        default=None,
        help="View transcript of a past session by ID",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save summary to this file path (optional)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed debug information",
    )

    args = parser.parse_args()

    # ── View past session ────────────────────────────────────────────────────
    if args.view_session:
        # TODO: Implement session replay from SQLite checkpointer
        # Query data/parliament.db for session_id and print transcript + vote record
        print(f"[TODO] Session replay not yet implemented. Session ID: {args.view_session}")
        sys.exit(0)

    # ── Load bill ────────────────────────────────────────────────────────────
    if not args.bill:
        parser.error("--bill is required unless --view-session is used")

    bill_path = Path(args.bill)
    if not bill_path.exists():
        print(f"Error: Bill file not found: {bill_path}")
        sys.exit(1)

    with open(bill_path) as f:
        bill_text = f.read()

    bill_title = bill_path.stem.replace("_", " ").title()

    # ── Parse agent list ─────────────────────────────────────────────────────
    agent_names = None
    if args.agents:
        agent_names = [name.strip() for name in args.agents.split(",")]

    # ── Run simulation ───────────────────────────────────────────────────────
    print(f"Bill:     {bill_title}")
    print(f"Chamber:  {args.chamber}")
    print(f"Rounds:   {args.rounds}")
    if agent_names:
        print(f"Agents:   {', '.join(agent_names)}")
    else:
        print(f"Agents:   top {args.n_agents} from profiles")
    print("=" * 80)

    result = run_parliamentary_session(
        bill_title=bill_title,
        bill_text=bill_text,
        chamber=args.chamber,
        agent_names=agent_names,
        n_agents=args.n_agents,
        max_debate_rounds=args.rounds,
        session_id=args.session_id,
    )

    # ── Save output ───────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(result.get("proceedings_summary", ""))
        print(f"\nSummary saved to: {output_path}")

    print("\nSession complete!")
    print(f"Session ID: {result.get('session_id', 'N/A')}")
    print(f"Result:     {result.get('vote_result', 'N/A')}")


if __name__ == "__main__":
    main()
