#!/usr/bin/env python3
"""
AssemblyBots - Parliamentary Multi-Agent Simulation
Main entry point for running parliamentary sessions
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from parliament_graph import run_parliamentary_session
from config_loader import ConfigLoader


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run a parliamentary simulation with multi-agent debate and voting"
    )
    
    # Bill input
    parser.add_argument(
        "--bill",
        type=str,
        required=True,
        help="Path to bill text file"
    )
    
    # Configuration
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing config files (default: config/)"
    )
    
    # Simulation settings
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum number of debate rounds (default: 3)"
    )
    
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Optional session ID (auto-generated if not provided)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save summary to this file (optional)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed debug information"
    )
    
    args = parser.parse_args()
    
    # Load bill
    bill_path = Path(args.bill)
    if not bill_path.exists():
        print(f"Error: Bill file not found: {bill_path}")
        sys.exit(1)
    
    with open(bill_path, 'r') as f:
        bill_text = f.read()
    
    bill_title = bill_path.stem.replace('_', ' ').title()
    
    # Load configuration
    print(f"Loading configuration from {args.config_dir}/")
    loader = ConfigLoader(args.config_dir)
    
    agents = loader.load_agents()
    prompts = loader.load_system_prompts()
    sim_config = loader.load_simulation_config()
    
    print(f"Loaded {len(agents)} agents:")
    for agent in agents:
        print(f"  - {agent.id}: {agent.role} ({agent.party})")
    
    # Run simulation
    print(f"\nStarting parliamentary session...")
    print(f"Bill: {bill_title}")
    print(f"Max debate rounds: {args.max_rounds}")
    print("=" * 80)
    
    result = run_parliamentary_session(
        bill_title=bill_title,
        bill_text=bill_text,
        max_debate_rounds=args.max_rounds,
        session_id=args.session_id
    )
    
    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(result["proceedings_summary"])
        
        print(f"\nSummary saved to: {output_path}")
    
    print("\n✓ Session complete!")
    print(f"Session ID: {result['session_id']}")
    print(f"Result: {result['vote_result']}")


if __name__ == "__main__":
    main()
