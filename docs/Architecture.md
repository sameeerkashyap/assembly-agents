assemblyAgents/
├── config/
│   ├── agents.yaml              # Agent role definitions
│   ├── system_prompts.yaml      # System prompts per role
│   └── simulation.yaml          # Simulation settings
│
├── src/
│   ├── agents/
│   │   ├── base_agent.py        # Generic agent class
│   │   ├── speaker_agent.py     # Special speaker agent
│   │   └── agent_factory.py     # Create agents from config
│   │
│   ├── orchestration/
│   │   ├── parliament.py        # Main orchestrator
│   │   ├── debate_manager.py    # Manages debate rounds
│   │   └── voting_system.py     # Handles voting logic
│   │
│   ├── memory/
│   │   ├── db_manager.py        # Database operations
│   │   └── schemas.py           # DB schemas
│   │
│   ├── tools/
│   │   └── web_scraper.py       # For speaker agent
│   │
│   └── main.py                   # Entry point
│
├── data/
│   └── parliament.db             # SQLite database
│
└── bills/
    └── example_bill.txt          # Input bills