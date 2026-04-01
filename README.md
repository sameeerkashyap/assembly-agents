# Democratic Agents 🏛️

A sophisticated multi-agent system (MAS) for simulating democratic legislative processes using state machine orchestration.

## Overview

Democratic Agents moves beyond simple chatbots to implement a structured state machine for simulating legislative democracy. The system orchestrates multiple AI agents through distinct legislative phases, modeling real-world parliamentary procedures.

![Democratic Agents](./images/cover.png)


## Legislative State Machine

The system operates through the following states:

1. **Bill Reading** - Agents review and analyze proposed legislation
2. **Debate** - Structured discussion and argumentation between agents
3. **Voting** - Democratic decision-making process
4. **Additional phases** (to be implemented)

## Architecture

The multi-agent system is designed with:
- **State Management**: Robust state machine controlling legislative flow
- **Agent Coordination**: Structured communication between autonomous agents
- **Democratic Processes**: Simulation of real-world parliamentary procedures

## Project Structure

```
assemblyAgents/
├── agents/          # Agent implementations
├── states/          # State machine definitions
├── utils/           # Helper utilities
├── config/          # Configuration files
└── tests/           # Test suite
```

## Installation

### Prerequisites
- Python 3.8+
- Conda (Anaconda or Miniconda)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd assemblyAgents
```

2. Create and activate the conda environment:
```bash
conda create -n assembly-ai python=3.11
conda activate assembly-ai
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

*Coming soon*

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
This project follows PEP 8 style guidelines. Use `black` and `flake8` for code formatting and linting.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Built with ⚡ for simulating democratic processes through AI
