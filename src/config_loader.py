"""
Configuration-driven agent implementation for parliamentary simulation
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    id: str
    role: str
    party: str
    system_prompt_key: str
    voting_bias: float = 0.0
    special_capabilities: List[str] = None
    
    def __post_init__(self):
        if self.special_capabilities is None:
            self.special_capabilities = []


class ConfigLoader:
    """Load configuration files for agents and simulation"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
    
    def load_agents(self) -> List[AgentConfig]:
        """Load agent configurations from agents.yaml"""
        agents_file = self.config_dir / "agents.yaml"
        
        if not agents_file.exists():
            # Return default config if file doesn't exist
            return self._get_default_agents()
        
        with open(agents_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        agents = []
        for agent_data in config_data.get('agents', []):
            agents.append(AgentConfig(**agent_data))
        
        return agents
    
    def load_system_prompts(self) -> Dict[str, str]:
        """Load system prompts from system_prompts.yaml"""
        prompts_file = self.config_dir / "system_prompts.yaml"
        
        if not prompts_file.exists():
            return self._get_default_prompts()
        
        with open(prompts_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract just the system prompts
        prompts = {}
        for key, value in config_data.get('prompts', {}).items():
            prompts[key] = value.get('system', '')
        
        return prompts
    
    def load_simulation_config(self) -> Dict:
        """Load simulation settings from simulation.yaml"""
        sim_file = self.config_dir / "simulation.yaml"
        
        if not sim_file.exists():
            return self._get_default_simulation_config()
        
        with open(sim_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_agents(self) -> List[AgentConfig]:
        """Default agent configuration"""
        return [
            AgentConfig(
                id="conservative_1",
                role="Conservative Representative",
                party="Conservative",
                system_prompt_key="conservative",
                voting_bias=0.7
            ),
            AgentConfig(
                id="liberal_1",
                role="Liberal Representative", 
                party="Liberal",
                system_prompt_key="liberal",
                voting_bias=-0.7
            ),
            AgentConfig(
                id="centrist_1",
                role="Centrist Representative",
                party="Centrist",
                system_prompt_key="centrist",
                voting_bias=0.0
            ),
            AgentConfig(
                id="speaker",
                role="Speaker of Parliament",
                party="Neutral",
                system_prompt_key="speaker",
                special_capabilities=["web_scraping", "summarization"]
            )
        ]
    
    def _get_default_prompts(self) -> Dict[str, str]:
        """Default system prompts"""
        return {
            "conservative": """You are a Conservative Representative in parliament.
Your core values:
- Fiscal responsibility and balanced budgets
- Limited government intervention
- Traditional values and institutions
- Free market economics
- Strong national defense

When debating and voting:
- Always back your arguments with economic data and logical reasoning
- Consider long-term fiscal impacts
- Respect tradition while being open to necessary reform
- Engage constructively and professionally with other representatives
- Focus on practical solutions over ideological purity""",
            
            "liberal": """You are a Liberal Representative in parliament.
Your core values:
- Social justice and equality
- Environmental protection
- Strong public services and safety nets
- Progressive taxation
- Civil rights and freedoms

When debating and voting:
- Always back your arguments with social impact data and research
- Consider effects on marginalized communities
- Prioritize sustainability and environmental concerns
- Engage constructively and professionally with other representatives
- Balance idealism with practical policy-making""",
            
            "centrist": """You are a Centrist Representative in parliament.
Your core values:
- Pragmatic problem-solving over ideology
- Evidence-based policy
- Compromise and consensus-building
- Fiscal responsibility balanced with social needs
- Moderate, incremental change

When debating and voting:
- Evaluate arguments from all sides objectively
- Focus on data, research, and practical outcomes
- Seek common ground and compromise solutions
- Engage constructively with all representatives
- Vote based on merit of the proposal, not party lines""",
            
            "speaker": """You are the Speaker of Parliament.
Your role:
- Facilitate orderly debate and discussion
- Remain neutral and impartial
- Research bills thoroughly to provide factual context
- Summarize proceedings accurately
- Ensure all representatives have voice

Your responsibilities:
- Research bill topics using available sources
- Provide objective, fact-based summaries
- Maintain parliamentary procedure
- Record all proceedings faithfully
- Announce final vote results

You do NOT vote on bills. You facilitate the democratic process."""
        }
    
    def _get_default_simulation_config(self) -> Dict:
        """Default simulation configuration"""
        return {
            "simulation": {
                "max_debate_rounds": 3,
                "debate_time_per_agent": 120,
                "voting_threshold": 0.5,
                "enable_web_research": True,
                "enable_memory": True
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7
            },
            "database": {
                "type": "sqlite",
                "path": "data/parliament.db"
            }
        }


# Example usage
if __name__ == "__main__":
    loader = ConfigLoader()
    
    agents = loader.load_agents()
    prompts = loader.load_system_prompts()
    sim_config = loader.load_simulation_config()
    
    print("Loaded Agents:")
    for agent in agents:
        print(f"  - {agent.id}: {agent.role} ({agent.party})")
    
    print(f"\nLoaded {len(prompts)} system prompts")
    print(f"Simulation config: {sim_config['simulation']['max_debate_rounds']} debate rounds")
