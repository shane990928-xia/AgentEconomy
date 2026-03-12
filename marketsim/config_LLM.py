"""
MarketSim LLM Configuration

Reads API keys and model settings from the project-level .env file,
sharing the same credentials as AgentEconomy.

Env vars used:
  DEEPSEEK_API_KEY   – API key (falls back to OPENAI_API_KEY)
  BASE_URL           – API base URL (falls back to https://api.deepseek.com)
  MARKETSIM_THINK_MODEL    – reasoning model (default: deepseek-reasoner)
  MARKETSIM_GENERATE_MODEL – generation model (default: deepseek-chat)
"""

import os
from pathlib import Path

# Load .env from project root (two levels up from marketsim/config_LLM.py)
# This is the same .env that AgentEconomy uses.
from dotenv import load_dotenv
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

root = "./"

apikeys = {}
# API key: prefer DEEPSEEK_API_KEY, fall back to OPENAI_API_KEY (they are the
# same key on the aggregation platform configured in .env)
apikeys['deepseek_api_key'] = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
apikeys['base_url'] = os.environ.get("BASE_URL", "https://api.deepseek.com")
apikeys['think_model_name'] = os.environ.get("MARKETSIM_THINK_MODEL", "deepseek-reasoner")
apikeys['generate_model_name'] = os.environ.get("MARKETSIM_GENERATE_MODEL", "deepseek-chat")

base_agentconfig = {}
base_agentconfig['company'] = "JNJ"
base_agentconfig['event'] = "Trump"

base_config = {
    'apikeys': apikeys,
    'root': root,
    'agent_config': base_agentconfig,
}
