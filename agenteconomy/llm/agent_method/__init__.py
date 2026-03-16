from __future__ import annotations

from .cot import run as run_cot
from .debate import run as run_debate
from .discussion import run as run_discussion
from .reflexion import run as run_reflexion
from .self_refine import run as run_self_refine


AGENT_METHOD_REGISTRY = {
    "cot": run_cot,
    "debate": run_debate,
    "discussion": run_discussion,
    "reflexion": run_reflexion,
    "self_refine": run_self_refine,
}


def normalize_agent_name(agent_name: str | None) -> str | None:
    if not isinstance(agent_name, str):
        return None
    normalized = agent_name.strip().lower().replace("-", "_")
    return normalized or None


def get_agent_method(agent_name: str | None):
    normalized = normalize_agent_name(agent_name)
    if normalized is None:
        return None
    return AGENT_METHOD_REGISTRY.get(normalized)
