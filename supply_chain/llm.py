"""
Supply Chain LLM Bridge
=======================

Provides a unified synchronous `LLM` class for all supply-chain experiments
(network / network_continuous / supplier).  Reads credentials from the shared
project-level .env file — the same one used by AgentEconomy and MarketSim.

Usage (inside any stage script):
    from llm import LLM
    agent = LLM()
    response = agent.chat("Your prompt here")

Env vars read from .env:
    OPENAI_API_KEY   – API key (also accepted as DEEPSEEK_API_KEY fallback)
    BASE_URL         – API base URL  (default: https://api.deepseek.com)
    SUPPLY_CHAIN_MODEL  – model name (default: value of MODEL env var, then deepseek-chat)
"""

import os
import time
import json
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load the project-level .env (two levels up from supply_chain/)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# ── Import the OpenAI sync client ────────────────────────────────────────────
try:
    from openai import OpenAI as _OpenAI
except ImportError as e:
    raise ImportError(
        "openai package is required for supply_chain LLM. "
        "Install it with: pip install openai"
    ) from e


class LLM:
    """
    Synchronous LLM wrapper for supply-chain stage scripts.

    Mirrors the interface expected by stage3a/4/5 etc.:
        agent = LLM()
        text  = agent.chat(prompt)        # returns str
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: float = 2.0,
    ):
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY", "")
        )
        self.base_url = base_url or os.environ.get("BASE_URL", "https://api.deepseek.com")
        self.model = (
            model
            or os.environ.get("SUPPLY_CHAIN_MODEL")
            or os.environ.get("MODEL", "deepseek-chat")
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = _OpenAI(api_key=self.api_key, base_url=self.base_url)
        print(
            f"[LLM] Initialized — model={self.model}, "
            f"base_url={self.base_url}"
        )

    def chat(self, prompt: str, system: str = "You are a helpful assistant.") -> str:
        """
        Call the LLM synchronously with automatic retry.

        Args:
            prompt: User message.
            system: System message (optional, has a sensible default).

        Returns:
            Response content string.
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                print(
                    f"[LLM] Attempt {attempt}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries. "
            f"Last error: {last_error}"
        )

    def parse_json(self, response: str) -> Optional[dict]:
        """Parse JSON from LLM response (handles markdown fences)."""
        if not response:
            return None
        # Try markdown-fenced JSON
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        # Extract first {...}
        try:
            start, end = response.find("{"), response.rfind("}")
            if start != -1 and end != -1:
                return json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            pass
        return None

