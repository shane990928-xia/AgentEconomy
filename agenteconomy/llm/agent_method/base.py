from __future__ import annotations

from typing import Awaitable, Callable, Literal


ModelType = Literal["simple", "strong"]
ModelCaller = Callable[[str, str, ModelType], Awaitable[str]]


def join_blocks(*blocks: str) -> str:
    parts = [block.strip() for block in blocks if isinstance(block, str) and block.strip()]
    return "\n\n".join(parts)


def join_sections(*sections: tuple[str, str]) -> str:
    blocks = []
    for title, content in sections:
        if isinstance(content, str) and content.strip():
            blocks.append(f"{title}:\n{content.strip()}")
    return "\n\n".join(blocks)


def compose_system_prompt(base_prompt: str, *extras: str) -> str:
    return join_blocks(base_prompt, *extras)


def indicates_correctness(feedback: str) -> bool:
    text = (feedback or "").strip().lower()
    positive_signals = (
        "correct",
        "no issues",
        "looks good",
        "satisfies the request",
        "ready to send",
    )
    return any(signal in text for signal in positive_signals)
