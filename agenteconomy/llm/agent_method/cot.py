from __future__ import annotations

from .base import ModelCaller, ModelType, compose_system_prompt


async def run(
    call_model: ModelCaller,
    prompt: str,
    system_prompt: str,
    model_type: ModelType,
) -> str:
    cot_system_prompt = compose_system_prompt(
        system_prompt,
        "Think carefully about the user's request before answering.",
        "Keep your private reasoning internal unless the user explicitly asks for it.",
        "Return the best final answer and preserve any format constraints from the user.",
    )
    return await call_model(prompt, cot_system_prompt, model_type)
