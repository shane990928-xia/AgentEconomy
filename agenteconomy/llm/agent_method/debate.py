from __future__ import annotations

from .base import ModelCaller, ModelType, compose_system_prompt, join_sections


async def run(
    call_model: ModelCaller,
    prompt: str,
    system_prompt: str,
    model_type: ModelType,
) -> str:
    pro_answer = await call_model(
        prompt,
        compose_system_prompt(
            system_prompt,
            "You are the proposer in an internal debate.",
            "Draft the strongest possible answer to the user's request.",
            "Do not mention the internal debate in your response.",
        ),
        model_type,
    )

    con_answer = await call_model(
        join_sections(
            ("Original user request", prompt),
            ("Draft answer to critique", pro_answer),
        ),
        compose_system_prompt(
            system_prompt,
            "You are the critic in an internal debate.",
            "Find factual errors, missing constraints, edge cases, or formatting problems in the draft answer.",
            "If the draft is already strong, explain what should be preserved.",
        ),
        model_type,
    )

    return await call_model(
        join_sections(
            ("Original user request", prompt),
            ("Proposed draft", pro_answer),
            ("Critical review", con_answer),
        ),
        compose_system_prompt(
            system_prompt,
            "You are the final decision maker.",
            "Resolve the internal debate and produce the single best final response to the original user request.",
            "Follow the user's requested format exactly and do not mention the internal debate unless asked.",
        ),
        model_type,
    )
