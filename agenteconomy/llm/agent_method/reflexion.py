from __future__ import annotations

from .base import (
    ModelCaller,
    ModelType,
    compose_system_prompt,
    indicates_correctness,
    join_sections,
)


async def run(
    call_model: ModelCaller,
    prompt: str,
    system_prompt: str,
    model_type: ModelType,
) -> str:
    initial_answer = await call_model(
        prompt,
        compose_system_prompt(
            system_prompt,
            "Produce your best initial answer to the user's request.",
            "Do not mention any internal review process.",
        ),
        model_type,
    )

    reflection = await call_model(
        join_sections(
            ("Original user request", prompt),
            ("Current draft", initial_answer),
        ),
        compose_system_prompt(
            system_prompt,
            "You are reviewing your own draft before sending it.",
            "Identify concrete issues, missing constraints, or reasons the draft is already correct.",
            "If the draft is already ready to send, say so clearly.",
        ),
        model_type,
    )

    if indicates_correctness(reflection):
        return initial_answer

    return await call_model(
        join_sections(
            ("Original user request", prompt),
            ("Current draft", initial_answer),
            ("Reflection", reflection),
        ),
        compose_system_prompt(
            system_prompt,
            "Revise the draft using the reflection.",
            "Return only the improved final response to the user's request.",
        ),
        model_type,
    )
