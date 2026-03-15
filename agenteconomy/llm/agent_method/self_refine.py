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
            "Generate an initial answer to the user's request.",
            "Keep your private reasoning internal unless the user explicitly asks for it.",
        ),
        model_type,
    )

    feedback = await call_model(
        join_sections(
            ("Original user request", prompt),
            ("Initial answer", initial_answer),
        ),
        compose_system_prompt(
            system_prompt,
            "Audit the initial answer.",
            "State whether it is correct and list the most important fixes if needed.",
            "If no fixes are needed, say so clearly.",
        ),
        model_type,
    )

    if indicates_correctness(feedback):
        return initial_answer

    return await call_model(
        join_sections(
            ("Original user request", prompt),
            ("Initial answer", initial_answer),
            ("Audit feedback", feedback),
        ),
        compose_system_prompt(
            system_prompt,
            "Produce a refined final answer using the audit feedback.",
            "Return only the final response that should be shown to the user.",
        ),
        model_type,
    )
