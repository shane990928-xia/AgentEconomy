from __future__ import annotations

from .base import ModelCaller, ModelType, compose_system_prompt, join_sections


EXPERT_FOCUS_AREAS = (
    "correctness and completeness",
    "practicality and clarity",
    "risks, edge cases, and hidden assumptions",
)


async def run(
    call_model: ModelCaller,
    prompt: str,
    system_prompt: str,
    model_type: ModelType,
) -> str:
    proposals: list[str] = []
    for expert_id, focus in enumerate(EXPERT_FOCUS_AREAS, start=1):
        expert_answer = await call_model(
            prompt,
            compose_system_prompt(
                system_prompt,
                f"You are expert #{expert_id} in a panel discussion.",
                f"Focus on {focus}.",
                "Prepare the best answer you can, but do not mention the panel discussion in the response.",
            ),
            model_type,
        )
        proposals.append(expert_answer)

    moderator_prompt = join_sections(
        ("Original user request", prompt),
        ("Expert #1 proposal", proposals[0]),
        ("Expert #2 proposal", proposals[1]),
        ("Expert #3 proposal", proposals[2]),
    )
    return await call_model(
        moderator_prompt,
        compose_system_prompt(
            system_prompt,
            "You are the moderator of an internal expert discussion.",
            "Synthesize the expert proposals into one final answer to the original user request.",
            "Preserve any formatting constraints from the user and do not mention the discussion unless asked.",
        ),
        model_type,
    )
