"""World-agnostic episode helpers.

Everything here is pure so the environment stays trivially testable.
"""

from __future__ import annotations

ENV_DESCRIPTION = (
    "Frontier Tuning world running as a Foundry RLE. Task content, rubrics, and "
    "user tools are served by TCaaS; grading is served by tc_graders."
)

TERMINAL_TOOL = "submit_answer"
"""The one tool that ends an episode. Ships with the image, never from TCaaS,
so every FT world terminates the same way."""

DEFAULT_SPLIT = "train"

MODEL_INSTRUCTIONS = (
    "Respond with exactly one raw JSON action object and no prose, markdown, or "
    "code fences. Each action names a tool and its arguments:\n"
    '{"tool_name":"<tool>","arguments":{...}}\n'
    "Call the listed tools to gather what you need, then call "
    f'{{"tool_name":"{TERMINAL_TOOL}","arguments":{{"answer":"..."}}}} '
    "to submit your final response. Submitting ends the episode. The prompt "
    "field states the task and the workflow you should follow."
)
"""STATIC protocol guidance. Goes in the harness profile, not the observation.

Per-skill workflow text is *episode content* and belongs in the prompt, because
it changes per task while this text never does.
"""


def compose_prompt(skill_workflow: str, user_query: str) -> str:
    """Build the model-visible prompt.

    The harness renders one observation field, so the gym composes the prompt
    itself and points ``promptPath`` at it.
    """
    workflow = (skill_workflow or "").strip()
    query = (user_query or "").strip()
    if not workflow:
        return f"User: {query}"
    return f"{workflow}\n\nUser: {query}"
