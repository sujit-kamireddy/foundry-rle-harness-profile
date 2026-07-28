"""Sandbox constants and prompt composition. No tool logic, no network.

Tool behaviour lives in ``tools/`` (base) and TCaaS (user tools); the hidden
number is never here - the grader owns correctness.
"""

from __future__ import annotations

MIN_NUMBER = 1
MAX_NUMBER = 10
MAX_STEPS = 10

ENV_DESCRIPTION = (
    "Number-guessing gym backed by a TCaaS task catalog and a tc_graders grader."
)
INSTRUCTIONS = (
    "Call compare(number) to learn whether the target is higher, lower, or equal, "
    "then call guess(number) to commit. Solve it in as few steps as possible."
)


def compose_prompt(skill: str, user_query: str) -> str:
    """Join the skill workflow and the user query into the one rendered prompt.

    The harness surfaces a single observation field (``promptPath``), so the gym
    does the stitching.
    REAL SYSTEM: once the harness supports observation-field placeholders this
    becomes a ``prompt_template`` in the profile and this function goes away.
    """
    return f"{skill}\n\nUser: {user_query}"
