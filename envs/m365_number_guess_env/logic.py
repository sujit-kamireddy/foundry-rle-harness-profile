from __future__ import annotations

import random
from typing import Optional


MIN_NUMBER = 1
MAX_NUMBER = 10
MAX_STEPS = 10

TASK_PROMPT = (
    "Guess the hidden number between "
    f"{MIN_NUMBER} and {MAX_NUMBER}. Use compare to narrow the range, then guess to commit."
)
INSTRUCTIONS = (
    "Call compare(number) to learn whether the target is higher, lower, or equal, "
    "then call guess(number) to commit. Solve it in as few steps as possible."
)


def select_target(seed: Optional[int] = None) -> int:
    """Pick the hidden target, deterministically when a seed is given."""
    return random.Random(seed).randint(MIN_NUMBER, MAX_NUMBER)


def compare_result(number: int, target: int) -> str:
    """Return 'higher', 'lower', or 'equal' for the target relative to number."""
    if target > number:
        return "higher"
    if target < number:
        return "lower"
    return "equal"
