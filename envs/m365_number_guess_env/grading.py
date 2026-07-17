"""Dedicated, transport-agnostic grader. ``grade`` is the single source of truth
for scoring an episode and is called once, when the episode ends."""

from __future__ import annotations

import math

try:
    from .logic import MAX_NUMBER, MIN_NUMBER
except ImportError:
    from logic import MAX_NUMBER, MIN_NUMBER


def distance_score(number: int, target: int) -> float:
    """Closeness to the target in [0, 1]; 1.0 for an exact match."""
    span = MAX_NUMBER - MIN_NUMBER
    if span == 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - abs(number - target) / span))


def optimal_steps() -> int:
    """Ideal action count: binary-search probes plus one final commit."""
    range_size = MAX_NUMBER - MIN_NUMBER + 1
    return math.ceil(math.log2(range_size)) + 1


def efficiency(steps_used: int) -> float:
    """Step efficiency in [0, 1]; 1.0 at the optimal step count."""
    if steps_used <= 0:
        return 0.0
    return max(0.0, min(1.0, optimal_steps() / steps_used))


def grade(*, solved: bool, steps_used: int, number: int, target: int) -> tuple[float, str]:
    """Terminal (reward, feedback) in [0, 1]: solve floor plus efficiency bonus,
    or partial credit for closeness when the episode ends unsolved."""
    if solved:
        reward = 0.5 + 0.5 * efficiency(steps_used)
        return reward, f"Correct! Solved in {steps_used} step(s)."
    reward = 0.5 * distance_score(number, target)
    return reward, f"Out of steps. The number was {target}."
