"""Scoring math for the number-guess task. Backs the ``efficient-solve`` rubric.

REAL SYSTEM: an LLM judge built from the rubric text replaces this; the mock
grader calls it so scores stay explainable while the plumbing is exercised.
"""

from __future__ import annotations

import math

from .logic import MAX_NUMBER, MIN_NUMBER

SOLVED_FLOOR = 0.5
"""A correct answer never scores below this; efficiency raises it toward 1.0.

A wrong answer scores 0.0, so correct always outranks wrong no matter how good
the process was. Without that, a policy could farm the process rubric by probing
and then answering wrong.
"""


def optimal_steps() -> int:
    """Ideal action count: binary-search probes plus one final commit."""
    range_size = MAX_NUMBER - MIN_NUMBER + 1
    return math.ceil(math.log2(range_size)) + 1


def efficiency(steps_used: int) -> float:
    """Step efficiency in [0, 1]; 1.0 at the optimal step count."""
    if steps_used <= 0:
        return 0.0
    return max(0.0, min(1.0, optimal_steps() / steps_used))


def grade(*, solved: bool, steps_used: int) -> float:
    """Score in [0, 1]: a solve floor plus an efficiency bonus, or zero if wrong."""
    if not solved:
        return 0.0
    return SOLVED_FLOOR + 0.5 * efficiency(steps_used)
