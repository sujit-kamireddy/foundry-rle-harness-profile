"""Offline stand-ins for this world's two rubrics.

The template ships four generic checks, all of which compare strings: cited ids,
expected ids, vocabulary, tool-was-called. This world scores something none of
them reach - *how efficiently* the answer was found - so it brings its own.

Loaded via `FT_WORLD_CHECKS=m365_dropin.world_checks`.

The contract is builtin types only::

    fn(params: dict, expected: dict, answer: str,
       calls: list[tuple[str, dict]]) -> float | (float, str)

`params` is the rubric's `check_params`, `expected` is the task's `data`,
`answer` is the terminal answer text, and `calls` is every tool call in order.
Return a score in [0, 1]; the optional second element is the reasoning string
that shows up in the trajectory.

REAL SYSTEM: this file disappears too. A real judge reads each rubric's
`criteria` prose, which is why the criteria in `catalog.json` are written as
questions to a grader rather than as notes to this module.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

Call = Tuple[str, Dict[str, Any]]


def _bounds(params: Dict[str, Any]) -> Tuple[int, int]:
    return int(params["min"]), int(params["max"])


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _submitted_number(answer: str) -> Optional[int]:
    """The number the policy committed to, or None if it never named one.

    The terminal tool takes free text, so the answer is parsed rather than
    assumed. `None` is a real outcome - a truncated episode reaches here with an
    empty answer - and it scores zero rather than defaulting to a lucky guess.
    """
    match = re.search(r"-?\d+", answer or "")
    return int(match.group()) if match else None


def _optimal_steps(low: int, high: int) -> int:
    """Binary-search probes plus the one commit that ends the episode."""
    return math.ceil(math.log2(high - low + 1)) + 1


def solve_efficiency(
    params: Dict[str, Any], expected: Dict[str, Any], answer: str, calls: List[Call]
) -> Tuple[float, str]:
    """Did it find the number, and how few steps did that take.

    Solving is worth a floor of 0.5 and efficiency pays the rest, so the reward
    separates *correct* from *correct and cheap* instead of collapsing both to
    1.0. A miss still earns partial credit for closeness, which keeps a gradient
    for a policy that has not yet solved anything.
    """
    low, high = _bounds(params)
    target = expected.get("target")
    if not isinstance(target, int):
        return 0.0, "This task carries no integer 'target'."

    submitted = _submitted_number(answer)
    if submitted is None:
        return 0.0, "The answer names no number."

    steps_used = len(calls)
    if submitted == target:
        optimal = _optimal_steps(low, high)
        efficiency = _clamp(optimal / steps_used) if steps_used > 0 else 0.0
        return (
            0.5 + 0.5 * efficiency,
            f"Correct in {steps_used} step(s); optimal is {optimal}.",
        )

    span = high - low
    closeness = 1.0 if span == 0 else _clamp(1.0 - abs(submitted - target) / span)
    return 0.5 * closeness, f"Answered {submitted}, target was {target}."


def probe_before_commit(
    params: Dict[str, Any], expected: Dict[str, Any], answer: str, calls: List[Call]
) -> Tuple[float, str]:
    """Did it narrow the range before committing.

    Scored on probe count rather than as a yes/no, because a policy that learns
    to probe once needs a signal telling it that probing four times is better.
    Only probes *before* the commit count - the point is that evidence preceded
    the decision.
    """
    terminal = str(params["terminal_tool"])
    target_probes = int(params["min_probes"])

    probes = 0
    for name, _ in calls:
        if name == terminal:
            break
        probes += 1

    if target_probes <= 0:
        return 1.0, "No probing required."
    return (
        _clamp(probes / target_probes),
        f"{probes} probe(s) before committing; {target_probes} expected.",
    )


CHECKS = {
    "solve_efficiency": solve_efficiency,
    "probe_before_commit": probe_before_commit,
}

REQUIRED_PARAMS = {
    "solve_efficiency": ("min", "max"),
    "probe_before_commit": ("terminal_tool", "min_probes"),
}
"""Params these checks cannot work without.

Declared, not defaulted, for the same reason the template refuses to guess an id
pattern: a check that silently falls back to the wrong bounds scores wrong on
every episode, and a reward that is quietly wrong looks exactly like a policy
that is quietly bad.
"""
