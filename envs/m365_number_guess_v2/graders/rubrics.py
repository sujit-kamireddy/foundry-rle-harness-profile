"""Rubric scorers: ``rubric_id`` -> callable, one per rubric in the catalog.

The two shipped here differ in kind on purpose - one reads the outcome, one reads
the trajectory - which is what makes aggregation worth testing.
REAL SYSTEM: one LLM judge scores every rubric by reading its ``criteria``, so
adding a rubric is a content change. Lacking a judge, each rubric needs a
hand-written stand-in here - that registry is a mock artifact, not a design.
"""

from __future__ import annotations

from typing import Callable

from ..grading import grade
from .models import Item, Sample
from .trajectory import read_tool_calls

TERMINAL_TOOL = "guess"


class Unscorable(ValueError):
    """The sample cannot be read. Becomes a failure response, never a zero."""


def _committed_answer(sample: Sample) -> tuple[int | None, int]:
    """Recover ``(answer, steps_used)``; answer is ``None`` if nothing was committed.

    Never committing is a *policy* outcome, so it scores low rather than raising.
    ``Unscorable`` is reserved for a structurally broken sample.
    """
    if sample.output_trajectory is None:
        raise Unscorable("sample has no output_trajectory")
    try:
        calls = read_tool_calls(sample.output_trajectory)
    except (KeyError, ValueError) as exc:
        raise Unscorable(f"malformed trajectory: {exc}") from exc
    for index, (tool_name, arguments) in enumerate(calls, start=1):
        if tool_name == TERMINAL_TOOL and isinstance(arguments.get("number"), int):
            return arguments["number"], index
    return None, len(calls)


def efficient_solve(item: Item, sample: Sample) -> float:
    """Outcome rubric: did it land on the answer, and in how few steps?"""
    if item.expected is None:
        raise Unscorable("item.expected is required to score correctness")
    answer, steps_used = _committed_answer(sample)
    if answer is None:
        return 0.0
    return grade(solved=answer == int(item.expected), steps_used=steps_used)


def probe_before_commit(item: Item, sample: Sample) -> float:
    """Trajectory rubric: did it gather evidence before committing?"""
    if sample.output_trajectory is None:
        raise Unscorable("sample has no output_trajectory")
    calls = read_tool_calls(sample.output_trajectory)
    probes = sum(1 for name, _ in calls if name != TERMINAL_TOOL)
    if probes == 0:
        return 0.0
    return 1.0


SCORERS: dict[str, Callable[[Item, Sample], float]] = {
    "efficient-solve": efficient_solve,
    "probe-before-commit": probe_before_commit,
}
