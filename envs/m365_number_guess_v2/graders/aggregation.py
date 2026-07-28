"""Combines per-rubric scores into the one number that becomes reward.

Mirrors ``tc_graders``' ``AggregationConfig`` strategies and its default
``pass_threshold`` of 0.5, so grader-passed and harness-success agree.
REAL SYSTEM: tc_graders performs the aggregation; this exists so the mock
produces the same shape of answer.
"""

from __future__ import annotations

PASS_THRESHOLD = 0.5


def aggregate(scores: list[float], weights: list[float], strategy: str) -> float:
    """Reduce per-rubric scores to one. Unknown strategies are a programming error."""
    if not scores:
        raise ValueError("no rubric scores to aggregate")
    if strategy == "weighted_mean":
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        return sum(s * w for s, w in zip(scores, weights)) / total
    if strategy == "mean":
        return sum(scores) / len(scores)
    if strategy == "min":
        return min(scores)
    if strategy == "max":
        return max(scores)
    raise ValueError(f"unknown aggregation strategy {strategy!r}")
