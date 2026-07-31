"""Rubric aggregation.

Kept separate so a new strategy is a function, not a branch inside the grader.
"""

from __future__ import annotations

from typing import Dict, List

from .models import RubricRef, RubricResult


def weighted_mean(results: List[RubricResult], rubrics: List[RubricRef]) -> float:
    weights: Dict[str, float] = {r.rubric_id: r.weight for r in rubrics}
    total = sum(weights.get(r.rubric_id, 1.0) for r in results)
    if total <= 0:
        return 0.0
    return sum(r.score * weights.get(r.rubric_id, 1.0) for r in results) / total


def mean(results: List[RubricResult], rubrics: List[RubricRef]) -> float:
    return sum(r.score for r in results) / len(results) if results else 0.0


def minimum(results: List[RubricResult], rubrics: List[RubricRef]) -> float:
    return min((r.score for r in results), default=0.0)


STRATEGIES = {
    "weighted_mean": weighted_mean,
    "mean": mean,
    "min": minimum,
}


def aggregate(
    name: str, results: List[RubricResult], rubrics: List[RubricRef]
) -> float:
    return STRATEGIES.get(name, weighted_mean)(results, rubrics)
