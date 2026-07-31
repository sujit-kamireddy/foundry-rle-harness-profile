"""The tc_graders wire contract.

``item`` is what the model was given, ``sample`` is what it produced, and the
*caller* supplies the reference in ``item.expected`` - the grader never looks
anything up. That keeps task data on the TCaaS side of the boundary and the
grader stateless.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RubricRef(BaseModel):
    rubric_id: str
    weight: float = 1.0
    range: List[float] = Field(default_factory=lambda: [0.0, 1.0])
    criteria: str = ""

    outcome: bool = Field(
        default=False, description="Whether this rubric gates the reward."
    )
    check: Optional[str] = Field(
        default=None, description="Generic scorer name for the offline mock."
    )
    check_params: Dict[str, Any] = Field(default_factory=dict)


class Item(BaseModel):
    """What the model was given."""

    input: str
    expected: str = Field(
        default="", description="Reference supplied by the caller, JSON-encoded."
    )


class Sample(BaseModel):
    """What the model produced."""

    output_text: str = ""
    output_trajectory: List[Dict[str, Any]] = Field(default_factory=list)


class GradeRequest(BaseModel):
    rubrics: List[RubricRef]
    aggregation: str = "weighted_mean"
    item: Item
    sample: Sample


class RubricResult(BaseModel):
    rubric_id: str
    score: float
    reasoning: str = ""


class GradeResponse(BaseModel):
    score: Optional[float] = Field(
        default=None,
        description="None means the grader could not score. Never treat as 0.0.",
    )
    extra_outputs: Dict[str, Any] = Field(default_factory=dict)
