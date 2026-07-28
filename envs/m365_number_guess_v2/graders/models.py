"""Wire types for the grading call, mirroring ``tc_graders``.

The contract that matters: ``item`` is what the model was given, ``sample`` is
what it produced, and the *caller* supplies the reference in ``item.expected`` -
the grader never looks anything up.
REAL SYSTEM: the real ``GraderJob`` carries an instantiated MultiGrader config
where we send a rubric list; ``item``/``sample`` stay byte-compatible.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class TrajectoryMessage(BaseModel):
    """One OpenAI-style message: user prompt, assistant tool call, or tool result."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class GraderTrajectory(BaseModel):
    """The episode as messages. Its presence is what makes trajectory rubrics gradable."""

    trajectory_id: str
    messages: list[TrajectoryMessage] = Field(default_factory=list)


class Item(BaseModel):
    """What the model was given, plus the reference answer for this task."""

    input: str = Field(description="The exact prompt the model saw.")
    expected: Optional[str] = Field(
        default=None, description="Reference answer, caller-supplied."
    )


class Sample(BaseModel):
    """What the model produced."""

    output_text: Optional[str] = Field(
        default=None, description="The terminal action, verbatim."
    )
    output_trajectory: Optional[GraderTrajectory] = None


class RubricRef(BaseModel):
    """A rubric to apply, with the weight it carries in the aggregate."""

    rubric_id: str
    weight: float = 1.0
    range: tuple[float, float] = (0.0, 1.0)


class GradeRequest(BaseModel):
    """One grading job: which rubrics, how to combine them, and the item/sample pair."""

    rubrics: list[RubricRef]
    aggregation: str = Field(
        default="weighted_mean", description="mean|weighted_mean|min|max."
    )
    item: Item
    sample: Sample


class RubricResult(BaseModel):
    """One rubric's verdict, surfaced so a multi-rubric setup stays debuggable."""

    rubric_id: str
    score: float
    passed: bool


class GradeResponse(BaseModel):
    """``score`` is the aggregate, or ``None`` when grading itself failed.

    A ``None`` score is an infrastructure fault, never a policy score of zero -
    the caller must raise rather than reward on it (§5.3).
    """

    score: Optional[float] = None
    passed: bool = False
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    extra_outputs: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def failure(cls, error: str) -> "GradeResponse":
        """Grading could not be performed. Never fabricate a score instead."""
        return cls(score=None, passed=False, error=error)
