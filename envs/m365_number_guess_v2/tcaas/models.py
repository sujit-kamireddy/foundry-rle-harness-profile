"""Wire types for the TCaaS catalog: skills, rubrics, tasks, and task bundles.

Deliberately free of any OpenEnv import so the catalog can be exercised without
the gym runtime.
REAL SYSTEM: these mirror TCaaS resources; a real client would import them from
the TCaaS SDK instead of redefining them here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Skill(BaseModel):
    """A workflow: how to use the available tools to satisfy a kind of request."""

    skill_id: str
    name: str
    workflow: str = Field(description="Guidance text shown to the policy.")


class Rubric(BaseModel):
    """One scoring criterion, generated from - and owned by - a single skill.

    ``skill_id`` points at its parent because real rubrics cite the skill's text
    (``QualityRubric.cites``); they are never shared across skills.
    """

    rubric_id: str
    skill_id: str
    rubric_type: str = Field(
        description="user_facing | trajectory_non_tool | trajectory_tool."
    )
    criteria: str
    weight: float = 1.0
    range: tuple[float, float] = (0.0, 1.0)


class Task(BaseModel):
    """One training/eval sample: a user query plus the data needed to grade it."""

    task_id: str
    skill_id: str
    user_query: str
    data: dict[str, Any] = Field(default_factory=dict)


class TaskBundle(BaseModel):
    """Everything the gym needs for one episode, resolved in a single call.

    Rubrics are inlined rather than left as ids so the later grade request is
    self-describing and the gym makes exactly one TCaaS call per episode.
    """

    task_id: str
    split: str
    skill_id: str
    skill: str = Field(description="The skill's workflow text.")
    user_query: str
    data: dict[str, Any] = Field(default_factory=dict)
    rubrics: list[Rubric] = Field(default_factory=list)
