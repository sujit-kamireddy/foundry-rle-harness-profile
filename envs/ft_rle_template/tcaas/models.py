"""The FT world data model as TCaaS serves it.

One vocabulary shared by the gym, the mock service, and the profile renderer, so
a world shape change lands in exactly one file.

A skill owns its rubrics (``rubric.skill_id``), never the reverse - real rubrics
cite the skill's own text. Resolution is one hop each way, so the gym makes a
single TCaaS call per episode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Rubric(BaseModel):
    """One scored criterion. Generated from a skill, so it carries ``skill_id``."""

    rubric_id: str
    skill_id: str
    title: str = ""
    criteria: str = ""
    weight: float = 1.0
    range: List[float] = Field(default_factory=lambda: [0.0, 1.0])

    outcome: bool = Field(
        default=False,
        description="True if this rubric decides whether the task was actually done.",
    )
    """Declared per rubric rather than hardcoded by id.

    Outcome rubrics gate the reward: if all of them score zero the episode scores
    zero overall, so process rubrics cannot pay out for a wrong answer. A world
    that declares none gets no gate, which the grader reports rather than hides.
    """

    check: Optional[str] = Field(
        default=None,
        description="Generic scorer the offline mock uses. Ignored by a real judge.",
    )
    check_params: Dict[str, Any] = Field(default_factory=dict)


class ToolBinding(BaseModel):
    """How the offline mock serves one tool from catalog data.

    This exists so adding a tool is a data change, not a code change. A real MCP
    endpoint ignores all of it - the gym-side proxy never reads this field.
    """

    dataset: str = Field(description="Key under the catalog's `datasets` map.")
    entity_type: str = Field(default="", description="Reported back to the caller.")

    required: List[str] = Field(
        default_factory=list,
        description="Arguments that must be supplied, else the call is rejected.",
    )
    not_found_is_error: bool = Field(
        default=False,
        description="True for get-one tools: an empty match is a rejection.",
    )
    """Separates two things that are easy to conflate.

    ``required`` validates the *call*. ``not_found_is_error`` describes the
    *result*: a ``get_x`` with no match is a bad argument the policy should learn
    from, while a ``list_x`` with no match is a legitimate empty answer. Getting
    this wrong breaks the read-after-write overlay, because a list tool that
    rejects never reaches the overlay that would have surfaced the buffered write.
    """
    filters: List[str] = Field(
        default_factory=list,
        description="Optional arguments that narrow the result when supplied.",
    )
    where: Dict[str, Any] = Field(
        default_factory=dict, description="Constant predicate applied to every read."
    )

    appends: bool = Field(
        default=False, description="True for a write tool: append a record."
    )
    id_field: Optional[str] = Field(
        default=None, description="Field to populate with a generated id on append."
    )
    id_prefix: str = Field(default="rec")
    references: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Referential checks, each {field, dataset, key}.",
    )


class ToolSpec(BaseModel):
    """A callable tool, described once and reused by the gym and the renderer.

    ``effect`` is what makes dual mode possible: a ``write`` tool is buffered
    during training instead of reaching the backend. ``owner`` decides whether
    the gym runs it in-process or proxies it.
    """

    tool_name: str
    description: str = ""
    input_schema: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    effect: str = Field(default="read", description="read or write")
    owner: str = Field(default="tcaas", description="local or tcaas")

    produces_entity: Optional[str] = Field(
        default=None,
        description="Entity type a write tool creates, used by the read overlay.",
    )
    overlay_entities: List[str] = Field(
        default_factory=list,
        description="Entity types a read tool lists, used by the read overlay.",
    )

    serves: Optional[ToolBinding] = Field(
        default=None,
        description="Offline-mock data binding. Absent means the mock cannot run it.",
    )

    def is_write(self) -> bool:
        return self.effect.strip().lower() == "write"

    def to_openai_schema(self) -> Dict[str, Any]:
        """OpenAI-format function schema, the shape the policy sees."""
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class Skill(BaseModel):
    """An FT skill: routing label, full workflow prompt, and knowledge refs."""

    skill_id: str
    name: str = ""
    description: str = Field(default="", description="Short picker/routing label.")
    workflow: str = Field(default="", description="Full instructions the agent follows.")
    knowledge: List[Dict[str, Any]] = Field(default_factory=list)


class Task(BaseModel):
    """One FT sample. ``data`` is grading context, never gameplay state."""

    task_id: str
    skill_id: str
    user_query: str
    data: Dict[str, Any] = Field(default_factory=dict)


class TaskBundle(BaseModel):
    """Everything one episode needs, resolved by a single ``pick_task`` call."""

    task_id: str
    skill_id: str
    split: str
    skill: str = Field(description="Workflow text the policy reads.")
    user_query: str
    data: Dict[str, Any] = Field(default_factory=dict)
    rubrics: List[Rubric] = Field(default_factory=list)
    tools: List[ToolSpec] = Field(default_factory=list)


class WorldDescriptor(BaseModel):
    """World-level metadata the profile renderer needs.

    This is what the FT backend reads to render ``harness-profile.json`` before
    the container exists.
    """

    world_id: str
    name: str = ""
    description: str = ""
    content_version: str = Field(
        default="0",
        description="Pinned per RLE version so replay and champion/challenger hold.",
    )
    skills: List[Skill] = Field(default_factory=list)
    rubrics: List[Rubric] = Field(default_factory=list)
    tools: List[ToolSpec] = Field(default_factory=list)
    split_sizes: Dict[str, int] = Field(default_factory=dict)
