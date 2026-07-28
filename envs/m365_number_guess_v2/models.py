from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class NumberGuessAction(Action):
    """One tool call from the policy.

    ``tool_name`` is a plain string, not a Literal: the callable surface is
    assembled at reset from base plus TCaaS tools, so the sandbox cannot
    enumerate it up front. The profile's ``actionSpace`` pins the names for a
    given deployment.
    """

    tool_name: str = Field(default="guess", description="Name of the tool to call.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments."
    )


class NumberGuessObservation(Observation):
    """What the policy sees. ``prompt`` is the only field the harness renders.

    Tool guidance deliberately lives in exactly two places: the *protocol* (JSON
    action shape) in the profile's ``observationRendering.instructions``, and the
    *workflow* in the skill text, which arrives inside ``prompt``.
    """

    prompt: str = Field(description="Composed skill workflow plus user query.")
    skill: str = Field(description="Skill workflow text, kept separate for debugging.")
    user_query: str = Field(description="The user's request for this task.")
    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="OpenAI-format schemas for the callable tools.",
    )
    feedback: Optional[str] = Field(
        default=None, description="Result of the last tool call."
    )
    step: int = Field(default=0, ge=0, description="Number of tool calls made so far.")
    last_tool: Optional[str] = Field(
        default=None, description="Last tool called, if any."
    )


class NumberGuessState(State):
    """Per-instance episode state. Never module-global: sessions run concurrently."""

    done: bool = Field(default=False, description="Whether the episode is complete.")
    task_id: Optional[str] = Field(
        default=None, description="TCaaS task driving this episode."
    )
    split: Optional[str] = Field(
        default=None, description="Split the task was drawn from."
    )
    last_tool: Optional[str] = Field(
        default=None, description="Last tool called, if any."
    )
    committed_answer: Optional[int] = Field(
        default=None, description="Answer committed via the terminal tool, if any."
    )
