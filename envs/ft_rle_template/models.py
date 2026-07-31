"""OpenEnv wire types for the FT gym.

The action is the tool-call envelope the harness profile advertises:
``{"tool_name": ..., "arguments": {...}}``. One envelope covers every tool, so
adding a user tool in TCaaS never changes this file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class FTAction(Action):
    """One tool call. ``tool_name`` must name a tool from the observation."""

    tool_name: str = Field(description="Tool to invoke for this step.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool."
    )


class FTObservation(Observation):
    """What the policy sees after reset and after every step.

    ``prompt`` and ``feedback`` are the two fields the harness profile points at
    (``promptPath`` / ``feedbackPath``). The rest is debugging surface that the
    default renderer may include but the policy does not depend on.
    """

    prompt: str = Field(description="Skill workflow plus the user query.")
    skill: str = Field(default="", description="Skill workflow text alone.")
    skill_id: str = Field(default="")
    user_query: str = Field(default="")
    tools: List[Dict[str, Any]] = Field(
        default_factory=list, description="OpenAI-format schemas for callable tools."
    )
    feedback: Optional[str] = Field(
        default=None, description="Result of the last tool call."
    )
    step: int = Field(default=0)
    last_tool: Optional[str] = Field(default=None)
    pending_effects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Writes this episode buffered instead of committing.",
    )
    """A declared field, not metadata, on purpose.

    OpenEnv's ``serialize_observation`` excludes ``metadata`` from the wire, so
    anything a caller must be able to audit has to be a real field. This is the
    record of what training *would* have written to the customer's tenant, so
    losing it silently is not acceptable.
    """


class FTState(State):
    """Episode state. ``episode_id`` and ``step_count`` come from the base."""

    done: bool = False
    task_id: Optional[str] = None
    split: Optional[str] = None
    last_tool: Optional[str] = None
    submitted: bool = False
    pending_effects: int = 0
