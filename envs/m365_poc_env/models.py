from __future__ import annotations

from typing import Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


class M365HelloWorldAction(Action):
    """Action submitted by the model for the M365 hello-world task."""

    message: str = Field(default="", description="Greeting message to submit.")


class M365HelloWorldObservation(Observation):
    """Observation returned by the M365 hello-world environment."""

    problem: str = Field(description="Task prompt shown to the model.")
    instructions: str = Field(description="Success criteria for the task.")
    feedback: Optional[str] = Field(default=None, description="Feedback from the last action.")
    step: int = Field(default=0, ge=0, description="Current step count.")
    last_message: Optional[str] = Field(default=None, description="Last submitted message, if any.")


class M365HelloWorldState(State):
    """State for the M365 hello-world environment."""

    done: bool = Field(default=False, description="Whether the episode is complete.")
    last_message: Optional[str] = Field(default=None, description="Last submitted message, if any.")