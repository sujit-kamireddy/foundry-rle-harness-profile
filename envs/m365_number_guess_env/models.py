from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class NumberGuessAction(Action):
    """Action submitted by the model for the number-guessing task."""

    action_type: Literal["compare", "guess"] = Field(
        default="guess",
        description="'compare' to probe the range, 'guess' to commit a final answer.",
    )
    number: int = Field(default=0, description="The number to compare against or to guess.")


class NumberGuessObservation(Observation):
    """Observation returned by the number-guessing environment."""

    problem: str = Field(description="Task prompt shown to the model.")
    instructions: str = Field(description="How to use the compare and guess tools.")
    feedback: Optional[str] = Field(default=None, description="Feedback from the last action.")
    step: int = Field(default=0, ge=0, description="Number of actions taken so far.")
    last_action: Optional[str] = Field(default=None, description="Last action type taken, if any.")
    last_number: Optional[int] = Field(default=None, description="Last number submitted, if any.")


class NumberGuessState(State):
    """State for the number-guessing environment."""

    done: bool = Field(default=False, description="Whether the episode is complete.")
    solved: bool = Field(default=False, description="Whether the target has been guessed.")
    last_action: Optional[str] = Field(default=None, description="Last action type taken, if any.")
    last_number: Optional[int] = Field(default=None, description="Last number submitted, if any.")
