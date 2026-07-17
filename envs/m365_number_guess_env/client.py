"""Typed OpenEnv WebSocket client for the number-guessing environment.

The REST /reset and /step endpoints are single-shot (a fresh env per call).
Stateful multi-turn rollouts use this WebSocket client instead: one persistent
connection keeps a single server-side environment alive for the whole episode.
"""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from .models import NumberGuessAction, NumberGuessObservation, NumberGuessState
except ImportError:
    from models import NumberGuessAction, NumberGuessObservation, NumberGuessState


class NumberGuessClient(EnvClient[NumberGuessAction, NumberGuessObservation, NumberGuessState]):
    """Persistent-session client for the number-guessing environment."""

    def _step_payload(self, action: NumberGuessAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[NumberGuessObservation]:
        return StepResult(
            observation=NumberGuessObservation(**payload.get("observation", {})),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: dict[str, Any]) -> NumberGuessState:
        return NumberGuessState(**payload)
