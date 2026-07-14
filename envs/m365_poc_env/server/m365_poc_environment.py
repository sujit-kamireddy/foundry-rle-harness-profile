from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata

    from ..logic import SUCCESS_HINT, TASK_PROMPT, evaluate_message
    from ..models import M365HelloWorldAction, M365HelloWorldObservation, M365HelloWorldState
except ImportError:
    from logic import SUCCESS_HINT, TASK_PROMPT, evaluate_message
    from models import M365HelloWorldAction, M365HelloWorldObservation, M365HelloWorldState
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata


class M365HelloWorldEnvironment(Environment[M365HelloWorldAction, M365HelloWorldObservation, M365HelloWorldState]):
    """Small OpenEnv environment for M365 + Foundry RLE collaboration demos."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = M365HelloWorldState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> M365HelloWorldObservation:
        self._reset_rubric()
        self._state = M365HelloWorldState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            done=False,
            last_message=None,
        )
        return self._observation(feedback=None, reward=0.0, done=False)

    def step(
        self,
        action: M365HelloWorldAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> M365HelloWorldObservation:
        success, feedback = evaluate_message(action.message)
        self._state.step_count += 1
        self._state.done = True
        self._state.last_message = action.message
        return self._observation(
            feedback=feedback,
            reward=1.0 if success else 0.0,
            done=True,
            success=success,
        )

    @property
    def state(self) -> M365HelloWorldState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="m365_poc_env",
            description=TASK_PROMPT,
            version="0.1.0",
            author="Microsoft Foundry RLE",
        )

    def _observation(
        self,
        feedback: Optional[str],
        reward: float,
        done: bool,
        success: Optional[bool] = None,
    ) -> M365HelloWorldObservation:
        metadata = {
            "expected": SUCCESS_HINT,
            "episode_id": self._state.episode_id,
        }
        if success is not None:
            metadata["success"] = success
        return M365HelloWorldObservation(
            problem=TASK_PROMPT,
            instructions=SUCCESS_HINT,
            feedback=feedback,
            step=self._state.step_count,
            last_message=self._state.last_message,
            done=done,
            reward=reward,
            metadata=metadata,
        )