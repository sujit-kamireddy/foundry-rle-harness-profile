from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata

    from ..logic import (
        INSTRUCTIONS,
        MAX_STEPS,
        TASK_PROMPT,
        compare_result,
        select_target,
    )
    from ..grading import distance_score, grade
    from ..models import NumberGuessAction, NumberGuessObservation, NumberGuessState
except ImportError:
    from logic import (
        INSTRUCTIONS,
        MAX_STEPS,
        TASK_PROMPT,
        compare_result,
        select_target,
    )
    from grading import distance_score, grade
    from models import NumberGuessAction, NumberGuessObservation, NumberGuessState
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata


class NumberGuessEnvironment(Environment[NumberGuessAction, NumberGuessObservation, NumberGuessState]):
    """Multi-turn number-guessing OpenEnv environment that rewards efficient strategy."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._target = select_target()
        self._state = NumberGuessState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> NumberGuessObservation:
        self._target = select_target(seed)
        self._state = NumberGuessState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            done=False,
            solved=False,
            last_action=None,
            last_number=None,
        )
        return self._observation(feedback=None, reward=0.0, done=False)

    def step(
        self,
        action: NumberGuessAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> NumberGuessObservation:
        self._state.step_count += 1
        self._state.last_action = action.action_type
        self._state.last_number = action.number

        solved = action.action_type == "guess" and action.number == self._target
        terminal = solved or self._state.step_count >= MAX_STEPS

        if not terminal:
            if action.action_type == "compare":
                feedback = self._compare_feedback(action.number)
            else:
                feedback = "Incorrect. Use compare to localize the target, then guess again."
            return self._observation(feedback=feedback, reward=0.0, done=False)

        self._state.done = True
        self._state.solved = solved
        reward, feedback = grade(
            solved=solved,
            steps_used=self._state.step_count,
            number=action.number,
            target=self._target,
        )
        return self._observation(feedback=feedback, reward=reward, done=True, success=solved)

    def _compare_feedback(self, number: int) -> str:
        result = compare_result(number, self._target)
        if result == "equal":
            return f"The target is equal to {number}."
        return f"The target is {result} than {number}."

    @property
    def state(self) -> NumberGuessState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="m365_number_guess_env",
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
    ) -> NumberGuessObservation:
        metadata: dict[str, Any] = {"episode_id": self._state.episode_id}
        if self._state.last_number is not None:
            metadata["distance_score"] = distance_score(self._state.last_number, self._target)
        if success is not None:
            metadata["success"] = success
        return NumberGuessObservation(
            problem=TASK_PROMPT,
            instructions=INSTRUCTIONS,
            feedback=feedback,
            step=self._state.step_count,
            last_action=self._state.last_action,
            last_number=self._state.last_number,
            done=done,
            reward=reward,
            metadata=metadata,
        )
