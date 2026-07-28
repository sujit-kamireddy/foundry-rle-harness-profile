"""The gym: a generic sandbox wired to TCaaS for task content and user tools.

Reset resolves ``(split, seed)`` into a task bundle and assembles the callable
tool surface; step routes one tool call and grades once the episode ends.
REAL SYSTEM: unchanged - both clients are injected, so pointing them at deployed
services is configuration, not a code change.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..graders.client import GraderClient, to_reward
from ..graders.models import GradeRequest, Item, RubricRef, Sample
from ..graders.trajectory import TrajectoryRecorder
from ..logic import (
    ENV_DESCRIPTION,
    INSTRUCTIONS,
    MAX_NUMBER,
    MAX_STEPS,
    MIN_NUMBER,
    compose_prompt,
)
from ..models import NumberGuessAction, NumberGuessObservation, NumberGuessState
from ..tcaas.client import TCaaSClient
from ..tcaas.models import TaskBundle
from ..tools.base import ToolExecutionResult
from ..tools.local import LocalToolExecutor
from ..tools.proxy import TCaaSToolExecutor
from ..tools.registry import ToolRegistry

TERMINAL_TOOL = "guess"
DEFAULT_SPLIT = "train"
SUCCESS_THRESHOLD = 0.5
"""Matches the profile's reward thresholds and tc_graders' default pass mark, so
grader-passed and harness-success agree by construction."""


class NumberGuessEnvironment(
    Environment[NumberGuessAction, NumberGuessObservation, NumberGuessState]
):
    """Multi-turn tool-use environment whose task content comes from TCaaS."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self, tcaas: TCaaSClient | None = None, graders: GraderClient | None = None
    ) -> None:
        super().__init__()
        self._tcaas = tcaas or TCaaSClient()
        self._graders = graders or GraderClient()
        self._bundle: TaskBundle | None = None
        self._tools: ToolRegistry | None = None
        self._recorder: TrajectoryRecorder | None = None
        self._state = NumberGuessState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs: Any,
    ) -> NumberGuessObservation:
        """Resolve the task for this seed and assemble the tool surface.

        ``split`` is optional because the harness is not documented to send it,
        so we fall back to the profile's declared default. A TCaaS failure here
        is fatal: a fallback task would train on the wrong data (§5.3).
        """
        bundle = self._tcaas.pick_task(split or DEFAULT_SPLIT, seed or 0)
        self._bundle = bundle
        self._tools = ToolRegistry(
            base=LocalToolExecutor(MIN_NUMBER, MAX_NUMBER),
            user=TCaaSToolExecutor(self._tcaas, bundle.task_id),
        )
        episode = episode_id or str(uuid4())
        self._recorder = TrajectoryRecorder(
            episode, compose_prompt(bundle.skill, bundle.user_query)
        )
        self._state = NumberGuessState(
            episode_id=episode,
            step_count=0,
            done=False,
            task_id=bundle.task_id,
            split=bundle.split,
            last_tool=None,
            committed_answer=None,
        )
        return self._observation(feedback=None, reward=0.0, done=False)

    def step(
        self,
        action: NumberGuessAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> NumberGuessObservation:
        """Route one tool call, then grade once if this step ends the episode."""
        if self._bundle is None or self._tools is None or self._recorder is None:
            raise RuntimeError("step() called before reset()")
        if self._state.done:
            raise RuntimeError("step() called after the episode ended")

        self._state.step_count += 1
        self._state.last_tool = action.tool_name
        call_id = f"call-{self._state.step_count}"
        result = self._tools.execute(
            action.tool_name, action.arguments, call_id=call_id
        )
        self._recorder.record(
            action.tool_name, action.arguments, call_id, result.output
        )

        committed = self._committed_answer(action, result)
        if committed is not None:
            self._state.committed_answer = committed

        if committed is None and self._state.step_count < MAX_STEPS:
            return self._observation(feedback=result.output, reward=0.0, done=False)

        self._state.done = True
        reward, individual = self._grade(action)
        observation = self._observation(
            feedback=result.output, reward=reward, done=True
        )
        observation.metadata["individual_results"] = individual
        observation.metadata["success"] = reward >= SUCCESS_THRESHOLD
        return observation

    def _grade(self, action: NumberGuessAction) -> tuple[float, list[dict[str, Any]]]:
        """Grade the finished episode. Fires exactly once, from the terminal step.

        ``item.expected`` comes from the task bundle: the caller supplies the
        reference, and the sandbox itself never consults it.
        """
        if self._bundle is None or self._recorder is None:
            raise RuntimeError("no active episode")
        bundle = self._bundle
        request = GradeRequest(
            rubrics=[
                RubricRef(rubric_id=r.rubric_id, weight=r.weight, range=r.range)
                for r in bundle.rubrics
            ],
            aggregation="weighted_mean",
            item=Item(
                input=compose_prompt(bundle.skill, bundle.user_query),
                expected=str(bundle.data.get("target")),
            ),
            sample=Sample(
                output_text=action.model_dump_json(include={"tool_name", "arguments"}),
                output_trajectory=self._recorder.build(),
            ),
        )
        response = self._graders.grade(request)
        return to_reward(response), response.extra_outputs.get("individual_results", [])

    def _committed_answer(
        self, action: NumberGuessAction, result: ToolExecutionResult
    ) -> int | None:
        """An accepted terminal-tool call ends the episode; a rejected one is feedback."""
        if action.tool_name != TERMINAL_TOOL or not result.success:
            return None
        number = action.arguments.get("number")
        return number if isinstance(number, int) else None

    @property
    def state(self) -> NumberGuessState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="m365_number_guess_v2",
            description=ENV_DESCRIPTION,
            version="0.2.0",
            author="Microsoft Foundry RLE",
        )

    def _observation(
        self, feedback: Optional[str], reward: float, done: bool
    ) -> NumberGuessObservation:
        if self._bundle is None or self._tools is None:
            raise RuntimeError("no active episode")
        bundle = self._bundle
        metadata: dict[str, Any] = {
            "episode_id": self._state.episode_id,
            "task_id": bundle.task_id,
            "skill_id": bundle.skill_id,
            "split": bundle.split,
        }
        return NumberGuessObservation(
            prompt=compose_prompt(bundle.skill, bundle.user_query),
            skill=bundle.skill,
            user_query=bundle.user_query,
            instructions=INSTRUCTIONS,
            tools=self._tools.list_tool_schemas(),
            feedback=feedback,
            step=self._state.step_count,
            last_tool=self._state.last_tool,
            done=done,
            reward=reward,
            metadata=metadata,
        )
