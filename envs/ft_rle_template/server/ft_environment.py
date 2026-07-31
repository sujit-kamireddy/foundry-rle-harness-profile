"""The gym: a generic FT sandbox wired to TCaaS for content and user tools.

Reset resolves ``(split, seed)`` into a task bundle and assembles the callable
tool surface; step routes one tool call and grades once the episode ends.
REAL SYSTEM: unchanged - both clients are injected, so pointing them at deployed
services is configuration, not a code change.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..config import (
    MAX_STEPS_PER_EPISODE,
    SUCCESS_THRESHOLD,
    VIRTUALIZATION_BASE_URL,
    training_mode,
)
from ..graders.client import GraderClient, to_reward
from ..graders.models import GradeRequest, Item, RubricRef, Sample
from ..graders.trajectory import TrajectoryRecorder
from ..logic import DEFAULT_SPLIT, ENV_DESCRIPTION, TERMINAL_TOOL, compose_prompt
from ..models import FTAction, FTObservation, FTState
from ..tcaas.client import TCaaSClient
from ..tcaas.models import TaskBundle
from ..tools.base import ToolExecutionResult
from ..tools.local import LocalToolExecutor
from ..tools.proxy import TCaaSToolExecutor
from ..tools.registry import ToolRegistry
from ..tools.virtualization import make_containment


class FTEnvironment(Environment[FTAction, FTObservation, FTState]):
    """Multi-turn tool-use environment whose content comes from TCaaS."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        tcaas: TCaaSClient | None = None,
        graders: GraderClient | None = None,
        training: bool | None = None,
    ) -> None:
        super().__init__()
        self._tcaas = tcaas or TCaaSClient()
        self._graders = graders or GraderClient()
        self._training = training_mode() if training is None else training
        self._bundle: TaskBundle | None = None
        self._tools: ToolRegistry | None = None
        self._recorder: TrajectoryRecorder | None = None
        self._state = FTState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs: Any,
    ) -> FTObservation:
        """Resolve the task for this seed and assemble the tool surface.

        ``split`` is optional because the harness is not documented to send it,
        so we fall back to the profile's declared default. A TCaaS failure here
        is fatal: a fallback task would train on the wrong data.

        The write-ahead buffer is created fresh per episode, which is what keeps
        one rollout's buffered writes out of the next one.
        """
        bundle = self._tcaas.pick_task(split or DEFAULT_SPLIT, seed or 0)
        episode = episode_id or str(uuid4())
        self._bundle = bundle
        self._tools = ToolRegistry(
            base=LocalToolExecutor(),
            user=TCaaSToolExecutor(
                self._tcaas,
                bundle.task_id,
                bundle.tools,
                training=self._training,
                buffer=make_containment(episode, VIRTUALIZATION_BASE_URL),
            ),
        )
        self._recorder = TrajectoryRecorder(
            episode, compose_prompt(bundle.skill, bundle.user_query)
        )
        self._state = FTState(
            episode_id=episode,
            step_count=0,
            done=False,
            task_id=bundle.task_id,
            split=bundle.split,
            last_tool=None,
            submitted=False,
            pending_effects=0,
        )
        return self._observation(feedback=None, reward=0.0, done=False)

    def step(
        self,
        action: FTAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> FTObservation:
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
            action.tool_name,
            action.arguments,
            call_id,
            result.output,
            success=result.success,
        )
        self._state.pending_effects = len(self._tools.pending_effects)

        submitted = self._submitted_answer(action, result)
        exhausted = self._state.step_count >= MAX_STEPS_PER_EPISODE

        if submitted is None and not exhausted:
            return self._observation(feedback=result.output, reward=0.0, done=False)

        self._state.submitted = submitted is not None
        self._state.done = True
        reward, individual = self._grade(submitted or "")
        observation = self._observation(
            feedback=result.output, reward=reward, done=True
        )
        observation.metadata["individual_results"] = individual
        observation.metadata["success"] = reward >= SUCCESS_THRESHOLD
        observation.metadata["truncated"] = submitted is None and exhausted
        return observation

    def _grade(self, answer: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Grade the finished episode. Fires exactly once, from the final step.

        ``item.expected`` comes from the task bundle: the caller supplies the
        reference, and the sandbox itself never consults it.
        """
        if self._bundle is None or self._recorder is None:
            raise RuntimeError("no active episode")
        bundle = self._bundle
        request = GradeRequest(
            rubrics=[
                RubricRef(
                    rubric_id=r.rubric_id,
                    weight=r.weight,
                    range=r.range,
                    criteria=r.criteria,
                    outcome=r.outcome,
                    check=r.check,
                    check_params=r.check_params,
                )
                for r in bundle.rubrics
            ],
            aggregation="weighted_mean",
            item=Item(
                input=compose_prompt(bundle.skill, bundle.user_query),
                expected=json.dumps(bundle.data, sort_keys=True),
            ),
            sample=Sample(
                output_text=answer,
                output_trajectory=self._recorder.build(),
            ),
        )
        response = self._graders.grade(request)
        return to_reward(response), response.extra_outputs.get("individual_results", [])

    def _submitted_answer(
        self, action: FTAction, result: ToolExecutionResult
    ) -> str | None:
        """An accepted terminal call ends the episode; a rejected one is feedback."""
        if action.tool_name != TERMINAL_TOOL or not result.success:
            return None
        answer = (action.arguments or {}).get("answer")
        return answer if isinstance(answer, str) else None

    @property
    def state(self) -> FTState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="ft_rle_template",
            description=ENV_DESCRIPTION,
            version="0.1.0",
            author="Microsoft Frontier Tuning",
        )

    def _observation(
        self, feedback: Optional[str], reward: float, done: bool
    ) -> FTObservation:
        if self._bundle is None or self._tools is None:
            raise RuntimeError("no active episode")
        bundle = self._bundle
        metadata: Dict[str, Any] = {
            "episode_id": self._state.episode_id,
            "task_id": bundle.task_id,
            "skill_id": bundle.skill_id,
            "split": bundle.split,
            "tool_mode": "training" if self._training else "inference",
        }
        return FTObservation(
            prompt=compose_prompt(bundle.skill, bundle.user_query),
            skill=bundle.skill,
            skill_id=bundle.skill_id,
            user_query=bundle.user_query,
            tools=self._tools.list_tool_schemas(),
            feedback=feedback,
            step=self._state.step_count,
            last_tool=self._state.last_tool,
            pending_effects=self._tools.pending_effects,
            done=done,
            reward=reward,
            metadata=metadata,
        )
