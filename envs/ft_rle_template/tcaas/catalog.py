"""In-memory stand-in for TCaaS storage, loaded from ``catalog.json``.

``pick_task`` is a pure lookup with no RNG, so the same seed always yields the
same task. GRPO samples K trajectories from *one* task by reusing a seed across
the group, so that property holds by construction rather than by convention.
REAL SYSTEM: back this with TCaaS storage; ``pick_task(split, seed)`` is the only
signature the gym depends on and it does not change.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from .models import Rubric, Skill, Task, TaskBundle, ToolSpec, WorldDescriptor

CATALOG_PATH = Path(__file__).with_name("catalog.json")
"""The world M365 drops in. Overridable with ``FT_CATALOG_PATH`` so tests and
validation runs can point at another world without editing the image."""


def catalog_path() -> Path:
    return Path(os.getenv("FT_CATALOG_PATH") or CATALOG_PATH)


class Catalog:
    """Resolves a ``(split, seed)`` into everything one episode needs."""

    def __init__(self, raw: Dict) -> None:
        self._world_id = raw["world_id"]
        self._name = raw.get("name", "")
        self._description = raw.get("description", "")
        self._content_version = str(raw.get("content_version", "0"))
        self._skills = {s["skill_id"]: Skill(**s) for s in raw.get("skills", [])}
        self._rubrics = [Rubric(**r) for r in raw.get("rubrics", [])]
        self._tools = [ToolSpec(**t) for t in raw.get("tools", [])]
        self._tasks: Dict[str, List[Task]] = {
            split: [Task(**t) for t in items]
            for split, items in raw.get("tasks", {}).items()
        }
        self._datasets: Dict[str, List[Dict]] = {
            name: [dict(r) for r in rows]
            for name, rows in raw.get("datasets", {}).items()
        }
        self._validate()

    def _validate(self) -> None:
        """Fail at load, not mid-episode.

        A dropped-in world that references a missing skill or dataset would
        otherwise surface as a confusing runtime error inside a rollout, or - far
        worse - as a rubric that silently scores zero forever.
        """
        from ..graders.rubrics import unscorable_reason
        from ..tools.local import LocalToolExecutor

        problems: List[str] = []
        for rubric in self._rubrics:
            if rubric.skill_id not in self._skills:
                problems.append(
                    f"rubric {rubric.rubric_id} cites unknown skill {rubric.skill_id}"
                )
            if rubric.check is not None:
                reason = unscorable_reason(rubric.check, rubric.check_params)
                if reason is not None:
                    problems.append(f"rubric {rubric.rubric_id} {reason}")
        reserved = {spec.tool_name for spec in LocalToolExecutor().specs()}
        for tool in self._tools:
            if tool.tool_name in reserved:
                problems.append(
                    f"tool {tool.tool_name} collides with a base tool that ships with "
                    "the image; the base tool wins, so the world's tool would never "
                    "run and calling it would end the episode"
                )
        for split, items in self._tasks.items():
            for task in items:
                if task.skill_id not in self._skills:
                    problems.append(
                        f"task {task.task_id} in {split} cites unknown skill "
                        f"{task.skill_id}"
                    )
        for tool in self._tools:
            binding = tool.serves
            if binding is None:
                continue
            if binding.dataset not in self._datasets:
                problems.append(
                    f"tool {tool.tool_name} serves unknown dataset {binding.dataset}"
                )
            for ref in binding.references:
                if ref.get("dataset") not in self._datasets:
                    problems.append(
                        f"tool {tool.tool_name} references unknown dataset "
                        f"{ref.get('dataset')}"
                    )
        if problems:
            raise ValueError("invalid catalog: " + "; ".join(problems))

    def datasets(self) -> Dict[str, List[Dict]]:
        return self._datasets

    @property
    def splits(self) -> List[str]:
        return sorted(self._tasks)

    @property
    def world_id(self) -> str:
        return self._world_id

    def split_sizes(self) -> Dict[str, int]:
        return {split: len(items) for split, items in self._tasks.items()}

    def tools(self) -> List[ToolSpec]:
        return list(self._tools)

    def rubrics_for(self, skill_id: str) -> List[Rubric]:
        return [r for r in self._rubrics if r.skill_id == skill_id]

    def descriptor(self) -> WorldDescriptor:
        return WorldDescriptor(
            world_id=self._world_id,
            name=self._name,
            description=self._description,
            content_version=self._content_version,
            skills=list(self._skills.values()),
            rubrics=list(self._rubrics),
            tools=list(self._tools),
            split_sizes=self.split_sizes(),
        )

    def pick_task(self, split: str, seed: int) -> TaskBundle:
        """Pure lookup: ``tasks[split][seed % len(split)]``."""
        items = self._tasks.get(split)
        if not items:
            raise KeyError(f"unknown or empty split: {split!r}")
        task = items[seed % len(items)]
        skill = self._skills.get(task.skill_id)
        if skill is None:
            raise KeyError(f"task {task.task_id} references unknown skill {task.skill_id}")
        return TaskBundle(
            task_id=task.task_id,
            skill_id=task.skill_id,
            split=split,
            skill=skill.workflow,
            user_query=task.user_query,
            data=task.data,
            rubrics=self.rubrics_for(task.skill_id),
            tools=list(self._tools),
        )

    def find_task(self, task_id: str) -> Task:
        for items in self._tasks.values():
            for task in items:
                if task.task_id == task_id:
                    return task
        raise KeyError(f"unknown task: {task_id!r}")


@lru_cache(maxsize=4)
def _load(path: str) -> Catalog:
    return Catalog(json.loads(Path(path).read_text(encoding="utf-8")))


def load_catalog() -> Catalog:
    """Cached per path, so switching worlds is a path change, not a restart."""
    return _load(str(catalog_path().resolve()))
