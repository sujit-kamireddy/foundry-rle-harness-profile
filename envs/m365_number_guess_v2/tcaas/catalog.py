"""In-memory TCaaS catalog: one tenant's skills, rubrics, and tasks by split.

``pick_task`` is the whole point - it turns ``(split, seed)`` into a task with no
RNG, so a GRPO group that reuses one seed lands on the same sample every time.
REAL SYSTEM: this JSON file becomes TCaaS-backed storage; the lookup contract
(seed in, task bundle out) is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .identity import TenantContext
from .models import Rubric, Skill, Task, TaskBundle

CATALOG_PATH = Path(__file__).with_name("catalog.json")


class Catalog:
    """Loaded catalog with skill -> rubric and split -> task indexes."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.tenant = TenantContext(**raw["tenant"])
        self.skills: dict[str, Skill] = {
            s["skill_id"]: Skill(**s) for s in raw["skills"]
        }
        self.rubrics: list[Rubric] = [Rubric(**r) for r in raw["rubrics"]]
        self.tasks: dict[str, list[Task]] = {
            split: [Task(**t) for t in items] for split, items in raw["tasks"].items()
        }
        self._validate()

    @classmethod
    def load(cls, path: Path | None = None) -> "Catalog":
        """Read the catalog from disk (defaults to the bundled ``catalog.json``)."""
        return cls(json.loads((path or CATALOG_PATH).read_text()))

    def _validate(self) -> None:
        """Fail at load time on dangling references, never mid-episode."""
        for rubric in self.rubrics:
            if rubric.skill_id not in self.skills:
                raise ValueError(
                    f"rubric {rubric.rubric_id} cites unknown skill {rubric.skill_id}"
                )
        for split, tasks in self.tasks.items():
            if not tasks:
                raise ValueError(f"split {split!r} is empty")
            for task in tasks:
                if task.skill_id not in self.skills:
                    raise ValueError(
                        f"task {task.task_id} cites unknown skill {task.skill_id}"
                    )

    def rubrics_for(self, skill_id: str) -> list[Rubric]:
        """The rubric set owned by one skill. N skills each keep their own."""
        return [r for r in self.rubrics if r.skill_id == skill_id]

    def split_size(self, split: str) -> int:
        """Number of distinct tasks in a split; ``evalDefaults.limit`` mirrors this."""
        return len(self._split(split))

    def pick_task(self, split: str, seed: int) -> TaskBundle:
        """Resolve ``(split, seed)`` to a full bundle, deterministically.

        Seeds wrap around the split, making the gym an infinite sampler: seeds
        ``0..len(split)-1`` cover it exactly once, and anything beyond repeats.
        """
        tasks = self._split(split)
        task = tasks[seed % len(tasks)]
        skill = self.skills[task.skill_id]
        return TaskBundle(
            task_id=task.task_id,
            split=split,
            skill_id=skill.skill_id,
            skill=skill.workflow,
            user_query=task.user_query,
            data=task.data,
            rubrics=self.rubrics_for(skill.skill_id),
        )

    def get_task(self, task_id: str) -> Task:
        """Look a task up by id - used by the tool endpoints that own task data."""
        for tasks in self.tasks.values():
            for task in tasks:
                if task.task_id == task_id:
                    return task
        raise KeyError(task_id)

    def _split(self, split: str) -> list[Task]:
        if split not in self.tasks:
            raise KeyError(f"unknown split {split!r}; have {sorted(self.tasks)}")
        return self.tasks[split]
