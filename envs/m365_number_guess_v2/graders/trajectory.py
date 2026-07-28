"""Records the episode as OpenAI-style messages for the grader.

A tool call becomes an assistant message, its result the matching ``tool``
message. This is the only form trajectory rubrics can read.
REAL SYSTEM: unchanged - ``tc_graders`` expects exactly this under
``sample.output_trajectory``, and derives ``has_trajectory`` from its presence.
"""

from __future__ import annotations

import json
from typing import Any

from .models import GraderTrajectory, TrajectoryMessage


class TrajectoryRecorder:
    """Per-episode message log. Lives on the env instance, never module-global."""

    def __init__(self, trajectory_id: str, prompt: str) -> None:
        self._trajectory_id = trajectory_id
        self._messages: list[TrajectoryMessage] = [
            TrajectoryMessage(role="user", content=prompt)
        ]

    def record(
        self, tool_name: str, arguments: dict[str, Any], call_id: str, output: str
    ) -> None:
        """Append one call and its result, keeping ``tool_call_id`` matched up."""
        self._messages.append(
            TrajectoryMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            )
        )
        self._messages.append(
            TrajectoryMessage(role="tool", tool_call_id=call_id, content=output)
        )

    def build(self) -> GraderTrajectory:
        return GraderTrajectory(
            trajectory_id=self._trajectory_id, messages=list(self._messages)
        )


def read_tool_calls(trajectory: GraderTrajectory) -> list[tuple[str, dict[str, Any]]]:
    """Recover ``(tool_name, arguments)`` in order. Raises on a malformed log."""
    calls: list[tuple[str, dict[str, Any]]] = []
    for message in trajectory.messages:
        for call in message.tool_calls or []:
            function = call["function"]
            calls.append((function["name"], json.loads(function["arguments"])))
    return calls
