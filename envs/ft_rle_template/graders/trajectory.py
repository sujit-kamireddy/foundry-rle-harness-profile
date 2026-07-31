"""Records an episode as OpenAI-style messages for the grader.

Trajectory rubrics grade *how* the answer was reached, so the tool calls have to
survive in the shape a judge can read.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


class TrajectoryRecorder:
    """One per episode. Append-only."""

    def __init__(self, episode_id: str, prompt: str) -> None:
        self.episode_id = episode_id
        self._messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]

    def record(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        call_id: str,
        output: str,
        *,
        success: bool = True,
    ) -> None:
        self._messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments or {}, sort_keys=True),
                        },
                    }
                ],
            }
        )
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": output,
                "metadata": {"success": success},
            }
        )

    def build(self) -> List[Dict[str, Any]]:
        return list(self._messages)

    def tool_calls(self) -> List[str]:
        names: List[str] = []
        for message in self._messages:
            for call in message.get("tool_calls") or []:
                names.append(call["function"]["name"])
        return names
