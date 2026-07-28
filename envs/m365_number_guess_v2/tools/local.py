"""Base tools built into the sandbox itself, run in-process.

These exist for every deployment regardless of user content: ``guess`` is how a
policy commits an answer. It records the answer and nothing more - the sandbox
never learns the target, so correctness is the grader's call.
REAL SYSTEM: this is the gym image's own tool surface; user tools arrive
separately from TCaaS (see ``proxy.py``).
"""

from __future__ import annotations

from typing import Any

from .base import ToolExecutionResult

GUESS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "guess",
        "description": "Commit a final answer for the hidden number. Ends the episode.",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {"type": "integer", "description": "The number to commit."},
            },
            "required": ["number"],
            "additionalProperties": False,
        },
    },
}


class LocalToolExecutor:
    """Runs the sandbox's built-in tools in-process, with no network hop."""

    def __init__(self, min_number: int, max_number: int) -> None:
        self._min = min_number
        self._max = max_number

    def list_tool_schemas(self) -> list[dict[str, Any]]:
        return [GUESS_SCHEMA]

    def execute(
        self, tool_name: str, arguments: dict[str, Any], *, call_id: str | None = None
    ) -> ToolExecutionResult:
        if tool_name != "guess":
            return ToolExecutionResult(
                tool_name=tool_name,
                output=f"Unknown tool {tool_name!r}.",
                success=False,
                call_id=call_id,
                error="unknown_tool",
            )
        return self._guess(arguments.get("number"), call_id)

    def _guess(self, number: Any, call_id: str | None) -> ToolExecutionResult:
        """Validate and record a committed answer; grading decides if it is right."""
        if not isinstance(number, int) or not self._min <= number <= self._max:
            return ToolExecutionResult(
                tool_name="guess",
                output=f"Provide an integer between {self._min} and {self._max}.",
                success=False,
                call_id=call_id,
                error="invalid_argument",
            )
        return ToolExecutionResult(
            tool_name="guess",
            output=f"Answer submitted: {number}.",
            call_id=call_id,
        )
