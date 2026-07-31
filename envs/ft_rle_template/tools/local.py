"""Base tools that ship with the image, independent of any world.

Only one today: the terminal tool. It ships locally rather than coming from
TCaaS so every FT world ends an episode the same way, and so the harness
profile's ``terminalActions`` is a constant the renderer can rely on.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..logic import TERMINAL_TOOL
from ..tcaas.models import ToolSpec
from .base import ToolExecutionResult

SUBMIT_ANSWER_SPEC = ToolSpec(
    tool_name=TERMINAL_TOOL,
    description="Submit the final response to the user. Ends the episode.",
    effect="read",
    owner="local",
    input_schema={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final response to the user's request.",
            }
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
)


class LocalToolExecutor:
    """Runs base tools in-process. Recording an answer is not judging it."""

    def __init__(self) -> None:
        self._specs = {SUBMIT_ANSWER_SPEC.tool_name: SUBMIT_ANSWER_SPEC}

    def specs(self) -> List[ToolSpec]:
        return list(self._specs.values())

    def handles(self, tool_name: str) -> bool:
        return tool_name in self._specs

    def list_tool_schemas(self) -> List[Dict[str, Any]]:
        return [spec.to_openai_schema() for spec in self._specs.values()]

    def execute(
        self, tool_name: str, arguments: Dict[str, Any], *, call_id: str | None = None
    ) -> ToolExecutionResult:
        if tool_name != TERMINAL_TOOL:
            return ToolExecutionResult(
                tool_name=tool_name,
                output=f"Unknown tool {tool_name!r}.",
                success=False,
                call_id=call_id,
                error="unknown_tool",
            )
        answer = (arguments or {}).get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return ToolExecutionResult(
                tool_name=tool_name,
                output="answer is required and must be a non-empty string.",
                success=False,
                call_id=call_id,
                error="invalid_argument",
            )
        return ToolExecutionResult(
            tool_name=tool_name,
            output="Answer submitted.",
            call_id=call_id,
        )
