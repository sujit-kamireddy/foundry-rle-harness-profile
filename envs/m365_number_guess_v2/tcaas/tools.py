"""MOCK: server-side implementations of the user tools TCaaS publishes.

They live here, not in the gym, because they read task data the sandbox never
sees. Each takes the task's ``data`` plus the call arguments and returns the
string the policy reads back.
REAL SYSTEM: these become real user-tool endpoints; the gym-side proxy is
unaffected because it only forwards names and arguments.
"""

from __future__ import annotations

from typing import Any, Callable

USER_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "compare",
            "description": "Probe whether the target is higher, lower, or equal to a number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "Number to compare the target against.",
                    },
                },
                "required": ["number"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolRejected(ValueError):
    """Bad arguments. Surfaces to the policy as feedback, not as an outage."""


def compare(data: dict[str, Any], arguments: dict[str, Any]) -> str:
    """Report the target's position relative to the caller's number."""
    number = arguments.get("number")
    if not isinstance(number, int):
        raise ToolRejected("compare requires an integer 'number'.")
    target = data["target"]
    if target > number:
        return f"The target is higher than {number}."
    if target < number:
        return f"The target is lower than {number}."
    return f"The target is equal to {number}."


USER_TOOLS: dict[str, Callable[[dict[str, Any], dict[str, Any]], str]] = {
    "compare": compare,
}
