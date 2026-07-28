"""Routes a tool call to whichever backend owns it.

This is the sandbox's whole tool story: base tools ship with the image, user
tools are layered on from TCaaS, and everything downstream - feedback,
trajectory, grading - sees one result type and never learns which ran.
REAL SYSTEM: unchanged; only the backends behind it are swapped.
"""

from __future__ import annotations

from typing import Any

from .base import ToolExecutionResult, ToolExecutor


class ToolRegistry:
    """Name -> executor, with base tools registered first and user tools layered on."""

    def __init__(self, base: ToolExecutor, user: ToolExecutor | None = None) -> None:
        self._backends: list[ToolExecutor] = [base] if user is None else [base, user]
        self._routes: dict[str, ToolExecutor] = {}
        for backend in self._backends:
            for schema in backend.list_tool_schemas():
                self._routes.setdefault(schema["function"]["name"], backend)

    @property
    def tool_names(self) -> list[str]:
        return list(self._routes)

    def list_tool_schemas(self) -> list[dict[str, Any]]:
        """Merged schemas; base tools win a name collision."""
        seen: dict[str, dict[str, Any]] = {}
        for backend in self._backends:
            for schema in backend.list_tool_schemas():
                seen.setdefault(schema["function"]["name"], schema)
        return list(seen.values())

    def execute(
        self, tool_name: str, arguments: dict[str, Any], *, call_id: str | None = None
    ) -> ToolExecutionResult:
        """Dispatch by name. An unknown tool is a rejected call, not a crash."""
        backend = self._routes.get(tool_name)
        if backend is None:
            return ToolExecutionResult(
                tool_name=tool_name,
                output=f"Unknown tool {tool_name!r}. Available: {', '.join(self._routes)}.",
                success=False,
                call_id=call_id,
                error="unknown_tool",
            )
        return backend.execute(tool_name, arguments, call_id=call_id)
