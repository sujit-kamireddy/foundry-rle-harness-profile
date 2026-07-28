"""The vocabulary every tool backend shares, local or remote.

Mirrors ``tc_tools``' ``ToolExecutor`` protocol, trimmed to the essentials and
kept synchronous because OpenEnv's ``step`` is synchronous.
REAL SYSTEM: adopt ``tc_tools.ToolExecutor`` directly; the split below between a
rejected call and a transport failure is the part worth keeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class ToolTransportError(RuntimeError):
    """The tool could not be reached or answered. Never a policy signal.

    ``tc_tools`` folds this into ``success=False`` so inference can retry; for RL
    that would launder an outage into a reward, so we raise instead (§5.3).
    """


@dataclass(frozen=True)
class ToolExecutionResult:
    """Outcome of one tool call, identical in shape for local and remote tools.

    ``success=False`` means the call was *rejected* (bad arguments) - that is
    legitimate feedback and the episode continues.
    """

    tool_name: str
    output: str
    success: bool = True
    call_id: str | None = None
    error: str | None = None


@runtime_checkable
class ToolExecutor(Protocol):
    """Anything that can list tool schemas and run a call by name."""

    def list_tool_schemas(self) -> list[dict[str, Any]]:
        """OpenAI-format function schemas this backend serves."""
        ...

    def execute(
        self, tool_name: str, arguments: dict[str, Any], *, call_id: str | None = None
    ) -> ToolExecutionResult:
        """Run one call. Raises ``ToolTransportError`` if the backend is unusable."""
        ...
