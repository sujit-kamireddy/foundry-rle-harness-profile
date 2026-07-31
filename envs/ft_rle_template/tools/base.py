"""The vocabulary every tool backend shares, local or remote.

Mirrors ``tc_tools``' ``ToolExecutor`` protocol, trimmed to the essentials and
kept synchronous because OpenEnv's ``step`` is synchronous.

The split below is the part worth keeping: a *rejected call* is feedback the
policy should learn from, a *transport failure* is an outage. Folding the second
into ``success=False`` would launder an outage into a reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable


class ToolTransportError(RuntimeError):
    """The tool could not be reached or answered. Never a policy signal."""


@dataclass(frozen=True)
class ToolExecutionResult:
    """Outcome of one tool call, identical in shape for local and remote tools.

    ``success=False`` means the call was *rejected* (bad arguments) - legitimate
    feedback, and the episode continues.
    """

    tool_name: str
    output: str
    success: bool = True
    call_id: str | None = None
    error: str | None = None
    buffered: bool = False
    """True when a write was captured by the write-ahead buffer rather than
    applied. Recorded for diagnostics; the policy is not told."""


@runtime_checkable
class ToolExecutor(Protocol):
    """Anything that can list tool schemas and run a call by name."""

    def list_tool_schemas(self) -> List[Dict[str, Any]]:
        """OpenAI-format function schemas this backend serves."""
        ...

    def execute(
        self, tool_name: str, arguments: Dict[str, Any], *, call_id: str | None = None
    ) -> ToolExecutionResult:
        """Run one call. Raises ``ToolTransportError`` if the backend is unusable."""
        ...
