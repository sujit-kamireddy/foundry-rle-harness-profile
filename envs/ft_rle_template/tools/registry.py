"""One callable surface over base tools and the world's user tools.

The gym is a generic sandbox: it ships base tools and is *extended* with user
tools from TCaaS. One result type downstream, so nothing else learns which
backend ran - and the policy cannot tell them apart either.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..tcaas.models import ToolSpec
from .base import ToolExecutionResult
from .local import LocalToolExecutor
from .proxy import TCaaSToolExecutor


class ToolRegistry:
    """Dispatches a call to whichever backend owns the tool."""

    def __init__(self, base: LocalToolExecutor, user: TCaaSToolExecutor) -> None:
        self._base = base
        self._user = user

    def specs(self) -> List[ToolSpec]:
        return self._user.specs() + self._base.specs()

    def list_tool_schemas(self) -> List[Dict[str, Any]]:
        """User tools first: the terminal tool reads better last."""
        return self._user.list_tool_schemas() + self._base.list_tool_schemas()

    def execute(
        self, tool_name: str, arguments: Dict[str, Any], *, call_id: str | None = None
    ) -> ToolExecutionResult:
        if self._base.handles(tool_name):
            return self._base.execute(tool_name, arguments, call_id=call_id)
        return self._user.execute(tool_name, arguments, call_id=call_id)

    @property
    def pending_effects(self) -> List[Dict[str, Any]]:
        return self._user.buffer.summary()
