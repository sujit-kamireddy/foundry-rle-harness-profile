"""User tools, executed by TCaaS over HTTP.

TCaaS publishes both the schema and the endpoint, so the gym never implements
these - it forwards arguments and relays the answer. That is what lets a tool
like ``compare`` consult task data the sandbox is not allowed to see.
REAL SYSTEM: same shape against the deployed TCaaS; auth would be added here.
"""

from __future__ import annotations

from typing import Any

import httpx

from .base import ToolExecutionResult, ToolTransportError


class TCaaSToolExecutor:
    """Proxies calls to TCaaS-hosted tools, scoped to one episode's task."""

    def __init__(self, client: Any, task_id: str) -> None:
        self._client = client
        self._task_id = task_id

    def list_tool_schemas(self) -> list[dict[str, Any]]:
        """Schemas come from TCaaS, so new user tools need no gym change."""
        return self._client.list_tool_schemas()

    def execute(
        self, tool_name: str, arguments: dict[str, Any], *, call_id: str | None = None
    ) -> ToolExecutionResult:
        try:
            output = self._client.call_tool(self._task_id, tool_name, arguments)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    output=exc.response.json().get("detail", "Rejected."),
                    success=False,
                    call_id=call_id,
                    error="invalid_argument",
                )
            raise ToolTransportError(f"{tool_name} failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ToolTransportError(f"{tool_name} unreachable: {exc}") from exc
        return ToolExecutionResult(tool_name=tool_name, output=output, call_id=call_id)
