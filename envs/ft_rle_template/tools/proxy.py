"""User tools, executed by TCaaS over HTTP - in one of two modes.

TCaaS publishes both the schema and the endpoint, so the gym never implements
these; it forwards arguments and relays the answer. The only thing the gym adds
is the dual-mode split:

============  =======================  ==================================
effect        inference mode           training mode
============  =======================  ==================================
``read``      real call                real call, then overlaid with the
                                       episode's buffered writes
``write``     real call                buffered, never sent
============  =======================  ==================================

REAL SYSTEM: same shape against the customer's registered MCP endpoints; auth
would be added here.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx

from ..tcaas.models import ToolSpec
from .base import ToolExecutionResult, ToolTransportError
from .buffer import WriteAheadBuffer, render


class TCaaSToolExecutor:
    """Proxies calls to TCaaS-hosted tools, scoped to one episode's task."""

    def __init__(
        self,
        client: Any,
        task_id: str,
        specs: List[ToolSpec],
        *,
        training: bool,
        buffer: Any | None = None,
    ) -> None:
        self._client = client
        self._task_id = task_id
        self._specs = {spec.tool_name: spec for spec in specs}
        self._training = training
        self._buffer = buffer if buffer is not None else WriteAheadBuffer()

    @property
    def buffer(self) -> Any:
        """The containment strategy for this episode: local buffer or FT session."""
        return self._buffer

    def specs(self) -> List[ToolSpec]:
        return list(self._specs.values())

    def handles(self, tool_name: str) -> bool:
        return tool_name in self._specs

    def list_tool_schemas(self) -> List[Dict[str, Any]]:
        """Schemas come from TCaaS, so new user tools need no gym change."""
        return [spec.to_openai_schema() for spec in self._specs.values()]

    def execute(
        self, tool_name: str, arguments: Dict[str, Any], *, call_id: str | None = None
    ) -> ToolExecutionResult:
        spec = self._specs.get(tool_name)
        if spec is None:
            return ToolExecutionResult(
                tool_name=tool_name,
                output=f"Unknown tool {tool_name!r}.",
                success=False,
                call_id=call_id,
                error="unknown_tool",
            )

        if self._training and spec.is_write():
            payload = self._buffer.record(spec, arguments or {})
            return ToolExecutionResult(
                tool_name=tool_name,
                output=render(payload),
                call_id=call_id,
                buffered=True,
            )

        payload = self._call(spec, arguments or {}, call_id)
        if isinstance(payload, ToolExecutionResult):
            return payload

        if self._training:
            payload["_request_arguments"] = dict(arguments or {})
            payload = self._buffer.overlay(spec, payload)

        return ToolExecutionResult(
            tool_name=tool_name, output=render(payload), call_id=call_id
        )

    def _call(
        self, spec: ToolSpec, arguments: Dict[str, Any], call_id: str | None
    ) -> Dict[str, Any] | ToolExecutionResult:
        """400 is a rejection the policy should see; anything else is an outage.

        A delegated containment strategy contributes headers here, so reads are
        tagged with the virtualization session and FT applies its own pending
        effects server-side.
        """
        headers = getattr(self._buffer, "request_headers", None)
        extra = headers() if callable(headers) else None
        try:
            response = self._client.call_tool(
                self._task_id, spec.tool_name, arguments, headers=extra
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                detail = _detail(exc.response)
                return ToolExecutionResult(
                    tool_name=spec.tool_name,
                    output=detail,
                    success=False,
                    call_id=call_id,
                    error="invalid_argument",
                )
            raise ToolTransportError(f"{spec.tool_name} failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ToolTransportError(f"{spec.tool_name} unreachable: {exc}") from exc
        return response.json()


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", "Rejected."))
    except Exception:  # noqa: BLE001 - a malformed error body is still a rejection
        return "Rejected."
