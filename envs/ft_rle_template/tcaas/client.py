"""Gym-side client for TCaaS.

Every call carries tenant identity. A failure here is fatal to the episode: a
fallback task or a silently empty tool list would train on the wrong data.
REAL SYSTEM: same shape against deployed TCaaS; auth moves from headers to a
bearer token.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx

from ..config import REQUEST_TIMEOUT_S, TCAAS_BASE_URL
from .identity import TenantIdentity, current_identity
from .models import TaskBundle, ToolSpec, WorldDescriptor


class TCaaSUnavailable(RuntimeError):
    """TCaaS could not be reached or answered. Never a policy signal."""


class TCaaSClient:
    def __init__(
        self,
        base_url: str | None = None,
        identity: TenantIdentity | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._base_url = (base_url or TCAAS_BASE_URL).rstrip("/")
        self._identity = identity or current_identity()
        self._timeout = timeout_s or REQUEST_TIMEOUT_S

    def _get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        try:
            response = httpx.get(
                f"{self._base_url}{path}",
                params=params,
                headers=self._identity.headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TCaaSUnavailable(f"GET {path} failed: {exc}") from exc
        return response.json()

    def world(self) -> WorldDescriptor:
        """World metadata used to render the harness profile."""
        return WorldDescriptor(**self._get("/world"))

    def pick_task(self, split: str, seed: int) -> TaskBundle:
        """Resolve ``(split, seed)`` into one episode's content."""
        return TaskBundle(**self._get("/tasks", {"split": split, "seed": seed}))

    def list_tools(self) -> List[ToolSpec]:
        return [ToolSpec(**t) for t in self._get("/tools")]

    def call_tool(
        self,
        task_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        headers: Dict[str, str] | None = None,
    ) -> httpx.Response:
        """Raw call. The proxy owns rejection-vs-outage classification."""
        return httpx.post(
            f"{self._base_url}/tools/{tool_name}/call",
            json={"task_id": task_id, "arguments": arguments or {}},
            headers={**self._identity.headers(), **(headers or {})},
            timeout=self._timeout,
        )
