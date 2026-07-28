"""Gym-side HTTP client for TCaaS.

Every call is fail-loud and tenant-scoped: a missing task or an unreachable
service must surface as an error, never as a fallback task, or the run trains on
the wrong data (§5.3).
REAL SYSTEM: swap the base URL and send a real credential instead of headers.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import REQUEST_TIMEOUT_S, TCAAS_BASE_URL
from .identity import TenantContext
from .models import TaskBundle


class TCaaSClient:
    """Thin synchronous client; the OpenEnv env methods it serves are sync too."""

    def __init__(
        self,
        base_url: str = TCAAS_BASE_URL,
        timeout_s: float = REQUEST_TIMEOUT_S,
        tenant: TenantContext | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        self._http = http or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_s
        )
        self._http.headers.update((tenant or TenantContext.from_env()).headers())

    def pick_task(self, split: str, seed: int) -> TaskBundle:
        """Resolve one episode's task bundle from ``(split, seed)``."""
        payload = self._request("GET", "/tasks", params={"split": split, "seed": seed})
        return TaskBundle(**payload)

    def list_tool_schemas(self) -> list[dict[str, Any]]:
        """OpenAI-format schemas for the user tools TCaaS hosts."""
        return self._request("GET", "/tools")

    def split_size(self, split: str) -> int:
        """Task count in a split, used to pin ``evalDefaults.limit``."""
        return self._request("GET", "/splits")[split]

    def call_tool(self, task_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke a TCaaS-hosted user tool and return what the policy reads back."""
        payload = {"task_id": task_id, "arguments": arguments}
        return self._request("POST", f"/tools/{tool_name}", json=payload)["output"]

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        response = self._http.request(method, path, params=params, json=json)
        response.raise_for_status()
        return response.json()
