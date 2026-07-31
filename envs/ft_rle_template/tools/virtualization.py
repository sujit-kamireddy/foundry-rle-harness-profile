"""Two ways to contain a training write, behind one interface.

FT already has tool virtualization: a session that captures effects, answers
reads with them applied, and only lands them on an explicit commit. Reimplementing
that in the gym would mean two containment implementations diverging, and the
gym's version can never know a connector's merge semantics the way the connector
does.

So containment is a strategy:

``local``
    ``WriteAheadBuffer`` in this package. Self-contained, works offline, and
    needs no FT service. The default, and what the mock world runs on.

``delegated``
    ``VirtualizedSession``. Opens an FT tool-virtualization session per episode,
    tags every call with it, and reads the effect list back for the audit trail.
    FT's own overlay semantics apply, so the "generic argument-subset match" gap
    disappears.

The invariant both must hold, and the reason this file exists rather than a bare
``if``: **neither exposes a commit path.** The delegated strategy deliberately
does not wrap FT's commit endpoint. Training cannot land an effect because no
object reachable from an episode has a method that does.

REAL SYSTEM: ``VirtualizedSession`` below is written against the documented
session shape (open / call-with-session / effects / reset). Endpoint paths and
auth will need adjusting once the contract is fixed; the strategy interface will
not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

import httpx

from ..config import REQUEST_TIMEOUT_S, VIRTUALIZATION_MODE
from ..tcaas.models import ToolSpec
from .base import ToolTransportError
from .buffer import WriteAheadBuffer


@runtime_checkable
class ContainmentStrategy(Protocol):
    """What the proxy needs in order to keep a training write out of a tenant.

    Note what is absent: nothing here can commit.
    """

    def record(self, spec: ToolSpec, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Contain a write and return the result the caller would have seen."""

    def overlay(self, spec: ToolSpec, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply contained writes to a real read result."""

    def request_headers(self) -> Dict[str, str]:
        """Headers to attach to outbound tool calls, if any."""

    def summary(self) -> List[Dict[str, Any]]:
        """The audit trail of what training would have written."""

    def __len__(self) -> int: ...


class VirtualizedSession:
    """Delegates containment to an FT tool-virtualization session.

    Reads still go to the real backend, but tagged with the session so FT applies
    the session's own pending effects - which is the whole point of delegating:
    the connector knows how to merge its entities, and the gym does not.
    """

    def __init__(
        self,
        base_url: str,
        episode_id: str,
        headers: Dict[str, str] | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._episode_id = episode_id
        self._headers = dict(headers or {})
        self._timeout = timeout_s
        self._session_id = self._open()
        self._local_count = 0

    def _open(self) -> str:
        try:
            response = httpx.post(
                f"{self._base_url}/sessions",
                json={"episode_id": self._episode_id, "mode": "virtualized"},
                headers=self._headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolTransportError(
                f"could not open a virtualization session: {exc}"
            ) from exc
        return str(response.json()["session_id"])

    def request_headers(self) -> Dict[str, str]:
        """Every tool call carries the session, so FT contains it server-side."""
        return {
            "x-ft-virtualization-session": self._session_id,
            "X-Debug-EnableToolVirtualization": "true",
        }

    def record(self, spec: ToolSpec, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Writes are *sent*, but into the session, so they never land.

        This is the inversion versus the local strategy: containment happens on
        FT's side, and the gym's job is only to refuse to ever commit.
        """
        self._local_count += 1
        try:
            response = httpx.post(
                f"{self._base_url}/tools/{spec.tool_name}/call",
                json={"arguments": arguments or {}},
                headers={**self._headers, **self.request_headers()},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolTransportError(f"{spec.tool_name} failed in session: {exc}") from exc
        return response.json()

    def overlay(self, spec: ToolSpec, result: Dict[str, Any]) -> Dict[str, Any]:
        """A no-op: the session already applied its effects to the read."""
        result.pop("_request_arguments", None)
        return result

    def summary(self) -> List[Dict[str, Any]]:
        try:
            response = httpx.get(
                f"{self._base_url}/sessions/{self._session_id}/effects",
                headers=self._headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolTransportError(f"could not read session effects: {exc}") from exc
        return list(response.json().get("effects", []))

    def __len__(self) -> int:
        return self._local_count


def make_containment(
    episode_id: str,
    base_url: str | None = None,
    headers: Dict[str, str] | None = None,
) -> ContainmentStrategy:
    """Pick a containment strategy for one episode."""
    if VIRTUALIZATION_MODE == "delegated":
        if not base_url:
            raise ValueError(
                "FT_VIRTUALIZATION=delegated requires FT_VIRTUALIZATION_BASE_URL"
            )
        return VirtualizedSession(base_url, episode_id, headers)
    return WriteAheadBuffer()
