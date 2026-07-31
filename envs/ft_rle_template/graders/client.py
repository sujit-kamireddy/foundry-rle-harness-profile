"""Gym-side client for tc_graders, plus the reward rules.

Two rules carry the weight here:

1. ``score is None`` raises. A dead grader is an infrastructure fault, not a
   policy scoring zero - rewarding 0.0 there would train on an outage.
2. A wrong answer scores 0.0 overall, so a correct solve always outranks a wrong
   one. Process rubrics must not pay out when the outcome is wrong.
"""

from __future__ import annotations

import httpx

from ..config import GRADERS_BASE_URL, REQUEST_TIMEOUT_S
from ..tcaas.identity import TenantIdentity, current_identity
from .models import GradeRequest, GradeResponse


class GraderUnavailable(RuntimeError):
    """tc_graders could not score this episode. Never a reward of 0.0."""


class GraderClient:
    def __init__(
        self,
        base_url: str | None = None,
        identity: TenantIdentity | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._base_url = (base_url or GRADERS_BASE_URL).rstrip("/")
        self._identity = identity or current_identity()
        self._timeout = timeout_s or REQUEST_TIMEOUT_S

    def grade(self, request: GradeRequest) -> GradeResponse:
        try:
            response = httpx.post(
                f"{self._base_url}/grade",
                json=request.model_dump(),
                headers=self._identity.headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GraderUnavailable(f"grade failed: {exc}") from exc
        return GradeResponse(**response.json())


def to_reward(response: GradeResponse) -> float:
    """Convert a grade into a reward, refusing to invent one."""
    if response.score is None:
        raise GraderUnavailable(
            "grader returned no score; refusing to reward 0.0 for an outage"
        )
    return float(response.score)
