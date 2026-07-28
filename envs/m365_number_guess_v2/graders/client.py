"""Gym-side grading client and the one place score becomes reward.

The conversion is deliberately unforgiving: a missing score means *grading
failed*, not *the policy scored zero*, so it raises. Rewarding 0.0 there would
train the policy on an infrastructure fault (§5.3).
REAL SYSTEM: repoint at tc_graders; the None-score rule stays exactly as is.
"""

from __future__ import annotations

import httpx

from ..config import GRADERS_BASE_URL, REQUEST_TIMEOUT_S
from .models import GradeRequest, GradeResponse


class GradingUnavailable(RuntimeError):
    """Grading could not produce a score. The episode errors; it is not rewarded."""


class GraderClient:
    """Thin synchronous client for the grading service."""

    def __init__(
        self,
        base_url: str = GRADERS_BASE_URL,
        timeout_s: float = REQUEST_TIMEOUT_S,
        http: httpx.Client | None = None,
    ) -> None:
        self._http = http or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_s
        )

    def grade(self, request: GradeRequest) -> GradeResponse:
        """Send one grading job. A transport failure is fatal, not a zero."""
        try:
            response = self._http.post("/grade", json=request.model_dump())
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GradingUnavailable(f"grader unreachable: {exc}") from exc
        return GradeResponse(**response.json())


def to_reward(response: GradeResponse) -> float:
    """Turn the aggregate into the finite [0, 1] reward the profile requires.

    Scores are already [0, 1] here because every rubric declares that range. A
    grader scoring on another scale would be normalized at this one call site.
    """
    if response.score is None:
        raise GradingUnavailable(
            response.error or response.skip_reason or "grader returned no score"
        )
    return max(0.0, min(1.0, response.score))
