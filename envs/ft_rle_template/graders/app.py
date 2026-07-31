"""Mock tc_graders service.

Scores each rubric, applies the outcome gate, and aggregates to one reward.
REAL SYSTEM: delete this module and set ``GRADERS_BASE_URL``.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException

from . import rubrics as rubric_scorers
from .aggregation import aggregate
from .models import GradeRequest

app = FastAPI(title="Mock tc_graders")


@app.post("/grade")
def grade(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    request = GradeRequest(**payload)
    try:
        results = rubric_scorers.score_all(request.rubrics, request.item, request.sample)
    except rubric_scorers.RubricNotScorable as exc:
        # 501, not a zero score. A world the mock cannot grade must stop the run,
        # not quietly flatten the reward signal to zero for every episode.
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    failed = rubric_scorers.outcome_failed(request.rubrics, results)
    score = 0.0 if failed else aggregate(request.aggregation, results, request.rubrics)

    return {
        "score": score,
        "extra_outputs": {
            "individual_results": [r.model_dump() for r in results],
            "outcome_failed": failed,
        },
    }
