"""MOCK tc_graders: one ``POST /grade`` endpoint over the item/sample contract.

Scores every requested rubric, aggregates, and reports per-rubric detail under
``extra_outputs.individual_results`` - where the real service puts it.
REAL SYSTEM: replace with the tc_graders deployment and repoint
``GRADERS_BASE_URL``; the request/response shapes are what we kept faithful.
"""

from __future__ import annotations

from fastapi import FastAPI

from .aggregation import PASS_THRESHOLD, aggregate
from .models import GradeRequest, GradeResponse, RubricResult
from .rubrics import SCORERS, Unscorable


def create_graders_app() -> FastAPI:
    """Build the mock grading service."""
    app = FastAPI(title="tc_graders (mock)")

    @app.post("/grade")
    def run_grade(request: GradeRequest) -> GradeResponse:
        """Score, aggregate, and report. An unreadable sample fails loudly."""
        if not request.rubrics:
            return GradeResponse.failure("no rubrics requested")

        results: list[RubricResult] = []
        weights: list[float] = []
        for rubric in request.rubrics:
            scorer = SCORERS.get(rubric.rubric_id)
            if scorer is None:
                return GradeResponse.failure(f"no scorer for rubric {rubric.rubric_id}")
            try:
                score = scorer(request.item, request.sample)
            except Unscorable as exc:
                return GradeResponse.failure(str(exc))
            low, high = rubric.range
            if not low <= score <= high:
                return GradeResponse.failure(
                    f"rubric {rubric.rubric_id} scored {score}, outside its range {rubric.range}"
                )
            results.append(
                RubricResult(
                    rubric_id=rubric.rubric_id,
                    score=score,
                    passed=score >= PASS_THRESHOLD,
                )
            )
            weights.append(rubric.weight)

        try:
            total = aggregate([r.score for r in results], weights, request.aggregation)
        except ValueError as exc:
            return GradeResponse.failure(str(exc))

        return GradeResponse(
            score=total,
            passed=total >= PASS_THRESHOLD,
            extra_outputs={"individual_results": [r.model_dump() for r in results]},
        )

    return app


app = create_graders_app()
