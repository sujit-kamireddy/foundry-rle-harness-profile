"""MOCK TCaaS service: serves the catalog and executes the user tools.

Mounted as a sub-app of the gym server (the container runs one uvicorn process),
but reached only over HTTP, so the service boundary stays honest.
REAL SYSTEM: replace with the TCaaS deployment and repoint ``TCAAS_BASE_URL``;
the gym-side client and tool proxy stay as they are.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from .catalog import Catalog
from .identity import TENANT_HEADER, USER_HEADER
from .models import Rubric, Skill, TaskBundle
from .tools import USER_TOOL_SCHEMAS, USER_TOOLS, ToolRejected

CATALOG = Catalog.load()


def require_tenant(
    tenant_id: str = Header(alias=TENANT_HEADER),
    user_id: str = Header(alias=USER_HEADER),
) -> None:
    """Every route is tenant-scoped: content belongs to exactly one tenant/user.

    REAL SYSTEM: validate a bearer token's claims instead of trusting headers.
    """
    if (tenant_id, user_id) != (CATALOG.tenant.tenant_id, CATALOG.tenant.user_id):
        raise HTTPException(status_code=403, detail="content belongs to another tenant")


class ToolCallRequest(BaseModel):
    """One user-tool invocation, scoped to the task whose data it may read."""

    task_id: str
    arguments: dict[str, Any] = {}


class ToolCallResponse(BaseModel):
    """What the policy reads back from the tool."""

    output: str


def create_tcaas_app() -> FastAPI:
    """Build the mock service. One tenant's catalog, loaded once, served read-only."""
    app = FastAPI(title="TCaaS (mock)", dependencies=[Depends(require_tenant)])

    @app.get("/skills")
    def list_skills() -> list[Skill]:
        return list(CATALOG.skills.values())

    @app.get("/rubrics")
    def list_rubrics(skill_id: str | None = Query(default=None)) -> list[Rubric]:
        return CATALOG.rubrics_for(skill_id) if skill_id else CATALOG.rubrics

    @app.get("/tools")
    def list_tools() -> list[dict[str, Any]]:
        """OpenAI-format schemas for the user tools TCaaS hosts."""
        return USER_TOOL_SCHEMAS

    @app.get("/splits")
    def split_sizes() -> dict[str, int]:
        """Task count per split; the profile's ``evalDefaults.limit`` mirrors it."""
        return {split: len(tasks) for split, tasks in CATALOG.tasks.items()}

    @app.get("/tasks")
    def pick_task(split: str, seed: int) -> TaskBundle:
        """The sample selector: same ``(split, seed)`` always yields the same task."""
        try:
            return CATALOG.pick_task(split, seed)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/tools/{tool_name}")
    def call_tool(tool_name: str, request: ToolCallRequest) -> ToolCallResponse:
        """Run a user tool against its task's data. Bad arguments give a 400."""
        tool = USER_TOOLS.get(tool_name)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"unknown tool {tool_name}")
        try:
            task = CATALOG.get_task(request.task_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"unknown task {request.task_id}"
            ) from exc
        try:
            return ToolCallResponse(output=tool(task.data, request.arguments))
        except ToolRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_tcaas_app()
