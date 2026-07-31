"""Mock TCaaS service.

Rides along as a sub-app of the gym container so the image runs one uvicorn
process, but the gym still reaches it over HTTP - the boundary that matters
stays honest.
REAL SYSTEM: delete this module and set ``TCAAS_BASE_URL``.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import Body, FastAPI, Header, HTTPException, Query

from . import tools as tool_impls
from .catalog import load_catalog

app = FastAPI(title="Mock TCaaS")


def _require_tenant(tenant_id: str | None, world_id: str | None) -> None:
    """Presence is not enough: the world id must be one this service serves.

    A container is provisioned for exactly one world, so a mismatched header is
    a cross-tenant read attempt, not a bad request.
    """
    if not tenant_id or not world_id:
        raise HTTPException(status_code=401, detail="tenant and world headers required")
    if world_id != load_catalog().world_id:
        raise HTTPException(status_code=403, detail=f"world {world_id!r} not served here")


@app.get("/world")
def get_world(
    x_tcaas_tenant_id: str | None = Header(default=None),
    x_tcaas_world_id: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_tenant(x_tcaas_tenant_id, x_tcaas_world_id)
    return load_catalog().descriptor().model_dump()


@app.get("/tasks")
def get_task(
    split: str = Query(...),
    seed: int = Query(0),
    x_tcaas_tenant_id: str | None = Header(default=None),
    x_tcaas_world_id: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_tenant(x_tcaas_tenant_id, x_tcaas_world_id)
    try:
        return load_catalog().pick_task(split, seed).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/tools")
def get_tools(
    x_tcaas_tenant_id: str | None = Header(default=None),
    x_tcaas_world_id: str | None = Header(default=None),
) -> Any:
    _require_tenant(x_tcaas_tenant_id, x_tcaas_world_id)
    return [t.model_dump() for t in load_catalog().tools()]


@app.post("/tools/{tool_name}/call")
def call_tool(
    tool_name: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    x_tcaas_tenant_id: str | None = Header(default=None),
    x_tcaas_world_id: str | None = Header(default=None),
) -> Dict[str, Any]:
    """400 means the call was rejected; anything else is an outage.

    That split is the contract the gym-side proxy relies on to keep a bad
    argument out of the infrastructure-fault path.
    """
    _require_tenant(x_tcaas_tenant_id, x_tcaas_world_id)
    try:
        return tool_impls.dispatch(
            tool_name, payload.get("arguments") or {}, payload.get("task_id")
        )
    except tool_impls.ToolRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except tool_impls.ToolUnservable as exc:
        # Deliberately not a 400. A misconfigured world is an operator error, and
        # dressing it up as a policy signal would let training run on nothing.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
