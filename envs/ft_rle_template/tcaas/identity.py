"""Tenant scope for every TCaaS and grader call.

One tenant per container: the RLE instance is provisioned for a single FT world,
so tenancy is deployment configuration and must never be a ``reset`` parameter -
a seed that could cross tenants would be a data-boundary bug, not a task knob.
REAL SYSTEM: replace these headers with a bearer token carrying the same claims.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


from ..config import using_mock_services


class IdentityNotConfigured(RuntimeError):
    """A real deployment that was never told which tenant it serves."""


@dataclass(frozen=True)
class TenantIdentity:
    tenant_id: str
    user_id: str
    world_id: str

    def headers(self) -> Dict[str, str]:
        return {
            "x-tcaas-tenant-id": self.tenant_id,
            "x-tcaas-user-id": self.user_id,
            "x-tcaas-world-id": self.world_id,
        }


DEMO = {
    "FT_TENANT_ID": "tenant-demo",
    "FT_USER_ID": "user-demo",
}


def current_identity() -> TenantIdentity:
    """Demo defaults apply offline only; a real deployment must be told who it is.

    Failing closed matters more than convenience. A container pointed at real
    TCaaS but missing its tenant configuration would otherwise run as
    ``tenant-demo`` against another tenant's world, and every downstream check
    would pass because the request is internally consistent.

    The world id falls back to the world this image ships, so a dropped-in
    catalog still works with no configuration at all while the mocks are in use.
    """
    configured = {name: os.getenv(name) for name in DEMO}
    world_id = os.getenv("FT_WORLD_ID")

    if not using_mock_services():
        missing = sorted(name for name, value in configured.items() if not value)
        if not world_id:
            missing.append("FT_WORLD_ID")
        if missing:
            raise IdentityNotConfigured(
                "Real TCaaS or grader endpoints are configured, so the demo tenant "
                f"defaults do not apply. Set {sorted(missing)}."
            )

    return TenantIdentity(
        tenant_id=configured["FT_TENANT_ID"] or DEMO["FT_TENANT_ID"],
        user_id=configured["FT_USER_ID"] or DEMO["FT_USER_ID"],
        world_id=world_id or _bundled_world_id(),
    )


def _bundled_world_id() -> str:
    from .catalog import load_catalog

    return load_catalog().world_id
