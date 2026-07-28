"""Who this gym container belongs to.

Identity is deployment-scoped, not per-episode: one container serves one
tenant/user, so the seed selects a task *within* an already-scoped catalog and
``tenant_id`` never appears on ``/reset``.
REAL SYSTEM: these become claims on a real credential; the scoping model - set
once at deploy time, sent on every call - is unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

TENANT_HEADER = "x-tcaas-tenant-id"
USER_HEADER = "x-tcaas-user-id"


@dataclass(frozen=True)
class TenantContext:
    """The tenant/user this container is provisioned for."""

    tenant_id: str
    user_id: str

    @classmethod
    def from_env(cls) -> "TenantContext":
        """Read the deployment's identity; TCaaS injects these at provision time."""
        return cls(
            tenant_id=os.getenv("TCAAS_TENANT_ID", "tenant-demo"),
            user_id=os.getenv("TCAAS_USER_ID", "user-demo"),
        )

    def headers(self) -> dict[str, str]:
        """Sent on every TCaaS call. REAL SYSTEM: a bearer token carrying these claims."""
        return {TENANT_HEADER: self.tenant_id, USER_HEADER: self.user_id}
