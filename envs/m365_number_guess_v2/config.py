"""Base URLs for the gym's external services, overridable by env var.

Defaults point at this same container, where both mocks run as sub-apps under
``/mock`` - clear of OpenEnv's reserved ``/{env_name}/...`` routes.
REAL SYSTEM: repoint at deployed TCaaS / tc_graders; the mocks approximate their
shapes only, so paths, payloads, and auth will need adjustment.
"""

from __future__ import annotations

import os

SELF_BASE_URL = os.getenv("SELF_BASE_URL", "http://localhost:8000")

TCAAS_BASE_URL = os.getenv("TCAAS_BASE_URL", f"{SELF_BASE_URL}/mock/tcaas")
GRADERS_BASE_URL = os.getenv("GRADERS_BASE_URL", f"{SELF_BASE_URL}/mock/graders")

REQUEST_TIMEOUT_S = float(os.getenv("SERVICE_REQUEST_TIMEOUT_S", "10"))
"""Per-request deadline, well under the profile's 60s episode budget."""
