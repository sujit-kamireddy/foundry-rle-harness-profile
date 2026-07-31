"""Runtime configuration for the FT gym, all overridable by environment variable.

Defaults point at this same container, where the TCaaS and tc_graders mocks run
as sub-apps under ``/mock`` - clear of OpenEnv's reserved ``/{env_name}/...``
routes.
REAL SYSTEM: repoint at deployed TCaaS / tc_graders; the mocks approximate their
shapes only, so paths, payloads, and auth will need adjustment.
"""

from __future__ import annotations

import os

SELF_BASE_URL = os.getenv("SELF_BASE_URL", "http://localhost:8000")

TCAAS_BASE_URL = os.getenv("TCAAS_BASE_URL", f"{SELF_BASE_URL}/mock/tcaas")
GRADERS_BASE_URL = os.getenv("GRADERS_BASE_URL", f"{SELF_BASE_URL}/mock/graders")

MOCK_PREFIX = f"{SELF_BASE_URL}/mock"


def using_mock_services() -> bool:
    """True while both content and grading still point at the bundled mocks.

    Anything else is a real deployment, where the demo tenant defaults must not
    silently apply.
    """
    return TCAAS_BASE_URL.startswith(MOCK_PREFIX) and GRADERS_BASE_URL.startswith(
        MOCK_PREFIX
    )

REQUEST_TIMEOUT_S = float(os.getenv("SERVICE_REQUEST_TIMEOUT_S", "10"))
"""Per-request deadline, kept well under the profile's episode budget."""

TOOL_MODE = os.getenv("FT_TOOL_MODE", "training").strip().lower()
"""``training`` or ``inference``.

The only switch that changes tool side effects:

- ``inference``  - reads and writes both hit the real backend.
- ``training``   - reads hit the real backend, writes go to a per-episode
  write-ahead buffer and are never committed.

Training is the default because this image exists to serve training and eval
runs; an operator must opt *in* to real writes.
"""

MAX_STEPS_PER_EPISODE = int(os.getenv("FT_MAX_STEPS_PER_EPISODE", "12"))
EPISODE_TIMEOUT_S = int(os.getenv("FT_EPISODE_TIMEOUT_S", "180"))

SUCCESS_THRESHOLD = float(os.getenv("FT_SUCCESS_THRESHOLD", "0.5"))
"""Kept identical in the rendered profile so grader-passed and harness-success
agree by construction."""


def training_mode() -> bool:
    return TOOL_MODE != "inference"


VIRTUALIZATION_MODE = os.getenv("FT_VIRTUALIZATION", "local").strip().lower()
"""``local`` or ``delegated``: who contains a training write.

- ``local``     - the gym's own write-ahead buffer. Offline, no dependencies.
- ``delegated`` - an FT tool-virtualization session. FT's connectors already know
  how to merge their own entities, so delegating removes the gym's generic
  overlay guesswork.

Neither can commit. The strategy decides *where* effects are held, never whether
they can escape.
"""

VIRTUALIZATION_BASE_URL = os.getenv("FT_VIRTUALIZATION_BASE_URL", "")
"""Required when ``FT_VIRTUALIZATION=delegated``."""

def world_tools_module() -> str:
    """Optional module exporting ``TOOLS`` for tools ``serves`` cannot express.

    A ``serves`` binding filters a collection. A tool that *computes* from the
    episode's task data - the classic probe-the-hidden-state tool - has nothing
    to filter, so the world supplies the function instead. Offline mock only: in
    production the tool is a real MCP endpoint and this is never read.

    Read per call, like ``catalog_path()``, so a world swaps without a restart.
    """
    return os.getenv("FT_WORLD_TOOLS", "")


def world_checks_module() -> str:
    """Optional module exporting ``CHECKS`` for rubrics the built-in scorers miss.

    Same story: offline mock only. A real judge reads each rubric's ``criteria``.
    """
    return os.getenv("FT_WORLD_CHECKS", "")
