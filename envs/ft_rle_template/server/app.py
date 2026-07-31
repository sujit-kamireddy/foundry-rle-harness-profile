from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

from ..graders.app import app as graders_app
from ..models import FTAction, FTObservation
from ..tcaas.app import app as tcaas_app
from .ft_environment import FTEnvironment
from .sticky_http import install_sticky_http_routes

MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    FTEnvironment,
    FTAction,
    FTObservation,
    env_name="ft_rle_template",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)

# OpenEnv's own /reset and /step build a throwaway environment per request, so a
# multi-turn episode cannot survive them. Foundry RLE drives a sandbox over
# exactly those routes, so the template serves them from one persistent
# environment instead. /ws and /mcp keep their per-session environments.
install_sticky_http_routes(app, FTEnvironment, FTAction)

# The image starts only `uvicorn ft_rle_template.server.app:app`, so the mocked
# services ride along as sub-apps. They are still reached over HTTP, so the
# service boundary stays honest. The `/mock` prefix keeps them clear of
# OpenEnv's reserved `/{env_name}/...` routes.
# REAL SYSTEM: drop these mounts and set TCAAS_BASE_URL / GRADERS_BASE_URL.
app.mount("/mock/tcaas", tcaas_app)
app.mount("/mock/graders", graders_app)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
