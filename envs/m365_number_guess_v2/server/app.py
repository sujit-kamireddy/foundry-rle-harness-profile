from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

from ..graders.app import app as graders_app
from ..models import NumberGuessAction, NumberGuessObservation
from ..tcaas.app import app as tcaas_app
from .number_guess_environment import NumberGuessEnvironment


MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    NumberGuessEnvironment,
    NumberGuessAction,
    NumberGuessObservation,
    env_name="m365_number_guess_v2",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)

# The image starts only `uvicorn m365_number_guess_v2.server.app:app`, so the
# mocked services ride along as sub-apps. They are still reached over HTTP, so
# the service boundary stays honest. The `/mock` prefix keeps them clear of
# OpenEnv's reserved `/{env_name}/...` routes.
# REAL SYSTEM: drop these mounts and set TCAAS_BASE_URL / GRADERS_BASE_URL.
app.mount("/mock/tcaas", tcaas_app)
app.mount("/mock/graders", graders_app)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
