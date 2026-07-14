from __future__ import annotations

import os

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import M365HelloWorldAction, M365HelloWorldObservation
    from .m365_poc_environment import M365HelloWorldEnvironment
except ImportError:
    from models import M365HelloWorldAction, M365HelloWorldObservation
    from openenv.core.env_server.http_server import create_app
    from server.m365_poc_environment import M365HelloWorldEnvironment


MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    M365HelloWorldEnvironment,
    M365HelloWorldAction,
    M365HelloWorldObservation,
    env_name="m365_poc_env",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()