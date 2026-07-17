from __future__ import annotations

import os

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import NumberGuessAction, NumberGuessObservation
    from .number_guess_environment import NumberGuessEnvironment
except ImportError:
    from models import NumberGuessAction, NumberGuessObservation
    from openenv.core.env_server.http_server import create_app
    from server.number_guess_environment import NumberGuessEnvironment


MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    NumberGuessEnvironment,
    NumberGuessAction,
    NumberGuessObservation,
    env_name="m365_number_guess_env",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
