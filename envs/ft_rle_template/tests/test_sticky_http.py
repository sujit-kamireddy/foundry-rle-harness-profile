"""The plain HTTP routes must keep an episode alive.

OpenEnv's stock ``/reset`` and ``/step`` build a throwaway environment per
request, so a multi-turn episode cannot survive them - ``step`` lands on an
environment that was never reset. Foundry RLE drives a leased sandbox over
exactly those routes, so this is the difference between a world that trains and
one that fails on its first tool call.

These tests pin the contract rather than the implementation: one environment
serves the whole episode, and the override is reversible.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ft_rle_template.models import FTAction
from ft_rle_template.server.ft_environment import FTEnvironment
from ft_rle_template.server.sticky_http import (
    install_sticky_http_routes,
    stateful_http_enabled,
)


@pytest.fixture
def sticky_app(tcaas, graders) -> Iterator[tuple[TestClient, list]]:
    """An app whose HTTP routes are served by one persistent environment."""
    built: list = []

    def factory() -> FTEnvironment:
        env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
        built.append(env)
        return env

    app = FastAPI()
    install_sticky_http_routes(app, factory, FTAction)
    with TestClient(app) as client:
        yield client, built


def _step(client: TestClient, tool: str, **arguments):
    return client.post(
        "/step", json={"action": {"tool_name": tool, "arguments": arguments}}
    )


def test_step_resumes_the_episode_reset_started(sticky_app):
    """The regression: stock OpenEnv answers this with 'step() before reset()'."""
    client, _ = sticky_app
    assert client.post("/reset", json={"seed": 0, "split": "train"}).status_code == 200
    started = client.get("/state").json()

    assert _step(client, "get_employee_record", employee_id="e-1041").status_code == 200

    now = client.get("/state").json()
    assert now["episode_id"] == started["episode_id"]
    assert now["step_count"] == started["step_count"] + 1


def test_the_whole_episode_is_served_by_one_environment(sticky_app):
    client, built = sticky_app
    client.post("/reset", json={"seed": 0, "split": "train"})
    _step(client, "get_employee_record", employee_id="e-1041")
    client.get("/state")
    assert len(built) == 1


def test_the_episode_still_ends_with_a_reward(sticky_app):
    """Statefulness must not cost us termination or grading."""
    client, _ = sticky_app
    client.post("/reset", json={"seed": 0, "split": "train"})
    body = _step(client, "submit_answer", answer="done").json()
    assert body["done"] is True
    assert isinstance(body["reward"], float)


def test_a_malformed_action_is_a_422_not_a_500(sticky_app):
    client, _ = sticky_app
    client.post("/reset", json={"seed": 0, "split": "train"})
    assert client.post("/step", json={"action": {"arguments": {}}}).status_code == 422


def test_sticky_routes_win_over_routes_registered_earlier(tcaas, graders):
    """Ours are inserted at the front, so OpenEnv's per-request pair never runs."""
    app = FastAPI()

    @app.get("/state")
    def stock_state() -> dict:
        return {"served_by": "stock"}

    install_sticky_http_routes(
        app, lambda: FTEnvironment(tcaas=tcaas, graders=graders), FTAction
    )
    with TestClient(app) as client:
        assert "served_by" not in client.get("/state").json()


def test_the_override_can_be_switched_off(monkeypatch, tcaas, graders):
    """A world that wants OpenEnv's stock behaviour must be able to have it."""
    monkeypatch.setenv("FT_STATEFUL_HTTP", "0")
    assert stateful_http_enabled() is False

    app = FastAPI()
    before = len(app.router.routes)
    assert (
        install_sticky_http_routes(
            app, lambda: FTEnvironment(tcaas=tcaas, graders=graders), FTAction
        )
        is None
    )
    assert len(app.router.routes) == before
