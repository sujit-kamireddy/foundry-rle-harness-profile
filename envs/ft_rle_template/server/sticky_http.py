"""Make the plain HTTP control routes stateful.

OpenEnv's ``/reset`` and ``/step`` handlers each do::

    _env = self._env_factory()
    try:
        ...
    finally:
        _env.close()

so every request builds and destroys its own environment. That is fine for a
stateless echo world and fatal for a multi-turn one: ``reset`` starts an episode
in an object that is thrown away before ``step`` arrives, and ``step`` raises
"step() called before reset()". The session pool that would fix this is only
consulted by ``/ws`` and ``/mcp``.

Foundry RLE drives a leased sandbox over exactly those plain HTTP routes, so
without this module every episode on RLE dies on its first tool call.

A leased sandbox hosts one rollout at a time, so the container can simply hold
one environment for its lifetime. These routes are inserted at the front of the
router, which is the only part of OpenEnv's behaviour they change - ``/ws`` and
``/mcp`` keep their own per-session environments and stay concurrent.

Set ``FT_STATEFUL_HTTP=0`` to fall back to OpenEnv's per-request behaviour.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any, Callable, Dict, Optional, Type

from fastapi import Body, FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.serialization import (
    deserialize_action,
    serialize_observation,
)
from openenv.core.env_server.types import (
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from pydantic import ValidationError


def stateful_http_enabled() -> bool:
    """Read per call so tests and worlds can flip it without a reimport."""
    return os.getenv("FT_STATEFUL_HTTP", "1").strip().lower() not in {"0", "false", "no"}


def _valid_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop arguments the environment does not declare, as OpenEnv does."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


class _StickyEnv:
    """Owns the one environment the HTTP routes share."""

    def __init__(self, factory: Callable[[], Environment]) -> None:
        self._factory = factory
        self._env: Optional[Environment] = None
        self._lock = asyncio.Lock()

    async def get(self) -> Environment:
        if self._env is None:
            # Construction can touch the network, so keep it off the event loop.
            self._env = await run_in_threadpool(self._factory)
        return self._env

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    async def close(self) -> None:
        env, self._env = self._env, None
        if env is not None:
            await run_in_threadpool(env.close)


async def _invoke(env: Environment, name: str, *args: Any, **kwargs: Any) -> Any:
    """Prefer the environment's async variant when it overrides the base."""
    async_fn = getattr(env, f"{name}_async")
    if async_fn.__func__ is not getattr(Environment, f"{name}_async"):
        return await async_fn(*args, **_valid_kwargs(async_fn, kwargs))
    sync_fn = getattr(env, name)
    return await run_in_threadpool(
        lambda: sync_fn(*args, **_valid_kwargs(sync_fn, kwargs))
    )


def install_sticky_http_routes(
    app: FastAPI,
    env_factory: Callable[[], Environment],
    action_cls: Type[Any],
) -> Optional[_StickyEnv]:
    """Front-run OpenEnv's ``/reset``, ``/step`` and ``/state`` routes.

    Returns the holder (or ``None`` when disabled) so callers can inspect or
    tear down the shared environment.
    """
    if not stateful_http_enabled():
        return None

    sticky = _StickyEnv(env_factory)
    router: Any = app.router

    @router.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
    async def sticky_reset(
        request: ResetRequest = Body(default_factory=ResetRequest),
    ) -> ResetResponse:
        env = await sticky.get()
        async with sticky.lock:
            observation = await _invoke(
                env, "reset", **request.model_dump(exclude_unset=True)
            )
        return ResetResponse(**serialize_observation(observation))

    @router.post("/step", response_model=StepResponse, tags=["Environment Control"])
    async def sticky_step(request: StepRequest) -> StepResponse:
        try:
            action = deserialize_action(request.action, action_cls)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=exc.errors()
            )
        env = await sticky.get()
        async with sticky.lock:
            observation = await _invoke(
                env,
                "step",
                action,
                **request.model_dump(exclude_unset=True, exclude={"action"}),
            )
        return StepResponse(**serialize_observation(observation))

    @router.get("/state", tags=["Environment Control"])
    async def sticky_state() -> Any:
        env = await sticky.get()
        return env.state

    # Registration appends; Starlette matches in order. Move the three routes we
    # just added to the front so they win over OpenEnv's per-request versions.
    ours = router.routes[-3:]
    del router.routes[-3:]
    router.routes[0:0] = ours

    router.on_shutdown.append(sticky.close)
    return sticky
