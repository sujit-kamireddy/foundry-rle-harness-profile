"""Live-server fixtures, parameterised by world.

The mocks are exercised over real HTTP rather than in-process, because the
service boundary is the thing under test: the gym must not be able to reach
TCaaS data any way other than an HTTP call.

Tests run against fixture worlds, never the shipped ``tcaas/catalog.json``. That
is deliberate - M365 replaces the shipped catalog, and the test suite has to keep
passing when they do.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest
import uvicorn
from fastapi import FastAPI

from ft_rle_template.graders.app import app as graders_app
from ft_rle_template.graders.client import GraderClient
from ft_rle_template.tcaas import catalog as catalog_module
from ft_rle_template.tcaas import tools as mock_tools
from ft_rle_template.tcaas.app import app as tcaas_app
from ft_rle_template.tcaas.client import TCaaSClient
from ft_rle_template.tcaas.identity import TenantIdentity

FIXTURES = Path(__file__).parent / "fixtures"
WORLD_HR = FIXTURES / "world_hr.json"
WORLD_IT = FIXTURES / "world_it.json"
SHIPPED_WORLD = Path(__file__).resolve().parents[1] / "tcaas" / "catalog.json"
"""The example M365 replaces. Only the profile drift test looks at it."""


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "world(path): run this test against a world file")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def services() -> Iterator[str]:
    root = FastAPI()
    root.mount("/mock/tcaas", tcaas_app)
    root.mount("/mock/graders", graders_app)

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(root, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 20
    while time.time() < deadline and not server.started:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("mock services did not start")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(autouse=True)
def world(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Select the world for a test. Defaults to HR; override with @pytest.mark.world.

    The server reads ``FT_CATALOG_PATH`` per request, so swapping worlds mid-session
    needs no restart - which is also how an operator validates their own catalog.
    """
    marker = request.node.get_closest_marker("world")
    path = Path(marker.args[0]) if marker else WORLD_HR

    previous = os.environ.get("FT_CATALOG_PATH")
    os.environ["FT_CATALOG_PATH"] = str(path)
    mock_tools.reset_store()
    try:
        yield path
    finally:
        mock_tools.reset_store()
        if previous is None:
            os.environ.pop("FT_CATALOG_PATH", None)
        else:
            os.environ["FT_CATALOG_PATH"] = previous


@pytest.fixture
def identity(world: Path) -> TenantIdentity:
    """Identity follows the selected world, mirroring per-deployment config."""
    return TenantIdentity(
        tenant_id="tenant-test",
        user_id="user-test",
        world_id=catalog_module.load_catalog().world_id,
    )


@pytest.fixture
def tcaas(services: str, identity: TenantIdentity) -> TCaaSClient:
    return TCaaSClient(base_url=f"{services}/mock/tcaas", identity=identity)


@pytest.fixture
def graders(services: str, identity: TenantIdentity) -> GraderClient:
    return GraderClient(base_url=f"{services}/mock/graders", identity=identity)
