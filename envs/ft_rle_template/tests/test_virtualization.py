"""Containment is pluggable, but neither strategy can commit."""

from __future__ import annotations

import inspect

import pytest

from ft_rle_template.tools.buffer import WriteAheadBuffer
from ft_rle_template.tools.virtualization import (
    ContainmentStrategy,
    VirtualizedSession,
    make_containment,
)

FORBIDDEN = {"commit", "flush", "apply", "drain", "send", "replay", "persist"}


def test_local_buffer_satisfies_the_strategy():
    assert isinstance(WriteAheadBuffer(), ContainmentStrategy)


def test_no_strategy_exposes_a_commit_path():
    """The invariant that makes the seam safe: neither side can land an effect."""
    for cls in (WriteAheadBuffer, VirtualizedSession):
        surface = {n for n in dir(cls) if not n.startswith("_")}
        assert not surface & FORBIDDEN, cls.__name__


def test_delegated_strategy_never_wraps_a_commit_endpoint():
    """FT's session API has a commit; this client deliberately never calls it."""
    source = inspect.getsource(VirtualizedSession)
    assert "/commit" not in source
    assert "sessions/{" not in source.replace(
        'f"{self._base_url}/sessions/{self._session_id}/effects"', ""
    )


def test_default_is_the_local_buffer():
    assert isinstance(make_containment("ep-1"), WriteAheadBuffer)


def test_delegated_requires_a_base_url(monkeypatch):
    """Misconfiguration must fail loudly, not silently fall back to local."""
    monkeypatch.setattr(
        "ft_rle_template.tools.virtualization.VIRTUALIZATION_MODE", "delegated"
    )
    with pytest.raises(ValueError, match="FT_VIRTUALIZATION_BASE_URL"):
        make_containment("ep-1", base_url="")


def test_local_buffer_supplies_no_call_headers():
    """Only the delegated strategy tags calls with a session."""
    assert WriteAheadBuffer().request_headers() == {}
