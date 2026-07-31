"""A dropped-in world must fail at load, not silently mid-training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ft_rle_template.graders.models import Item, RubricRef, Sample
from ft_rle_template.graders.rubrics import RubricNotScorable, score_all
from ft_rle_template.logic import TERMINAL_TOOL
from ft_rle_template.tcaas import identity as identity_module
from ft_rle_template.tcaas.catalog import Catalog
from ft_rle_template.tcaas.identity import IdentityNotConfigured


def _world(tmp_path: Path, **overrides) -> dict:
    base = json.loads(
        (Path(__file__).parent / "fixtures" / "world_it.json").read_text()
    )
    base.update(overrides)
    return base


def test_task_citing_an_unknown_skill_is_rejected(tmp_path):
    world = _world(tmp_path)
    world["tasks"]["train"][0]["skill_id"] = "does-not-exist"
    with pytest.raises(ValueError, match="unknown skill"):
        Catalog(world)


def test_rubric_citing_an_unknown_skill_is_rejected(tmp_path):
    world = _world(tmp_path)
    world["rubrics"][0]["skill_id"] = "does-not-exist"
    with pytest.raises(ValueError, match="unknown skill"):
        Catalog(world)


def test_tool_bound_to_a_missing_dataset_is_rejected(tmp_path):
    """Otherwise every call to that tool fails deep inside a rollout."""
    world = _world(tmp_path)
    world["tools"][0]["serves"]["dataset"] = "nope"
    with pytest.raises(ValueError, match="unknown dataset"):
        Catalog(world)


def test_reference_to_a_missing_dataset_is_rejected(tmp_path):
    world = _world(tmp_path)
    world["tools"][3]["serves"]["references"][0]["dataset"] = "nope"
    with pytest.raises(ValueError, match="unknown dataset"):
        Catalog(world)


def test_a_rubric_with_no_declared_check_raises():
    """The silent-zero failure mode, now loud.

    Before this, an unscorable rubric returned 0.0 with a polite note, so a
    dropped-in world trained happily on a flat reward signal.
    """
    rubric = RubricRef(rubric_id="mystery", check=None)
    with pytest.raises(RubricNotScorable, match="mystery"):
        score_all([rubric], Item(input="x"), Sample(output_text="y"))


def test_a_rubric_with_an_unknown_check_raises():
    rubric = RubricRef(rubric_id="mystery", check="vibes")
    with pytest.raises(RubricNotScorable, match="vibes"):
        score_all([rubric], Item(input="x"), Sample(output_text="y"))


def test_a_rubric_missing_its_check_params_raises():
    """No default id pattern.

    A guessed pattern that does not match the world's ids scores zero on every
    episode, and a rubric that can never be earned looks exactly like a bad
    policy in the training curve.
    """
    rubric = RubricRef(rubric_id="ungrounded", check="cites_grounded_entities")
    with pytest.raises(RubricNotScorable, match="pattern"):
        score_all([rubric], Item(input="x"), Sample(output_text="TKT-0001"))


def test_a_rubric_missing_its_check_params_is_rejected_at_load(tmp_path):
    world = _world(tmp_path)
    world["rubrics"][0]["check_params"] = {}
    with pytest.raises(ValueError, match="pattern"):
        Catalog(world)


def test_a_rubric_without_a_check_still_loads(tmp_path):
    """The real-judge path: a world graded by an LLM declares criteria, not checks."""
    world = _world(tmp_path)
    for rubric in world["rubrics"]:
        rubric.pop("check", None)
        rubric.pop("check_params", None)
    Catalog(world)


def test_a_world_tool_cannot_shadow_a_base_tool(tmp_path):
    """Base tools are matched first, so a collision silently wins.

    A world tool named ``submit_answer`` would never run, and every call to it
    would end the episode instead.
    """
    world = _world(tmp_path)
    world["tools"][0]["tool_name"] = TERMINAL_TOOL
    with pytest.raises(ValueError, match="collides with a base tool"):
        Catalog(world)


def test_demo_identity_is_refused_against_real_services(monkeypatch):
    """Failing closed beats reading another tenant's world under demo defaults."""
    monkeypatch.setattr(identity_module, "using_mock_services", lambda: False)
    for name in ("FT_TENANT_ID", "FT_USER_ID", "FT_WORLD_ID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(IdentityNotConfigured, match="FT_TENANT_ID"):
        identity_module.current_identity()


def test_configured_identity_is_accepted_against_real_services(monkeypatch):
    monkeypatch.setattr(identity_module, "using_mock_services", lambda: False)
    monkeypatch.setenv("FT_TENANT_ID", "tenant-real")
    monkeypatch.setenv("FT_USER_ID", "user-real")
    monkeypatch.setenv("FT_WORLD_ID", "world-real")
    assert identity_module.current_identity().tenant_id == "tenant-real"


def test_demo_identity_still_works_offline(monkeypatch):
    """The mocks must keep running with no configuration at all."""
    monkeypatch.setattr(identity_module, "using_mock_services", lambda: True)
    for name in ("FT_TENANT_ID", "FT_USER_ID", "FT_WORLD_ID"):
        monkeypatch.delenv(name, raising=False)
    assert identity_module.current_identity().tenant_id == "tenant-demo"
