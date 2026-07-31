"""Content service contract: determinism, split isolation, tenant scoping."""

from __future__ import annotations

import pytest

from ft_rle_template.tcaas.client import TCaaSClient, TCaaSUnavailable


def test_same_seed_yields_the_same_task(tcaas):
    """GRPO reuses a seed across a group, so this is a correctness requirement."""
    first = tcaas.pick_task("train", 7)
    second = tcaas.pick_task("train", 7)
    assert first.task_id == second.task_id
    assert first.user_query == second.user_query


def test_seeds_spread_across_the_split(tcaas):
    world = tcaas.world()
    seen = {tcaas.pick_task("train", s).task_id for s in range(world.split_sizes["train"])}
    assert len(seen) == world.split_sizes["train"]


def test_splits_do_not_overlap(tcaas):
    """Eval must not report on tasks the policy trained against."""
    world = tcaas.world()
    train = {tcaas.pick_task("train", s).task_id for s in range(world.split_sizes["train"])}
    val = {
        tcaas.pick_task("validation", s).task_id
        for s in range(world.split_sizes["validation"])
    }
    assert not train & val


def test_bundle_carries_everything_one_episode_needs(tcaas):
    bundle = tcaas.pick_task("train", 0)
    assert bundle.skill and bundle.user_query
    assert bundle.rubrics and bundle.tools
    assert bundle.split == "train"


def test_bundle_rubrics_belong_to_the_bundle_skill(tcaas):
    bundle = tcaas.pick_task("train", 0)
    assert {r.skill_id for r in bundle.rubrics} == {bundle.skill_id}


def test_world_descriptor_drives_the_profile(tcaas):
    world = tcaas.world()
    assert world.tools and world.skills
    assert world.content_version
    assert set(world.split_sizes) >= {"train", "validation"}


def test_unknown_split_is_an_error_not_a_silent_fallback(tcaas):
    """Silently falling back would train on the wrong data."""
    with pytest.raises(TCaaSUnavailable):
        tcaas.pick_task("nonexistent", 0)


def test_requests_carry_tenant_identity(services):
    """A world is tenant-scoped; an unscoped request must be refused."""
    import httpx

    response = httpx.get(f"{services}/mock/tcaas/world", timeout=5)
    assert response.status_code == 401


def test_wrong_tenant_is_refused(services):
    from ft_rle_template.tcaas.identity import TenantIdentity

    intruder = TCaaSClient(
        base_url=f"{services}/mock/tcaas",
        identity=TenantIdentity(
            tenant_id="other-tenant", user_id="other-user", world_id="other-world"
        ),
    )
    with pytest.raises(TCaaSUnavailable):
        intruder.world()
