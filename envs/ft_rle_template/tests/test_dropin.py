"""The claim under test: M365 drops in a world and everything works, no code edits.

Every test here runs against ``fixtures/world_it.json`` - a different domain,
different tools, different rubrics, different id shapes, and a write tool with a
different entity. Nothing in the template mentions tickets or assets.

If any of these fail, the folder is one worked example rather than a template.
"""

from __future__ import annotations

import json

import pytest

from ft_rle_template.models import FTAction
from ft_rle_template.profile import render_harness_profile
from ft_rle_template.server.ft_environment import FTEnvironment
from ft_rle_template.tcaas import tools as mock_tools
from ft_rle_template.tcaas.catalog import load_catalog

from .conftest import WORLD_IT

pytestmark = pytest.mark.world(str(WORLD_IT))


def test_profile_renders_the_new_action_space():
    profile = render_harness_profile(load_catalog().descriptor())
    assert profile["actionSpace"]["discoveredActions"] == [
        "list_open_tickets",
        "get_asset",
        "list_escalations",
        "escalate_ticket",
        "submit_answer",
    ]
    assert profile["evalDefaults"]["limit"] == 1


def test_reset_works_without_configuring_a_world_id(tcaas, graders):
    """A dropped-in catalog must not need FT_WORLD_ID set to boot."""
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    obs = env.reset(seed=0, split="train")
    assert "helpdesk" in obs.prompt.lower()


def test_new_read_tools_serve_from_catalog_data(tcaas, graders):
    """The gap that made this not a template: a new read tool with no Python."""
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    env.reset(seed=0, split="train")
    obs = env.step(FTAction(tool_name="list_open_tickets", arguments={}))
    items = json.loads(obs.feedback)["items"]
    assert {i["ticket_id"] for i in items} == {"TKT-9001", "TKT-9002", "TKT-9003"}


def test_constant_predicate_filters_closed_tickets(tcaas, graders):
    """`where` in the binding is what hides the closed ticket."""
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    env.reset(seed=0, split="train")
    obs = env.step(FTAction(tool_name="list_open_tickets", arguments={}))
    assert "TKT-9004" not in obs.feedback


def test_optional_filter_narrows_results(tcaas, graders):
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    env.reset(seed=0, split="train")
    obs = env.step(
        FTAction(tool_name="list_open_tickets", arguments={"severity": "critical"})
    )
    assert [i["ticket_id"] for i in json.loads(obs.feedback)["items"]] == ["TKT-9001"]


def test_required_argument_miss_is_a_rejection(tcaas, graders):
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    env.reset(seed=0, split="train")
    obs = env.step(FTAction(tool_name="get_asset", arguments={"asset_id": "AST-9999"}))
    assert not obs.done
    assert "AST-9999" in obs.feedback


def test_new_write_tool_is_contained(tcaas, graders):
    """The leak test, against a world the template has never seen."""
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    env.reset(seed=0, split="train")
    obs = env.step(
        FTAction(
            tool_name="escalate_ticket",
            arguments={"ticket_id": "TKT-9001", "reason": "blocking"},
        )
    )
    assert obs.pending_effects[0]["entity_type"] == "escalation"
    assert mock_tools.committed_writes() == []


def test_read_after_write_overlay_works_for_the_new_entity(tcaas, graders):
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    env.reset(seed=0, split="train")
    env.step(
        FTAction(
            tool_name="escalate_ticket",
            arguments={"ticket_id": "TKT-9001", "reason": "blocking"},
        )
    )
    obs = env.step(
        FTAction(tool_name="list_escalations", arguments={"ticket_id": "TKT-9001"})
    )
    assert "blocking" in obs.feedback


def test_referential_check_rejects_an_unknown_parent(tcaas, graders):
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=False)
    env.reset(seed=0, split="train")
    obs = env.step(
        FTAction(
            tool_name="escalate_ticket",
            arguments={"ticket_id": "TKT-0000", "reason": "typo"},
        )
    )
    assert not obs.done
    assert "TKT-0000" in obs.feedback


def test_the_new_worlds_rubrics_actually_score(tcaas, graders):
    """The silent failure this replaced: unknown rubrics scoring 0.0 forever."""
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    env.reset(seed=0, split="train")
    obs = env.step(FTAction(tool_name="list_open_tickets", arguments={}))
    env.step(
        FTAction(
            tool_name="escalate_ticket",
            arguments={"ticket_id": "TKT-9001", "reason": "blocking"},
        )
    )
    obs = env.step(
        FTAction(
            tool_name="submit_answer",
            arguments={"answer": "TKT-9001 is critical and blocking; work it first."},
        )
    )
    assert obs.reward == pytest.approx(1.0)
    assert {r["rubric_id"] for r in obs.metadata["individual_results"]} == {
        "cites-tickets",
        "correct-queue",
        "ranked-by-severity",
        "escalated-blockers",
    }


def test_a_wrong_answer_still_scores_zero(tcaas, graders):
    """The outcome gate must follow the new world's `outcome` declarations."""
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    env.reset(seed=0, split="train")
    env.step(FTAction(tool_name="list_open_tickets", arguments={}))
    obs = env.step(
        FTAction(
            tool_name="submit_answer",
            arguments={"answer": "Everything is low severity, nothing to do."},
        )
    )
    assert obs.reward == 0.0
