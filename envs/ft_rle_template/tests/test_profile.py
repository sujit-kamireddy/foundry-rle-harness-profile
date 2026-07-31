"""The profile is generated, so the test that matters is drift against the world.

This module runs against the *shipped* catalog, because it guards the checked-in
``harness-profile.json``. Every other test uses a fixture world.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ft_rle_template.config import MAX_STEPS_PER_EPISODE, SUCCESS_THRESHOLD
from ft_rle_template.logic import MODEL_INSTRUCTIONS, TERMINAL_TOOL
from ft_rle_template.profile import render_from_catalog, render_harness_profile
from ft_rle_template.tcaas.catalog import load_catalog

from .conftest import SHIPPED_WORLD

pytestmark = pytest.mark.world(str(SHIPPED_WORLD))

CHECKED_IN = Path(__file__).resolve().parents[1] / "harness-profile.json"


def test_checked_in_profile_matches_renderer():
    """The committed file is renderer output, not something hand-edited.

    If this fails after dropping in a new catalog, re-render it - that is the
    build step, and the failure is the reminder.
    """
    assert json.loads(CHECKED_IN.read_text()) == render_from_catalog()


def test_every_world_tool_is_an_action():
    world = load_catalog().descriptor()
    profile = render_harness_profile(world)
    declared = set(profile["actionSpace"]["discoveredActions"])
    assert {t.tool_name for t in world.tools} <= declared
    assert TERMINAL_TOOL in declared


def test_actions_use_the_tool_call_envelope():
    """Every action must be `{tool_name, arguments}` or the runtime can't parse it."""
    profile = render_from_catalog()
    for action in profile["actionSpace"]["actions"]:
        schema = action["inputSchema"]
        assert schema["required"] == ["tool_name", "arguments"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["tool_name"]["const"] == action["name"]


def test_only_the_base_tool_is_terminal():
    """A user tool must never be able to end an episode."""
    profile = render_from_catalog()
    assert profile["actionSpace"]["terminalActions"] == [TERMINAL_TOOL]


def test_eval_limit_tracks_the_validation_split():
    """Adding validation samples must widen eval without a code change."""
    world = load_catalog().descriptor()
    profile = render_harness_profile(world)
    assert profile["evalDefaults"]["limit"] == world.split_sizes["validation"]
    assert profile["evalDefaults"]["split"] in world.split_sizes


def test_thresholds_and_limits_come_from_config():
    """Profile and runtime must agree on success, or reports contradict rewards."""
    profile = render_from_catalog()
    assert profile["reward"]["successRewardThreshold"] == SUCCESS_THRESHOLD
    assert (
        profile["episodeCompletion"]["successCondition"]["rewardThreshold"]
        == SUCCESS_THRESHOLD
    )
    assert profile["limits"]["maxStepsPerEpisode"] == MAX_STEPS_PER_EPISODE


def test_instructions_are_static_protocol_only():
    """Skill text is per-episode content; it must not leak into the static profile."""
    profile = render_from_catalog()
    assert profile["observationRendering"]["instructions"] == MODEL_INSTRUCTIONS
    world = load_catalog().descriptor()
    rendered = json.dumps(profile)
    for skill in world.skills:
        assert skill.workflow not in rendered


def test_profile_pins_observation_paths():
    rendering = render_from_catalog()["observationRendering"]
    assert rendering["promptPath"] == "prompt"
    assert rendering["feedbackPath"] == "feedback"


def test_profile_records_the_content_version():
    """An RLE version has to be traceable to the world content it was built from."""
    world = load_catalog().descriptor()
    assert world.content_version in render_harness_profile(world)["notes"]
