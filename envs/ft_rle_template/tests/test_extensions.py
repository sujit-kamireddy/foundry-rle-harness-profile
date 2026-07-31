"""The claim under test: a world can add a tool and a rubric without a fork.

The template covers most FT worlds with data alone - a tool is a ``serves``
binding, a rubric is one of four generic checks. Two things it cannot express
declaratively are a tool that *computes* from the episode's hidden task data,
and a rubric that scores the trajectory rather than the text. Both arrive as
module paths rather than as edits to this package.

These tests run against ``fixtures/world_ext.json`` plus ``world_ext.py``, a
world module that imports nothing from ``ft_rle_template``. That independence is
the property worth protecting: if it breaks, every dropped-in world becomes
coupled to this template's internals.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import pytest

from ft_rle_template.extensions import WorldExtensionError, registry
from ft_rle_template.graders import rubrics
from ft_rle_template.graders.models import Item, RubricRef, Sample
from ft_rle_template.models import FTAction
from ft_rle_template.server.ft_environment import FTEnvironment
from ft_rle_template.tcaas import tools as mock_tools
from ft_rle_template.tcaas.catalog import load_catalog
from ft_rle_template.tools.base import ToolTransportError

from .conftest import FIXTURES

WORLD_EXT = FIXTURES / "world_ext.json"
EXT_MODULE = "ft_rle_template.tests.world_ext"

pytestmark = pytest.mark.world(str(WORLD_EXT))


@pytest.fixture
def extensions() -> Iterator[None]:
    """Point both hooks at the fixture module, as the container's env would."""
    os.environ["FT_WORLD_TOOLS"] = EXT_MODULE
    os.environ["FT_WORLD_CHECKS"] = EXT_MODULE
    try:
        yield
    finally:
        os.environ.pop("FT_WORLD_TOOLS", None)
        os.environ.pop("FT_WORLD_CHECKS", None)


@pytest.fixture
def checks_only() -> Iterator[None]:
    """Checks hook set, tools hook not. Enough to boot, not enough to act."""
    os.environ["FT_WORLD_CHECKS"] = EXT_MODULE
    os.environ.pop("FT_WORLD_TOOLS", None)
    try:
        yield
    finally:
        os.environ.pop("FT_WORLD_CHECKS", None)


def _sample(calls: list[str], answer: str = "done") -> Sample:
    return Sample(
        output_text=answer,
        output_trajectory=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": name, "arguments": "{}"}} for name in calls
                ],
            }
        ],
    )


def _item(expected: dict | None = None) -> Item:
    return Item(input="What is the secret?", expected=json.dumps(expected or {}))


def test_world_tool_answers_from_task_data(extensions, tcaas, graders):
    """The whole point: a tool reaching data no `serves` binding could reach.

    ``secret`` lives in the task, never in a dataset, and never in the sandbox.
    """
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    env.reset(seed=0, split="train")
    obs = env.step(FTAction(tool_name="echo_secret", arguments={"prefix": "saw"}))
    assert "saw:swordfish" in obs.feedback
    assert not obs.done


def test_world_tool_bad_argument_is_feedback_not_an_outage(extensions, tcaas, graders):
    """A plain ValueError from a world module must land as a 400, not a 500.

    This is what lets a world module stay dependency-free: it does not import a
    template exception class to get the rejection semantics right.
    """
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    env.reset(seed=0, split="train")
    obs = env.step(FTAction(tool_name="echo_secret", arguments={"prefix": 42}))
    assert "requires a string" in obs.feedback
    assert not obs.done


def test_world_tool_is_unreachable_without_the_env_var(checks_only, tcaas, graders):
    """No silent fallback: an unconfigured tools hook fails loudly at the call.

    It raises rather than becoming feedback, and that split is the contract: a
    bad *argument* is something the policy should learn from, but a tool the
    deployment cannot serve at all is a misconfiguration, and turning it into
    feedback would let a broken world train to completion on a flat reward.

    The catalog still loads here - only ``FT_WORLD_CHECKS`` is set - so this
    isolates the tool hook from the fail-at-load path below.
    """
    env = FTEnvironment(tcaas=tcaas, graders=graders, training=True)
    env.reset(seed=0, split="train")
    with pytest.raises(ToolTransportError, match="501"):
        env.step(FTAction(tool_name="echo_secret", arguments={"prefix": "saw"}))


def test_world_check_scores_the_trajectory(extensions):
    rubric = RubricRef(
        rubric_id="thorough",
        weight=1.0,
        check="call_count",
        check_params={"target_calls": 4},
    )
    results = rubrics.score_all([rubric], _item(), _sample(["echo_secret"] * 2))
    assert results[0].score == 0.5


def test_world_check_is_rejected_without_its_required_params(extensions):
    """A world's own REQUIRED_PARAMS gets the same enforcement as a built-in."""
    reason = rubrics.unscorable_reason("call_count", {})
    assert reason is not None and "target_calls" in reason


def test_world_check_returning_an_illegal_score_raises(extensions):
    """Weights only mean something if every rubric is on the same [0, 1] scale."""
    rubric = RubricRef(rubric_id="broken", weight=1.0, check="out_of_range")
    with pytest.raises(rubrics.RubricNotScorable, match="outside"):
        rubrics.score_all([rubric], _item(), _sample([]))


def test_world_check_is_unknown_without_the_env_var():
    assert rubrics.unscorable_reason("call_count", {"target_calls": 4}) is not None


def test_catalog_accepts_a_world_check_it_can_resolve(extensions):
    """Fail-at-load must consult the merged vocabulary, not just the built-ins.

    Without this the two hooks would contradict each other: the grader would
    score a rubric the catalog validator had already refused to load.
    """
    assert load_catalog().world_id == "world-ext"


def test_catalog_rejects_a_world_check_it_cannot_resolve(tmp_path):
    """Same catalog, hook unset. The world must not boot."""
    world = json.loads(WORLD_EXT.read_text())
    path = tmp_path / "world.json"
    path.write_text(json.dumps(world))

    previous = os.environ["FT_CATALOG_PATH"]
    os.environ["FT_CATALOG_PATH"] = str(path)
    mock_tools.reset_store()
    try:
        with pytest.raises(ValueError, match="call_count"):
            load_catalog()
    finally:
        os.environ["FT_CATALOG_PATH"] = previous


def test_a_missing_extension_module_fails_loudly():
    """Serving a world with its tools quietly missing is the failure to avoid."""
    with pytest.raises(WorldExtensionError, match="could not import"):
        registry("ft_rle_template.tests.no_such_module", "TOOLS")


def test_no_extension_configured_is_not_an_error():
    assert registry("", "TOOLS") == {}
