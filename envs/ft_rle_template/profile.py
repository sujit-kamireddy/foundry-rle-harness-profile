"""Render ``harness-profile.json`` from an FT world.

This closes the gap the v2 reference calls out: *"harness-profile.json is
static. In production TCaaS would render it per deployment, since it knows a
user's skills and tools before the container exists."*

The FT backend runs this when a world changes - a user adds a tool, edits a
skill, uploads samples - and republishes the profile. That is the whole sync
story: a world edit re-renders a JSON document, it does not rebuild an image.

What maps to what
-----------------
==========================  ==================================================
FT world                    harness profile
==========================  ==================================================
registered MCP tools        ``actionSpace.actions`` / ``modelActions``
base terminal tool          ``actionSpace.terminalActions``
samples by type             ``evalDefaults.split`` / ``trainingDefaults.split``
sample counts               ``evalDefaults.limit``
rubric pass mark            ``reward.successRewardThreshold``
==========================  ==================================================

Skill *instructions* are deliberately absent: they vary per episode, so they
belong in the observation prompt, not in this static document.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from .config import (
    EPISODE_TIMEOUT_S,
    MAX_STEPS_PER_EPISODE,
    SUCCESS_THRESHOLD,
)
from .logic import MODEL_INSTRUCTIONS, TERMINAL_TOOL
from .tcaas.models import ToolSpec, WorldDescriptor
from .tools.local import SUBMIT_ANSWER_SPEC

SCHEMA_VERSION = "rle.harness/v0.1"
PROFILE_ID = "default"

EVAL_SPLIT = "validation"
TRAIN_SPLIT = "train"
ROLLOUTS_PER_CASE_TRAIN = 8
ROLLOUTS_PER_CASE_EVAL = 1


def _envelope(spec: ToolSpec) -> Dict[str, Any]:
    """Wrap a tool's own schema in the ``{tool_name, arguments}`` action envelope.

    The runtime takes one action shape for every tool, so the profile advertises
    the envelope and pins ``tool_name`` with a const discriminator.
    """
    return {
        "name": spec.tool_name,
        "description": spec.description,
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "const": spec.tool_name},
                "arguments": spec.input_schema,
            },
            "required": ["tool_name", "arguments"],
            "additionalProperties": False,
        },
    }


def render_harness_profile(
    world: WorldDescriptor,
    *,
    profile_id: str = PROFILE_ID,
    max_steps: int = MAX_STEPS_PER_EPISODE,
    timeout_s: int = EPISODE_TIMEOUT_S,
    success_threshold: float = SUCCESS_THRESHOLD,
) -> Dict[str, Any]:
    """Build the profile document for one world version."""
    specs: List[ToolSpec] = list(world.tools) + [SUBMIT_ANSWER_SPEC]
    names = [spec.tool_name for spec in specs]

    eval_limit = world.split_sizes.get(EVAL_SPLIT)

    profile: Dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "id": profile_id,
        "runtime": {"kind": "openenv", "contract": "gym"},
        "observationRendering": {
            "type": "default_json",
            "promptPath": "prompt",
            "feedbackPath": "feedback",
            "instructions": MODEL_INSTRUCTIONS,
        },
        "actionSpace": {
            "source": "schema",
            "schemaRequirement": "required",
            "discoveredActions": names,
            "modelActions": [
                {
                    "actionName": spec.tool_name,
                    "toolName": spec.tool_name,
                    "toolDescription": spec.description,
                }
                for spec in specs
            ],
            "terminalActions": [TERMINAL_TOOL],
            "graderActions": [],
            "ignoredActions": [],
            "actions": [_envelope(spec) for spec in specs],
        },
        "episodeCompletion": {
            "mode": "open_ended_tool",
            "stopCondition": {"type": "done"},
            "successCondition": {
                "type": "reward_at_least",
                "rewardThreshold": success_threshold,
            },
            "maxStepsBehavior": "truncate",
        },
        "reward": {
            "aggregation": "final",
            "successCondition": "reward_at_least",
            "successRewardThreshold": success_threshold,
        },
        "limits": {
            "maxStepsPerEpisode": max_steps,
            "timeoutSeconds": timeout_s,
        },
        "evalDefaults": {
            "split": EVAL_SPLIT,
            "rolloutsPerCase": ROLLOUTS_PER_CASE_EVAL,
            "limit": eval_limit,
        },
        "trainingDefaults": {
            "split": TRAIN_SPLIT,
            "rolloutsPerCase": ROLLOUTS_PER_CASE_TRAIN,
        },
        "notes": (
            f"Rendered from FT world {world.world_id} at content version "
            f"{world.content_version}."
        ),
    }
    return profile


def render_from_catalog() -> Dict[str, Any]:
    """Offline path: render straight from the bundled catalog, no HTTP."""
    from .tcaas.catalog import load_catalog

    return render_harness_profile(load_catalog().descriptor())


def render_from_tcaas() -> Dict[str, Any]:
    """Production path: read the world from TCaaS and render."""
    from .tcaas.client import TCaaSClient

    return render_harness_profile(TCaaSClient().world())


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an FT harness profile.")
    parser.add_argument("--out", default="-", help="Output path, or - for stdout.")
    parser.add_argument(
        "--source",
        choices=("catalog", "tcaas"),
        default="catalog",
        help="Read the world from the bundled catalog or from TCaaS.",
    )
    args = parser.parse_args()

    profile = render_from_tcaas() if args.source == "tcaas" else render_from_catalog()
    text = json.dumps(profile, indent=2) + "\n"

    if args.out == "-":
        print(text, end="")
    else:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
