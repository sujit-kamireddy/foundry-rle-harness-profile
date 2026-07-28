"""The harness profile must agree with the catalog and the env.

These are drift tests: the profile is hand-written, so nothing but a test stops
it describing an env that no longer exists.
"""

import json
import pathlib
import unittest

from m365_number_guess_v2.graders.aggregation import PASS_THRESHOLD
from m365_number_guess_v2.logic import MAX_STEPS
from m365_number_guess_v2.tcaas.catalog import Catalog

ENV_ROOT = pathlib.Path(__file__).resolve().parents[1] / "envs" / "m365_number_guess_v2"
PROFILE = json.loads((ENV_ROOT / "harness-profile.json").read_text())


class ProfileTests(unittest.TestCase):
    def test_eval_limit_matches_the_validation_split(self):
        """``limit`` bounds the eval run, so it must cover the split exactly once."""
        split = PROFILE["evalDefaults"]["split"]
        self.assertEqual(
            Catalog.load().split_size(split), PROFILE["evalDefaults"]["limit"]
        )

    def test_declared_splits_exist_in_the_catalog(self):
        catalog = Catalog.load()
        for section in ("evalDefaults", "trainingDefaults"):
            self.assertIn(PROFILE[section]["split"], catalog.tasks)

    def test_success_thresholds_match_the_graders_pass_mark(self):
        """Grader-passed and harness-success must agree, or eval numbers diverge."""
        self.assertEqual(PASS_THRESHOLD, PROFILE["reward"]["successRewardThreshold"])
        self.assertEqual(
            PASS_THRESHOLD,
            PROFILE["episodeCompletion"]["successCondition"]["rewardThreshold"],
        )

    def test_step_cap_matches_the_env(self):
        self.assertEqual(MAX_STEPS, PROFILE["limits"]["maxStepsPerEpisode"])

    def test_prompt_path_points_at_a_real_observation_field(self):
        self.assertEqual("prompt", PROFILE["observationRendering"]["promptPath"])
        self.assertEqual("feedback", PROFILE["observationRendering"]["feedbackPath"])

    def test_declared_actions_use_the_generic_tool_shape(self):
        for action in PROFILE["actionSpace"]["actions"]:
            properties = action["inputSchema"]["properties"]
            self.assertEqual(action["name"], properties["tool_name"]["const"])
            self.assertIn("arguments", properties)

    def test_terminal_action_is_the_env_terminal_tool(self):
        self.assertEqual(["guess"], PROFILE["actionSpace"]["terminalActions"])


if __name__ == "__main__":
    unittest.main()
