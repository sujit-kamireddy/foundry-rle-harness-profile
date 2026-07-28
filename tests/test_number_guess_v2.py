"""v2 env: scoring math, prompt composition, and the TCaaS-driven episode loop.

The environment tests need OpenEnv, which only exists in the container image, so
they skip locally; everything below them runs anywhere.
"""

import unittest

import httpx
from fastapi.testclient import TestClient

from m365_number_guess_v2.graders.app import create_graders_app
from m365_number_guess_v2.graders.client import GraderClient, GradingUnavailable
from m365_number_guess_v2.grading import grade, optimal_steps
from m365_number_guess_v2.logic import MAX_STEPS, compose_prompt
from m365_number_guess_v2.tcaas.app import create_tcaas_app
from m365_number_guess_v2.tcaas.catalog import Catalog
from m365_number_guess_v2.tcaas.client import TCaaSClient

try:
    from m365_number_guess_v2.models import NumberGuessAction
    from m365_number_guess_v2.server.number_guess_environment import (
        NumberGuessEnvironment,
    )
except ImportError:
    NumberGuessAction = None
    NumberGuessEnvironment = None


def make_env():
    """An env wired to both mocks over in-process HTTP."""
    catalog = Catalog.load()
    return NumberGuessEnvironment(
        tcaas=TCaaSClient(tenant=catalog.tenant, http=TestClient(create_tcaas_app())),
        graders=GraderClient(http=TestClient(create_graders_app())),
    )


class GradingTests(unittest.TestCase):
    def test_optimal_solve_scores_one(self):
        self.assertEqual(1.0, grade(solved=True, steps_used=optimal_steps()))

    def test_slow_solve_scores_lower_but_still_passes(self):
        fast = grade(solved=True, steps_used=optimal_steps())
        slow = grade(solved=True, steps_used=optimal_steps() + 5)
        self.assertLess(slow, fast)
        self.assertGreaterEqual(slow, 0.5)

    def test_a_wrong_answer_earns_nothing(self):
        """Correct must always outrank wrong, however good the process was."""
        self.assertEqual(0.0, grade(solved=False, steps_used=2))
        self.assertEqual(0.0, grade(solved=False, steps_used=optimal_steps()))


class PromptCompositionTests(unittest.TestCase):
    def test_prompt_carries_both_skill_and_query(self):
        prompt = compose_prompt("Do the thing.", "Find my number.")
        self.assertIn("Do the thing.", prompt)
        self.assertIn("Find my number.", prompt)


@unittest.skipIf(NumberGuessEnvironment is None, "openenv package is not installed")
class EnvironmentTests(unittest.TestCase):
    def test_reset_serves_the_task_selected_by_the_seed(self):
        env = make_env()
        obs = env.reset(seed=3, split="train", episode_id="ep-1")

        expected = Catalog.load().pick_task("train", 3)
        self.assertEqual(expected.task_id, obs.metadata["task_id"])
        self.assertIn(expected.user_query, obs.prompt)
        self.assertIn(expected.skill, obs.prompt)

    def test_same_seed_gives_identical_model_visible_fields(self):
        """The GRPO group contract: one seed, one task, K trajectories."""
        first = make_env().reset(seed=42, split="train", episode_id="ep-a")
        second = make_env().reset(seed=42, split="train", episode_id="ep-b")

        visible = ("prompt", "skill", "user_query")
        for field in visible:
            self.assertEqual(getattr(first, field), getattr(second, field))
        self.assertNotEqual(first.metadata["episode_id"], second.metadata["episode_id"])

    def test_split_falls_back_to_the_profile_default(self):
        obs = make_env().reset(seed=0, episode_id="ep-2")
        self.assertEqual("train", obs.metadata["split"])

    def test_tool_surface_combines_base_and_user_tools(self):
        obs = make_env().reset(seed=0, split="train", episode_id="ep-3")
        names = {tool["function"]["name"] for tool in obs.tools}
        self.assertEqual({"guess", "compare"}, names)

    def test_probe_then_commit_ends_the_episode(self):
        env = make_env()
        env.reset(seed=0, split="train", episode_id="ep-4")
        target = Catalog.load().pick_task("train", 0).data["target"]

        probe = env.step(
            NumberGuessAction(tool_name="compare", arguments={"number": target})
        )
        self.assertFalse(probe.done)
        self.assertIn("equal", probe.feedback)

        commit = env.step(
            NumberGuessAction(tool_name="guess", arguments={"number": target})
        )
        self.assertTrue(commit.done)
        self.assertEqual(target, env.state.committed_answer)

    def test_a_correct_solve_is_rewarded_and_marked_successful(self):
        env = make_env()
        env.reset(seed=0, split="train", episode_id="ep-reward")
        target = Catalog.load().pick_task("train", 0).data["target"]

        env.step(NumberGuessAction(tool_name="compare", arguments={"number": 5}))
        obs = env.step(
            NumberGuessAction(tool_name="guess", arguments={"number": target})
        )

        self.assertGreaterEqual(obs.reward, 0.5)
        self.assertTrue(obs.metadata["success"])

    def test_a_wrong_solve_falls_below_the_success_threshold(self):
        env = make_env()
        env.reset(seed=0, split="train", episode_id="ep-wrong")
        target = Catalog.load().pick_task("train", 0).data["target"]

        env.step(NumberGuessAction(tool_name="compare", arguments={"number": 5}))
        obs = env.step(
            NumberGuessAction(tool_name="guess", arguments={"number": target % 10 + 1})
        )

        self.assertLess(obs.reward, 0.5)
        self.assertFalse(obs.metadata["success"])

    def test_every_rubric_is_reported_in_metadata(self):
        env = make_env()
        env.reset(seed=0, split="train", episode_id="ep-rubrics")
        target = Catalog.load().pick_task("train", 0).data["target"]

        env.step(NumberGuessAction(tool_name="compare", arguments={"number": 5}))
        obs = env.step(
            NumberGuessAction(tool_name="guess", arguments={"number": target})
        )

        reported = {r["rubric_id"] for r in obs.metadata["individual_results"]}
        self.assertEqual({"efficient-solve", "probe-before-commit"}, reported)

    def test_an_unreachable_grader_errors_rather_than_rewarding_zero(self):
        """A dead grader is an infrastructure fault, never a policy score."""
        catalog = Catalog.load()
        env = NumberGuessEnvironment(
            tcaas=TCaaSClient(
                tenant=catalog.tenant, http=TestClient(create_tcaas_app())
            ),
            graders=GraderClient(
                http=httpx.Client(
                    base_url="http://graders",
                    transport=httpx.MockTransport(lambda request: httpx.Response(503)),
                )
            ),
        )
        env.reset(seed=0, split="train", episode_id="ep-dead")

        with self.assertRaises(GradingUnavailable):
            env.step(NumberGuessAction(tool_name="guess", arguments={"number": 1}))

    def test_committing_a_wrong_answer_still_ends_the_episode(self):
        """The sandbox never learns the target, so any accepted commit is terminal."""
        env = make_env()
        env.reset(seed=0, split="train", episode_id="ep-5")
        target = Catalog.load().pick_task("train", 0).data["target"]
        wrong = target % 10 + 1

        obs = env.step(
            NumberGuessAction(tool_name="guess", arguments={"number": wrong})
        )

        self.assertTrue(obs.done)
        self.assertEqual(wrong, env.state.committed_answer)

    def test_rejected_commit_is_feedback_not_termination(self):
        env = make_env()
        env.reset(seed=0, split="train", episode_id="ep-6")

        obs = env.step(NumberGuessAction(tool_name="guess", arguments={"number": 999}))

        self.assertFalse(obs.done)
        self.assertIsNone(env.state.committed_answer)

    def test_unknown_tool_is_feedback_not_a_crash(self):
        env = make_env()
        env.reset(seed=0, split="train", episode_id="ep-7")

        obs = env.step(NumberGuessAction(tool_name="teleport", arguments={}))

        self.assertFalse(obs.done)
        self.assertIn("Unknown tool", obs.feedback)

    def test_step_cap_truncates_the_episode(self):
        env = make_env()
        env.reset(seed=0, split="train", episode_id="ep-8")

        obs = None
        for _ in range(MAX_STEPS):
            obs = env.step(
                NumberGuessAction(tool_name="compare", arguments={"number": 1})
            )

        self.assertTrue(obs.done)
        self.assertEqual(MAX_STEPS, obs.step)

    def test_stepping_after_the_episode_ends_is_refused(self):
        env = make_env()
        env.reset(seed=0, split="train", episode_id="ep-9")
        env.step(NumberGuessAction(tool_name="guess", arguments={"number": 1}))

        with self.assertRaises(RuntimeError):
            env.step(NumberGuessAction(tool_name="guess", arguments={"number": 2}))

    def test_step_before_reset_is_refused(self):
        with self.assertRaises(RuntimeError):
            make_env().step(
                NumberGuessAction(tool_name="guess", arguments={"number": 1})
            )

    def test_concurrent_sessions_do_not_leak_state(self):
        """Interleaved episodes must keep separate tasks and step counters."""
        left, right = make_env(), make_env()
        left_obs = left.reset(seed=0, split="train", episode_id="ep-left")
        right_obs = right.reset(seed=1, split="train", episode_id="ep-right")
        self.assertNotEqual(left_obs.metadata["task_id"], right_obs.metadata["task_id"])

        left.step(NumberGuessAction(tool_name="compare", arguments={"number": 1}))
        left.step(NumberGuessAction(tool_name="compare", arguments={"number": 2}))
        right_step = right.step(
            NumberGuessAction(tool_name="compare", arguments={"number": 3})
        )

        self.assertEqual(2, left.state.step_count)
        self.assertEqual(1, right_step.step)
        self.assertEqual(right_obs.metadata["task_id"], right.state.task_id)


if __name__ == "__main__":
    unittest.main()
