import pathlib
import sys
import unittest


ENVS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "envs"
sys.path.insert(0, str(ENVS_ROOT))

from m365_number_guess_env.grading import distance_score, grade, optimal_steps
from m365_number_guess_env.logic import MAX_STEPS, compare_result

try:
    from m365_number_guess_env.models import NumberGuessAction
    from m365_number_guess_env.server.number_guess_environment import NumberGuessEnvironment
except ImportError:
    NumberGuessAction = None
    NumberGuessEnvironment = None


class NumberGuessLogicTests(unittest.TestCase):
    def test_compare_reports_direction(self):
        self.assertEqual("higher", compare_result(5, target=7))
        self.assertEqual("lower", compare_result(9, target=7))
        self.assertEqual("equal", compare_result(7, target=7))

    def test_distance_score_bounds(self):
        self.assertEqual(1.0, distance_score(7, target=7))
        self.assertGreater(distance_score(6, target=7), distance_score(1, target=7))
        self.assertGreaterEqual(distance_score(1, target=10), 0.0)

    def test_optimal_solve_scores_one(self):
        reward, feedback = grade(solved=True, steps_used=optimal_steps(), number=7, target=7)
        self.assertEqual(1.0, reward)
        self.assertIn("Solved", feedback)

    def test_slow_solve_scores_lower(self):
        fast, _ = grade(solved=True, steps_used=optimal_steps(), number=7, target=7)
        slow, _ = grade(solved=True, steps_used=optimal_steps() + 5, number=7, target=7)
        self.assertLess(slow, fast)
        self.assertGreaterEqual(slow, 0.5)

    def test_unsolved_capped_below_solve(self):
        reward, _ = grade(solved=False, steps_used=10, number=6, target=7)
        self.assertLess(reward, 0.5)


@unittest.skipIf(NumberGuessEnvironment is None, "openenv package is not installed")
class NumberGuessEnvironmentTests(unittest.TestCase):
    def test_optimal_solve_gets_full_reward(self):
        env = NumberGuessEnvironment()
        env.reset(seed=123, episode_id="episode-1")
        target = env._target  # noqa: SLF001 - test needs the hidden target

        # Localize with binary-search probes, then commit.
        low, high = 1, 10
        while low < high:
            mid = (low + high) // 2
            obs = env.step(NumberGuessAction(action_type="compare", number=mid))
            self.assertFalse(obs.done)
            if "higher" in obs.feedback:
                low = mid + 1
            elif "lower" in obs.feedback:
                high = mid
            else:
                low = high = mid

        result = env.step(NumberGuessAction(action_type="guess", number=low))
        self.assertEqual(low, target)
        self.assertTrue(result.done)
        self.assertTrue(result.metadata["success"])
        self.assertEqual(1.0, result.reward)

    def test_wrong_guess_continues_without_reward(self):
        env = NumberGuessEnvironment()
        env.reset(seed=1, episode_id="episode-2")
        wrong = env._target % 10 + 1  # noqa: SLF001 - guaranteed to differ from target

        obs = env.step(NumberGuessAction(action_type="guess", number=wrong))

        self.assertFalse(obs.done)
        self.assertEqual(0.0, obs.reward)
        self.assertNotIn("higher", obs.feedback)
        self.assertNotIn("lower", obs.feedback)

    def test_out_of_steps_terminates_with_terminal_grade(self):
        env = NumberGuessEnvironment()
        env.reset(seed=1, episode_id="episode-3")
        wrong = env._target % 10 + 1  # noqa: SLF001 - guaranteed to differ from target

        obs = None
        for _ in range(MAX_STEPS):
            obs = env.step(NumberGuessAction(action_type="guess", number=wrong))

        self.assertTrue(obs.done)
        self.assertFalse(obs.metadata["success"])
        self.assertLess(obs.reward, 0.5)
        self.assertGreaterEqual(obs.reward, 0.0)


if __name__ == "__main__":
    unittest.main()
