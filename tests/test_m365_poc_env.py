import pathlib
import sys
import unittest


ENV_ROOT = pathlib.Path(__file__).resolve().parents[1] / "envs" / "m365_poc_env"
sys.path.insert(0, str(ENV_ROOT))

from logic import evaluate_message

try:
    from models import M365HelloWorldAction
    from server.m365_poc_environment import M365HelloWorldEnvironment
except ImportError:
    M365HelloWorldAction = None
    M365HelloWorldEnvironment = None


class M365HelloWorldTests(unittest.TestCase):
    def test_success_requires_hello_and_m365(self):
        success, feedback = evaluate_message("Hello from M365 and Foundry RLE!")

        self.assertTrue(success)
        self.assertIn("Accepted", feedback)

    def test_missing_m365_fails(self):
        success, feedback = evaluate_message("Hello from Foundry RLE")

        self.assertFalse(success)
        self.assertIn("M365", feedback)

    @unittest.skipIf(M365HelloWorldEnvironment is None, "openenv package is not installed")
    def test_openenv_environment_step_success(self):
        env = M365HelloWorldEnvironment()

        initial = env.reset(episode_id="episode-1")
        result = env.step(M365HelloWorldAction(message="Hello from M365 and Foundry RLE!"))

        self.assertFalse(initial.done)
        self.assertTrue(result.done)
        self.assertEqual(1.0, result.reward)
        self.assertTrue(result.metadata["success"])


if __name__ == "__main__":
    unittest.main()