"""Mock tc_graders: rubric scoring, aggregation, and the score -> reward rule."""

import json
import unittest

import httpx
from fastapi.testclient import TestClient

from m365_number_guess_v2.graders.aggregation import aggregate
from m365_number_guess_v2.graders.app import create_graders_app
from m365_number_guess_v2.graders.client import (
    GraderClient,
    GradingUnavailable,
    to_reward,
)
from m365_number_guess_v2.graders.models import (
    GradeRequest,
    GradeResponse,
    Item,
    RubricRef,
    Sample,
)
from m365_number_guess_v2.graders.trajectory import TrajectoryRecorder, read_tool_calls

RUBRICS = [
    RubricRef(rubric_id="efficient-solve", weight=1.0),
    RubricRef(rubric_id="probe-before-commit", weight=0.5),
]


def trajectory(calls, prompt="prompt"):
    """Build a trajectory from ``(tool_name, arguments)`` pairs."""
    recorder = TrajectoryRecorder("ep-test", prompt)
    for index, (tool_name, arguments) in enumerate(calls, start=1):
        recorder.record(tool_name, arguments, f"call-{index}", "ok")
    return recorder.build()


def request_for(calls, expected="7", rubrics=None):
    return GradeRequest(
        rubrics=rubrics if rubrics is not None else RUBRICS,
        item=Item(input="prompt", expected=expected),
        sample=Sample(output_text="{}", output_trajectory=trajectory(calls)),
    )


class TrajectoryTests(unittest.TestCase):
    def test_each_call_gets_a_matched_tool_result(self):
        built = trajectory([("compare", {"number": 5})])
        roles = [m.role for m in built.messages]
        self.assertEqual(["user", "assistant", "tool"], roles)
        self.assertEqual(
            built.messages[1].tool_calls[0]["id"], built.messages[2].tool_call_id
        )

    def test_arguments_round_trip(self):
        built = trajectory([("compare", {"number": 5}), ("guess", {"number": 7})])
        self.assertEqual(
            [("compare", {"number": 5}), ("guess", {"number": 7})],
            read_tool_calls(built),
        )

    def test_arguments_are_json_encoded_like_openai(self):
        built = trajectory([("compare", {"number": 5})])
        raw = built.messages[1].tool_calls[0]["function"]["arguments"]
        self.assertEqual({"number": 5}, json.loads(raw))


class AggregationTests(unittest.TestCase):
    def test_weighted_mean_favours_the_heavier_rubric(self):
        self.assertAlmostEqual(0.8, aggregate([1.0, 0.4], [1.0, 0.5], "weighted_mean"))

    def test_strategies_differ(self):
        self.assertEqual(0.4, aggregate([1.0, 0.4], [1.0, 1.0], "min"))
        self.assertEqual(1.0, aggregate([1.0, 0.4], [1.0, 1.0], "max"))
        self.assertAlmostEqual(0.7, aggregate([1.0, 0.4], [1.0, 1.0], "mean"))

    def test_unknown_strategy_is_an_error(self):
        with self.assertRaises(ValueError):
            aggregate([1.0], [1.0], "vibes")


class GradeEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_graders_app())

    def _grade(self, request):
        response = self.client.post("/grade", json=request.model_dump())
        self.assertEqual(200, response.status_code)
        return GradeResponse(**response.json())

    def test_efficient_probing_solve_beats_a_blind_guess(self):
        probed = self._grade(
            request_for([("compare", {"number": 5}), ("guess", {"number": 7})])
        )
        blind = self._grade(request_for([("guess", {"number": 7})]))
        self.assertGreater(probed.score, blind.score)

    def test_wrong_answer_scores_below_the_pass_mark(self):
        response = self._grade(
            request_for([("compare", {"number": 5}), ("guess", {"number": 2})])
        )
        self.assertLess(response.score, 0.5)
        self.assertFalse(response.passed)

    def test_correct_always_outranks_wrong_whatever_the_process(self):
        """The invariant the 0.5 threshold depends on: the two bands cannot overlap."""
        best_wrong = self._grade(
            request_for([("compare", {"number": 5}), ("guess", {"number": 2})])
        )
        worst_correct = self._grade(request_for([("guess", {"number": 7})] * 1))
        self.assertLess(best_wrong.score, 0.5)
        self.assertGreaterEqual(worst_correct.score, 0.5)

    def test_every_rubric_reports_individually(self):
        response = self._grade(
            request_for([("compare", {"number": 5}), ("guess", {"number": 7})])
        )
        reported = {
            r["rubric_id"] for r in response.extra_outputs["individual_results"]
        }
        self.assertEqual({"efficient-solve", "probe-before-commit"}, reported)

    def test_rubrics_disagree_when_the_policy_skips_probing(self):
        """The two rubrics read different things, so a blind solve splits them."""
        response = self._grade(request_for([("guess", {"number": 7})]))
        by_id = {
            r["rubric_id"]: r for r in response.extra_outputs["individual_results"]
        }
        self.assertTrue(by_id["efficient-solve"]["passed"])
        self.assertEqual(0.0, by_id["probe-before-commit"]["score"])

    def test_never_committing_scores_zero_rather_than_failing(self):
        """Running out of steps is a policy outcome, not an infrastructure fault."""
        response = self._grade(request_for([("compare", {"number": 5})]))
        self.assertIsNotNone(response.score)
        self.assertLess(response.score, 0.5)

    def test_missing_trajectory_fails(self):
        request = GradeRequest(
            rubrics=RUBRICS, item=Item(input="p", expected="7"), sample=Sample()
        )
        response = self._grade(request)
        self.assertIsNone(response.score)

    def test_missing_expected_fails(self):
        response = self._grade(request_for([("guess", {"number": 7})], expected=None))
        self.assertIsNone(response.score)

    def test_unknown_rubric_fails(self):
        response = self._grade(
            request_for(
                [("guess", {"number": 7})], rubrics=[RubricRef(rubric_id="vibes")]
            )
        )
        self.assertIsNone(response.score)
        self.assertIn("no scorer", response.error)

    def test_no_rubrics_fails(self):
        response = self._grade(request_for([("guess", {"number": 7})], rubrics=[]))
        self.assertIsNone(response.score)


class RewardConversionTests(unittest.TestCase):
    def test_score_passes_through_clamped(self):
        self.assertEqual(0.83, to_reward(GradeResponse(score=0.83)))
        self.assertEqual(1.0, to_reward(GradeResponse(score=1.4)))
        self.assertEqual(0.0, to_reward(GradeResponse(score=-0.2)))

    def test_null_score_raises_instead_of_rewarding_zero(self):
        """An infrastructure fault must never look like a policy scoring zero."""
        with self.assertRaises(GradingUnavailable):
            to_reward(GradeResponse.failure("grader exploded"))

    def test_unreachable_grader_raises(self):
        def refuse(request):
            raise httpx.ConnectError("connection refused")

        client = GraderClient(
            http=httpx.Client(
                base_url="http://graders", transport=httpx.MockTransport(refuse)
            )
        )
        with self.assertRaises(GradingUnavailable):
            client.grade(request_for([("guess", {"number": 7})]))

    def test_server_error_raises(self):
        client = GraderClient(
            http=httpx.Client(
                base_url="http://graders",
                transport=httpx.MockTransport(lambda request: httpx.Response(500)),
            )
        )
        with self.assertRaises(GradingUnavailable):
            client.grade(request_for([("guess", {"number": 7})]))


if __name__ == "__main__":
    unittest.main()
