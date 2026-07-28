"""Runnable multi-turn rollout against the v2 environment.

Drives a full episode over the persistent WebSocket session with a scripted
binary-search agent - the loop a Foundry harness or an LLM policy would run.

Prerequisites: the environment server must be running, e.g.

    uvicorn m365_number_guess_v2.server.app:app --host 127.0.0.1 --port 8000

Then, from the repository's ``envs/`` directory:

    python -m m365_number_guess_v2.rollout_example --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from .client import NumberGuessClient
from .grading import optimal_steps
from .logic import MAX_NUMBER, MAX_STEPS, MIN_NUMBER
from .models import NumberGuessAction, NumberGuessObservation


class BinarySearchAgent:
    """Scripted policy: narrow the range with compare, then commit with guess."""

    def __init__(self, low: int, high: int) -> None:
        self.low = low
        self.high = high
        self._last_number: Optional[int] = None

    def act(self, obs: NumberGuessObservation) -> NumberGuessAction:
        # Fold the previous compare result into the feasible range.
        if (
            obs.feedback
            and obs.last_tool == "compare"
            and self._last_number is not None
        ):
            n = self._last_number
            if "higher" in obs.feedback:
                self.low = n + 1
            elif "lower" in obs.feedback:
                self.high = n - 1
            elif "equal" in obs.feedback:
                self.low = self.high = n

        if self.low >= self.high:
            self._last_number = self.low
            return NumberGuessAction(tool_name="guess", arguments={"number": self.low})
        mid = (self.low + self.high) // 2
        self._last_number = mid
        return NumberGuessAction(tool_name="compare", arguments={"number": mid})


async def run_rollout(base_url: str, seed: Optional[int]) -> None:
    reset_kwargs = {"seed": seed} if seed is not None else {}
    async with NumberGuessClient(base_url=base_url) as env:
        result = await env.reset(**reset_kwargs)
        agent = BinarySearchAgent(MIN_NUMBER, MAX_NUMBER)

        print(f"task: {result.observation.metadata.get('task_id')}")
        print(
            f"optimal step budget: {optimal_steps()}  (binary-search probes + 1 commit)"
        )
        for step in range(1, MAX_STEPS + 1):
            action = agent.act(result.observation)
            result = await env.step(action)
            print(
                f"step {step}: {action.tool_name}({action.arguments}) "
                f"-> {result.observation.feedback!r} "
                f"reward={result.reward} done={result.done}"
            )
            if result.done:
                break

        print(f"\nfinal reward: {result.reward}  (grader aggregate, aggregation=final)")
        for detail in result.observation.metadata.get("individual_results", []):
            print(f"  {detail['rubric_id']}: {detail['score']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a number-guess v2 rollout.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_rollout(args.base_url, args.seed))


if __name__ == "__main__":
    main()
