"""Runnable multi-turn rollout for the number-guessing environment.

Drives a full episode over the persistent WebSocket session using a scripted
binary-search agent, showing how a stateful multi-step rollout works (the loop
a Foundry RLE harness or an LLM policy would otherwise run).

Prerequisites: the environment server must be running, e.g.

    uvicorn server.app:app --host 127.0.0.1 --port 8000

Then, from the environment root:

    python rollout_example.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Optional

try:
    from .client import NumberGuessClient
    from .grading import optimal_steps
    from .logic import MAX_NUMBER, MAX_STEPS, MIN_NUMBER
    from .models import NumberGuessAction, NumberGuessObservation
except ImportError:
    from client import NumberGuessClient
    from grading import optimal_steps
    from logic import MAX_NUMBER, MAX_STEPS, MIN_NUMBER
    from models import NumberGuessAction, NumberGuessObservation


class BinarySearchAgent:
    """Scripted policy: narrow the range with compare, then commit with guess."""

    def __init__(self, low: int, high: int) -> None:
        self.low = low
        self.high = high

    def act(self, obs: NumberGuessObservation) -> NumberGuessAction:
        # Fold the previous compare result into the feasible range.
        if obs.feedback and obs.last_action == "compare" and obs.last_number is not None:
            n = obs.last_number
            if "higher" in obs.feedback:
                self.low = n + 1
            elif "lower" in obs.feedback:
                self.high = n - 1
            elif "equal" in obs.feedback:
                self.low = self.high = n

        if self.low >= self.high:
            return NumberGuessAction(action_type="guess", number=self.low)
        mid = (self.low + self.high) // 2
        return NumberGuessAction(action_type="compare", number=mid)


async def run_rollout(base_url: str, seed: Optional[int]) -> None:
    reset_kwargs = {"seed": seed} if seed is not None else {}
    async with NumberGuessClient(base_url=base_url) as env:
        result = await env.reset(**reset_kwargs)
        agent = BinarySearchAgent(MIN_NUMBER, MAX_NUMBER)

        print(f"optimal step budget: {optimal_steps()}  (binary-search probes + 1 commit)")
        for step in range(1, MAX_STEPS + 1):
            action = agent.act(result.observation)
            result = await env.step(action)
            print(
                f"step {step}: {action.action_type}({action.number}) "
                f"-> {result.observation.feedback!r} "
                f"reward={result.reward} done={result.done}"
            )
            if result.done:
                break

        print(f"\nfinal reward: {result.reward}  (terminal episode score, aggregation=final)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a number-guess rollout.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_rollout(args.base_url, args.seed))


if __name__ == "__main__":
    main()
