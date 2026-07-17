# Number Guess Environment

A multi-turn OpenEnv environment that measures how *efficiently* a model solves a
task. It is built from the [M365 POC](../m365_poc_env/README.md) hello-world
template and mirrors the same layout:

```text
envs/m365_number_guess_env/
  pyproject.toml
  models.py
  logic.py
  grading.py
  harness-profile.json
  server/
    app.py
    number_guess_environment.py
    Dockerfile
```

## Task

The environment hides an integer in `[1, 10]`. The model interacts across
multiple turns using two tools:

- `compare(number)` — the **strategy tool**. Reports whether the target is
  `higher`, `lower`, or `equal` to `number`, without committing. This is what
  enables a binary-search strategy. It does not end the episode.
- `guess(number)` — **commit** a final answer. A correct guess ends the episode.
  A wrong guess returns no direction, so blind guessing is discouraged.

## Reward

Scoring lives in a dedicated grader (`grading.py`) that is called **once**, when
the episode ends. Intermediate steps score `0.0`; the terminal step carries the
whole reward, and the harness folds it with `aggregation: final`. With
`optimal = ceil(log2(range)) + 1` (binary-search probes plus one commit,
`= 5` for `1..10`), the episode ends on a correct guess or after `MAX_STEPS`:

- Solved: `0.5 + 0.5 * clamp(optimal / steps_used, 0, 1)`.
  An optimal binary search yields `1.0`; slower solves decay toward `0.5`.
- Unsolved at episode end: `0.5 * (1 - |number - target| / (max - min))` —
  partial credit for closeness, always below any real solve.

`grade()` is transport-agnostic, so the same function backs both the OpenEnv step
reward and any out-of-band grader (e.g. an arena `GymGrader`).

## What the policy sees

| Input | Source |
|---|---|
| Task prompt | `TASK_PROMPT` in `logic.py` -> `observation.problem` (profile `promptPath`) |
| Instructions | `harness-profile.json` -> `observationRendering.instructions` |
| Tools + schemas | `harness-profile.json` -> `actionSpace` (from OpenEnv `/schema`) |
| Step feedback | grader/env -> `observation.feedback` (profile `feedbackPath`) |
| Success criteria | `harness-profile.json` + `grade()`, used by the harness, not shown to the policy |

Flow per episode:

```text
reset() -> observation.problem + instructions
  policy reads prompt + instructions + tools
  -> tool call (compare / guess) -> step() -> observation.feedback
  ... repeat until done (correct guess or MAX_STEPS) ...
harness scores reward (grade, aggregation=final) and success (reward >= 1.0)
```

The policy acts on the prompt, instructions, tools, and feedback. The success
threshold lives in the harness, not the prompt. To make the goal explicit to the
model, put it in the instructions.

## Local Run

```bash
cd envs/m365_number_guess_env
python -m venv .venv
. .venv/bin/activate
pip install -e .
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl -s http://127.0.0.1:8000/reset -X POST -H 'content-type: application/json' -d '{}'
curl -s http://127.0.0.1:8000/step -X POST -H 'content-type: application/json' \
  -d '{"action":{"action_type":"compare","number":5}}'
curl -s http://127.0.0.1:8000/step -X POST -H 'content-type: application/json' \
  -d '{"action":{"action_type":"guess","number":7}}'
```

## Container Run

```bash
cd envs/m365_number_guess_env
docker build -f server/Dockerfile -t m365-number-guess-env .
docker run --rm -p 8000:8000 m365-number-guess-env
```

## Contract

The server is created through OpenEnv's `create_app` wrapper. The environment
implements:

- `reset(seed=None, episode_id=None, **kwargs)` — the target is seedable for
  reproducibility.
- `step(action: NumberGuessAction, timeout_s=None, **kwargs)`
- `state`
- `get_metadata()`

The matching Foundry RLE harness profile is in
[harness-profile.json](harness-profile.json).
