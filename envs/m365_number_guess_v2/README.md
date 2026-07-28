# Number Guess v2: the envisioned M365 / Foundry RLE architecture

**Goal: validate the Foundry RLE generic gym integration end to end.**

The task is deliberately trivial (guess a number in `1..10`). Everything
*around* it is the point: task content comes from a **TCaaS** service, tools are
split between the sandbox and TCaaS, and reward comes from a **tc_graders**
service. Both external services are mocked, but reached over real HTTP — so the
seams that matter in production are exercised here.

> v1 (`../m365_number_guess_env/`) is the minimal OpenEnv reference and stays
> untouched. This is the architecturally realistic sibling.

---

## The shape of it

```text
        ┌──────────────────┐
        │ Foundry harness  │   out of scope — drives us via harness-profile.json
        │ training / eval  │
        └────────┬─────────┘
                 │ /reset(seed, split)   /step(tool_name, arguments)
                 ▼
   ╔═════════════════════════════════════════════════╗
   ║              GYM CONTAINER (in scope)           ║
   ║                                                 ║
   ║   server/number_guess_environment.py            ║
   ║        │                    │                   ║
   ║        │ task bundle        │ item + sample     ║
   ║        ▼                    ▼                   ║
   ║   tcaas/client.py      graders/client.py        ║
   ╚════════│════════════════════│═══════════════════╝
            │ HTTP               │ HTTP
            ▼                    ▼
   ┌──────────────────┐  ┌──────────────────────┐
   │  TCaaS  (MOCK)   │  │ tc_graders  (MOCK)   │
   │  /mock/tcaas     │  │ /mock/graders        │
   │                  │  │                      │
   │ skills, rubrics, │  │ scores each rubric,  │
   │ tasks, user tools│  │ aggregates -> reward │
   └──────────────────┘  └──────────────────────┘
```

Both mocks are FastAPI sub-apps of the same server (`server/app.py`) because the
image runs one uvicorn process — but the gym still talks to them over HTTP, so
swapping in the real services is a URL change (`config.py`).

---

## One episode, end to end

```text
 reset(seed=42, split="train")
   │
   ├─► tcaas.pick_task("train", 42) ─────────► TaskBundle
   │      tasks[split][42 % len(split)]        skill + user_query + data + rubrics
   │      no RNG: same seed ⇒ same task
   │
   ├─► ToolRegistry(base=guess, user=[compare])
   │
   └─◄ observation.prompt = skill + "\n\nUser: " + user_query

 step(tool_name="compare", arguments={"number": 5})
   └─► TCaaS runs it (it owns the answer)  ─►  "The target is higher than 5."

 step(tool_name="guess", arguments={"number": 7})    ← terminal tool
   └─► graders.grade(item, sample) ──────────► score 1.0, per-rubric detail
                                               reward = score, done = True
```

**Why the seed matters.** GRPO samples K trajectories from *one* task, so the
harness reuses a seed across the group. `pick_task` is a pure lookup, so that
holds by construction — [`tcaas/catalog.py`](tcaas/catalog.py).

---

## The four pieces

### 1. Content — `tcaas/`

```text
catalog.json ──► Catalog ──► TaskBundle
  tenant     one tenant per container; never a reset parameter  (identity.py)
  skills     a workflow: how to use the tools for a request
  rubrics    generated FROM a skill, so each carries skill_id
  tasks      by split, each with the data needed to grade it
```

A skill owns its rubrics (`rubric.skill_id`), never the reverse — real rubrics
cite the skill's own text. Resolution is one hop each way, so the gym makes a
single TCaaS call per episode.

| File | Role |
|---|---|
| [`tcaas/catalog.py`](tcaas/catalog.py) | `pick_task(split, seed)` — the sample selector |
| [`tcaas/identity.py`](tcaas/identity.py) | tenant scope, sent as headers on every call |
| [`tcaas/tools.py`](tcaas/tools.py) | user-tool implementations (server side) |
| [`tcaas/app.py`](tcaas/app.py) | the mock service |

### 2. Tools — `tools/`

The gym is a **generic sandbox**: it ships base tools and is *extended* with user
tools from TCaaS. One result type downstream, so nothing else learns which ran.

```text
                  ┌─ LocalToolExecutor    guess    in-process     (local.py)
ToolRegistry ─────┤
                  └─ TCaaSToolExecutor    compare  HTTP → TCaaS   (proxy.py)
                              │
                              ▼
              ToolExecutionResult(output, success, call_id)
```

`compare` is remote because **TCaaS owns the task data** — the sandbox never sees
the hidden number. `guess` only records an answer; correctness is the grader's
call.

The distinction that must not blur ([`tools/base.py`](tools/base.py)):

| Outcome | Meaning | Result |
|---|---|---|
| Bad arguments | a signal for the policy | `success=False`, episode continues |
| Service down | infrastructure fault | `ToolTransportError` raised |

### 3. Grading — `graders/`

Mirrors `tc_graders`' contract: **`item` is what the model was given, `sample` is
what it produced**, and the *caller* supplies the reference in `item.expected` —
the grader never looks anything up.

```text
episode ──► TrajectoryRecorder ──► OpenAI-style messages   (trajectory.py)
                                     user / assistant+tool_calls / tool
              │
              ▼
   { rubrics, aggregation, item, sample }  ──POST /grade──►  mock grader
                                                                │
        efficient-solve      outcome: correct? how few steps?    │  rubrics.py
        probe-before-commit  trajectory: did it gather evidence? │
                                                                ▼
                                     weighted_mean ──► score + individual_results
```

**Reward rules** ([`graders/client.py`](graders/client.py)):

- `score is None` ⇒ **raise**. A dead grader is an infrastructure fault, not a
  policy scoring zero. Rewarding `0.0` there would train on an outage.
- A wrong answer scores `0.0`, so a correct solve **always** outranks a wrong one
  (0.50–1.00 vs 0.00–0.33). Otherwise the process rubric would pay `+0.33`
  regardless of correctness — free reward for probing then answering wrong.

### 4. The env — `server/`

[`number_guess_environment.py`](server/number_guess_environment.py) holds the
episode: bundle, tool registry, and trajectory all live **on the instance**
(`SUPPORTS_CONCURRENT_SESSIONS = True`, up to 8 in flight). Grading fires exactly
once, on the terminal step.

---

## Where the data actually lives

Everything the gym serves comes from **one file**:
[`tcaas/catalog.json`](tcaas/catalog.json) — 1 tenant, 1 skill, 2 rubrics, 18
tasks. It stands in for TCaaS storage, so nothing else in the repo holds task
content.

```text
tcaas/catalog.json
├── tenant      tenant-demo / user-demo         one tenant per container
├── skills[1]   number-guess                    the workflow text the policy reads
├── rubrics[2]  efficient-solve      weight 1.0 ─┐ both carry skill_id
│               probe-before-commit  weight 0.5 ─┘ = number-guess
└── tasks
    ├── train[12]       guess-train-000 .. 011   trainingDefaults.split
    └── validation[6]   guess-val-000   .. 005   evalDefaults.limit: 6
```

One task record is the whole unit of work:

```json
{ "task_id": "guess-train-000", "skill_id": "number-guess",
  "user_query": "Find the hidden number between 1 and 10.",
  "data": { "target": 7 } }
```

`data.target` is **grading context, not env state**. The gym forwards it to
`item.expected` because `tc_graders` requires the caller to supply the reference;
gameplay never reads it, which is why `compare` has to be a remote TCaaS call.

**Tools are not in the catalog.** They are split by owner, which is the whole
point of the sandbox model:

| Tool | Schema + implementation | Why there |
|---|---|---|
| `guess` | [`tools/local.py`](tools/local.py) | base tool, ships with the image |
| `compare` | [`tcaas/tools.py`](tcaas/tools.py) | needs `data.target`, which only TCaaS holds |

The policy sees them merged into one list
([`tools/registry.py`](tools/registry.py)) and cannot tell them apart.

**Growing the sample** is a `catalog.json` edit: add tasks, then rerun
`tests/test_harness_profile.py`, which fails if `evalDefaults.limit` no longer
matches the validation split. Adding a *skill* also needs a stand-in scorer per
new rubric — see [Known gaps](#known-gaps).

---

## What the policy sees

Three channels reach the model, and they vary on different clocks:

```text
profile.observationRendering.instructions   STATIC   protocol: the action JSON shape
observation.prompt        (promptPath)      EPISODE  content: skill workflow + user query
observation.feedback      (feedbackPath)    STEP     result of the last tool call
```

The harness renders **one** observation field, so the gym composes the prompt
itself (`logic.compose_prompt`) and points `promptPath` at it:

```json
{
  "prompt": "<skill workflow>\n\nUser: <user query>",
  "skill": "...", "user_query": "...",
  "tools": [ { "type": "function", "function": { "name": "compare", ... } } ],
  "feedback": "The target is higher than 5."
}
```

Tool guidance lives in exactly two places, split by who owns it and how often it
changes:

| Layer | Where | Owner |
|---|---|---|
| **Protocol** — emit `{"tool_name":..,"arguments":{..}}` | `harness-profile.json` | the env, static |
| **Workflow** — how to approach the task | `skill.workflow` in the catalog | TCaaS, per skill |

`skill` and `user_query` also stay as separate observation fields for debugging,
and let us move to a `prompt_template` in one profile edit if the harness ever
supports observation-field placeholders.

> How the harness assembles `instructions` and `prompt` into system/user roles
> is **implementation-defined** — the schema does not specify it.

---

## Run it

```bash
cd envs/m365_number_guess_v2
docker build -f server/Dockerfile -t m365-number-guess-v2 .
docker run --rm -p 8000:8000 m365-number-guess-v2
```

A scripted binary-search rollout, showing the full loop:

```bash
docker exec <container> python -m m365_number_guess_v2.rollout_example \
    --base-url http://127.0.0.1:8000 --seed 0
```

```text
task: guess-train-000
step 1: compare({'number': 5}) -> 'The target is higher than 5.'  reward=0.0 done=False
step 2: compare({'number': 8}) -> 'The target is lower than 8.'   reward=0.0 done=False
step 3: compare({'number': 6}) -> 'The target is higher than 6.'  reward=0.0 done=False
step 4: guess({'number': 7})   -> 'Answer submitted: 7.'          reward=1.0 done=True

final reward: 1.0
  efficient-solve: 1.0
  probe-before-commit: 1.0
```

Poke the mocks directly (they need the tenant headers):

```bash
H='-H x-tcaas-tenant-id:tenant-demo -H x-tcaas-user-id:user-demo'
curl -s $H "http://localhost:8000/mock/tcaas/tasks?split=train&seed=0"
curl -s $H "http://localhost:8000/mock/tcaas/tools"
```

> **REST `/reset` and `/step` are single-shot** — each call gets a fresh
> environment, so curling a reset then a step raises `step() called before
> reset()`. Multi-turn episodes need the persistent WebSocket session
> ([`client.py`](client.py)); the payloads in [`examples/`](examples) show the
> wire shape, not a runnable sequence.

> Mounted under `/mock/...` because OpenEnv reserves `/{env_name}/...` at the
> root, which silently shadows two-segment mounts. Pinned by
> `tests/test_server_wiring.py`.

## Tests

`openenv` only exists inside the image, so env-level tests skip on a bare
checkout. Run everything in the container:

```bash
docker run --rm -v "$PWD/../..:/work" -w /work -e PYTHONPATH=/work/envs \
    m365-number-guess-v2 python -m unittest discover -t . -s tests
```

| Suite | Covers |
|---|---|
| `test_tcaas_mock.py` | seed determinism, split coverage, tenant scoping |
| `test_tools.py` | local vs proxied dispatch, rejection vs outage |
| `test_graders_mock.py` | rubric scoring, aggregation, reward conversion |
| `test_number_guess_v2.py` | the episode loop, concurrency, failure modes |
| `test_harness_profile.py` | profile ↔ catalog ↔ env drift |
| `test_server_wiring.py` | mock mounts stay reachable |

## Swapping in the real services

Every mock/production boundary carries a `REAL SYSTEM:` note, so
`grep -rn "REAL SYSTEM" .` enumerates all 22 of them. The ones that matter:

| Seam | Today | To go live |
|---|---|---|
| Task storage | `tcaas/catalog.json` | TCaaS-backed storage; `pick_task(split, seed)` is unchanged |
| Service URLs | `/mock/...` sub-apps | set `TCAAS_BASE_URL` / `GRADERS_BASE_URL`, drop the mounts |
| Identity | headers from env vars | bearer token carrying the same claims |
| User tools | `tcaas/tools.py` | real endpoints; the gym-side proxy is untouched |
| Rubric scoring | hand-written stand-ins | one LLM judge reading each rubric's `criteria` |
| Grade request | rubric list | an instantiated `MultiGrader`; `item`/`sample` stay identical |

The gym-side code that survives unchanged: the tool registry, the trajectory
recorder, the reward rules, and the env itself — both clients are injected, so
repointing them is configuration.

## Known gaps

- **`harness-profile.json` is static.** In production TCaaS would render it per
  deployment, since it knows a user's skills and tools before the container
  exists. Drift tests keep it honest meanwhile.