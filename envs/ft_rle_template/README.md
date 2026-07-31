# FT RLE template

The template folder Frontier Tuning drops into a build context to turn a
customer's **world** — skills, tools, samples, rubrics — into a Foundry RLE
environment for **training and evaluation**.

Inference is out of scope. FT already serves agents on the shared tuning
infrastructure; this repository only replaces the part FT does not have, which
is a sandbox that can roll out episodes and produce a reward.

Aligned with the `m365_number_guess_v2` reference environment, and closes the
gap that reference calls out: *"harness-profile.json is static. In production
TCaaS would render it per deployment."* Here it is rendered.

---

## How FT uses this

```
user edits a world in FT
        |
        v
M365 backend copies this folder into a build context
        |
        +--> python -m ft_rle_template.profile --source tcaas --out harness-profile.json
        |
        +--> docker build -f server/Dockerfile .
        |
        +--> push to ACR, then create/update the Foundry RLE
```

The image is **world-agnostic**. It contains no customer prompt, no sample, no
rubric text. Those are served at runtime by TCaaS, scoped to the tenant. A world
edit re-renders one JSON document; it does not rebuild an image.

Two things move at different speeds, and that is the point:

| Changes when                        | Artifact                |
| ----------------------------------- | ----------------------- |
| this template changes               | the container image     |
| a user adds a tool, skill, or sample | `harness-profile.json`  |
| a user edits prompt text            | nothing — TCaaS serves it |

## What M365 actually drops in

**One file: `tcaas/catalog.json`.** Nothing else. No Python is edited to add a
world, a skill, a tool, a dataset, or a rubric.

A regression suite enforces this. `tests/test_dropin.py` runs a second,
unrelated world — IT Helpdesk, different tools, different id shapes, a different
write entity — end to end. Nothing in the template mentions tickets or assets.

### Adding a tool

Declare the schema, the effect, and how the offline mock serves it:

```json
{
  "tool_name": "get_asset",
  "effect": "read",
  "input_schema": { "type": "object", "properties": { "asset_id": {"type": "string"} },
                    "required": ["asset_id"] },
  "serves": { "dataset": "assets", "entity_type": "asset",
              "required": ["asset_id"], "not_found_is_error": true }
}
```

| `serves` field       | Meaning                                              |
| -------------------- | ---------------------------------------------------- |
| `dataset`            | which `datasets` collection backs the tool            |
| `required`           | arguments that must be supplied, else rejected        |
| `filters`            | optional arguments that narrow the result             |
| `where`              | constant predicate, e.g. `{"status": "open"}`         |
| `not_found_is_error` | `true` for get-one tools; an empty match is rejected  |
| `appends`            | `true` for a write tool                               |

### When declaration is not enough

`serves` filters a collection. Two things it cannot express:

- a tool that **computes** from the episode's own task data — the classic
  probe-the-hidden-state tool, which has no collection to filter;
- a rubric that scores the **trajectory** rather than the answer text.

Both arrive as module paths, not as edits to this package:

| Env var           | Module exports                                             |
| ----------------- | ---------------------------------------------------------- |
| `FT_WORLD_TOOLS`  | `TOOLS = {name: fn(task_data, arguments)}`                  |
| `FT_WORLD_CHECKS` | `CHECKS = {name: fn(params, expected, answer, calls)}`, optionally `REQUIRED_PARAMS` |

A world module **imports nothing from this package** — it exchanges plain
dicts, lists, and floats, and rejects bad arguments with a plain `ValueError`.
So a world cannot be broken by a refactor in here, and `world_checks.py` is
unit-testable on its own.

Both hooks are **offline-mock concerns only**, exactly like `serves`: in
production the tool is a real MCP endpoint and the rubric is read by an LLM
judge off its `criteria`. See `extensions.py`, and `../m365_dropin/` for a
complete worked example.
| `id_field`/`id_prefix` | field to populate with a generated id               |
| `references`         | referential checks, `{field, dataset, key}`           |

`required` and `not_found_is_error` are separate on purpose. `required` validates
the *call*; `not_found_is_error` describes the *result*. A `list_x` that returns
nothing is a legitimate empty answer, and rejecting it would break the
read-after-write overlay — the list would never reach the overlay that surfaces
the buffered write.

`serves` is only for the offline mock. Point `TCAAS_BASE_URL` at real MCP
endpoints and it is ignored; the gym-side proxy never reads it.

### Adding a rubric

```json
{ "rubric_id": "cites-tickets", "skill_id": "triage", "criteria": "...",
  "weight": 1.0, "outcome": true,
  "check": "cites_grounded_entities", "check_params": {"pattern": "\\bTKT-\\d{4}\\b"} }
```

| `check`                     | Scores                                          | Required `check_params` |
| --------------------------- | ----------------------------------------------- | ----------------------- |
| `cites_grounded_entities`   | every id in the answer appeared in tool output   | `pattern`               |
| `matches_expected_entities` | cited ids equal the reference set                | `pattern`               |
| `mentions_any`              | the answer used a required vocabulary            | `words`                 |
| `called_tool`               | a required tool was used (optionally conditional) | `tool`                  |

There is **no default id pattern**. A world whose ids are GUIDs, UPNs, or
SharePoint URLs would score zero against a guessed `[A-Za-z]+-\d+` on every
episode — and a rubric that can never be earned is indistinguishable from a bad
policy in the training curve. Declaring the shape is cheap; debugging a flat
reward curve is not.

`outcome: true` marks a rubric as gating: if every outcome rubric scores zero the
episode scores zero, so process rubrics cannot pay out for a wrong answer. It is
declared per rubric rather than hardcoded by id.

An undeclared or unknown `check`, or one missing its required params, **raises**.
It does not score `0.0` with a polite note — that is how a dropped-in world
trains to completion on a flat reward signal with nothing to tell you.

A rubric with no `check` at all is the real-judge path: it loads fine, and the
LLM judge behind `GRADERS_BASE_URL` reads its `criteria` instead.

### Fail-at-load validation

`Catalog` rejects a world that

- cites an unknown skill,
- binds a tool to a missing dataset,
- declares a rubric `check` that is unknown or missing its required params, or
- names a tool that collides with a base tool shipped in the image —
  base tools are matched first, so the world's tool would never run and calling
  it would end the episode instead.

A bad drop-in fails at startup rather than deep inside a rollout.

### Verifying a drop-in

Unit tests cannot tell you whether *your* world produces a usable signal. This
can:

```bash
docker run -d -p 8000:8000 -v "$PWD/catalog.json:/world/catalog.json:ro" \
  -e FT_CATALOG_PATH=/world/catalog.json -e SELF_BASE_URL=http://localhost:8000 <image>
python -m ft_rle_template.verify_world --url ws://localhost:8000/ws
```

It rolls out a world-agnostic policy - one that reads the tool schemas from the
observation and calls tools with values earlier tools returned - against a
policy that answers blindly. If competence does not out-score noise, the reward
is degenerate and training will not learn. Usually that means a rubric's
`check_params` do not match the world's id format, or a `serves` binding returns
nothing. Make this the gate in the image build pipeline.

## Layout

```
config.py                 env-var configuration, including FT_TOOL_MODE
models.py                 the {tool_name, arguments} action envelope
logic.py                  terminal tool, static protocol text, prompt composition
extensions.py             optional world-supplied tools and rubric checks
profile.py                renders harness-profile.json from a world
harness-profile.json      generated; re-render after changing the catalog

tcaas/catalog.json        >>> the world M365 replaces <<<
tcaas/                    content service: client, models, identity, and a mock
tools/                    base tools, the TCaaS proxy, and containment strategies
graders/                  tc_graders client, trajectory recorder, and a mock
server/                   the OpenEnv environment, app, and Dockerfile
server/sticky_http.py     keeps an episode alive across plain HTTP calls
verify_world.py           smoke-tests a dropped-in world for signal degeneracy
tests/                    97 tests; fixtures/ holds three independent worlds
```

## The two ideas that matter

### 1. Dual-mode tools

FT tools must behave differently in training than in production, or a training
run mutates a real tenant. `ToolSpec.effect` declares `read` or `write`, and
`FT_TOOL_MODE` selects behaviour:

| effect  | `inference`         | `training`                                    |
| ------- | ------------------- | --------------------------------------------- |
| `read`  | real call           | real call, overlaid with this episode's writes |
| `write` | real call           | buffered, never sent                           |

The overlay is the subtle half. A policy that files a case note at step 2 and
lists case notes at step 4 must see its own write, or it is trained against a
world that forgets what it just did.

`training` is the default. An operator must opt *in* to real writes.

**Containment has two implementations, chosen by `FT_VIRTUALIZATION`:**

| mode        | Who contains writes    | Overlay                                    |
| ----------- | ---------------------- | ------------------------------------------ |
| `local`     | `tools/buffer.py`      | gym-side, generic argument-subset match     |
| `delegated` | FT's virtualization session | server-side; the gym overlays nothing |

`delegated` is the one to use in production. FT already runs tool
virtualization: a session that intercepts writes into a write-ahead buffer and
serves reads through it. Reimplementing that in the gym would mean guessing
merge semantics that each connector already knows — whether a write upserts or
appends, which fields key a record. `VirtualizedSession` opens an FT session on
reset, tags every outbound call with the session header, and reads the effect
log back for the audit trail. `local` remains the default so the template runs
offline with no FT dependency.

**Neither strategy exposes a commit path.** `WriteAheadBuffer` has no commit
method, and the delegated client deliberately does not wrap FT's commit
endpoint. Containment is structural, not a convention someone can forget: no
code path in either class can apply an effect. A test asserts this on both.

Buffered writes are exposed as `observation.pending_effects` — a declared field
rather than metadata, because OpenEnv's `serialize_observation` strips
`metadata` from the wire and the audit trail of what training *would* have
written must not vanish silently.

### 2. The profile is generated, not written

`profile.py` maps a world onto the harness profile:

| FT world              | harness profile                              |
| --------------------- | -------------------------------------------- |
| registered MCP tools  | `actionSpace.actions` / `modelActions`        |
| base terminal tool    | `actionSpace.terminalActions`                 |
| samples by type       | `evalDefaults.split` / `trainingDefaults.split` |
| sample counts         | `evalDefaults.limit`                          |
| rubric pass mark      | `reward.successRewardThreshold`               |

Skill *instructions* are deliberately absent. Three channels run on three
clocks, and mixing them is the classic mistake:

| Channel                                    | Changes      | Holds                        |
| ------------------------------------------ | ------------ | ---------------------------- |
| `observationRendering.instructions`        | never        | the JSON action protocol     |
| `observation.prompt`                       | per episode  | skill workflow + user query  |
| `observation.feedback`                     | per step     | the last tool result         |

A test asserts no skill workflow text appears anywhere in the rendered profile.

## Failure semantics

Getting this wrong corrupts the reward signal, so the rules are explicit:

| Situation                  | Result                                  |
| -------------------------- | --------------------------------------- |
| bad tool arguments         | `success=False`, episode continues       |
| unknown tool               | `success=False`, episode continues       |
| world tool shadows a base tool | rejected at load                     |
| rubric check unknown or missing params | rejected at load             |
| world declares an unservable tool | **501** from the mock             |
| rubric declares no scorable check | **501** from the grader           |
| real endpoints, no tenant configured | raises `IdentityNotConfigured` |
| wrong world id on a request | **403**                                 |
| missing tenant headers     | **401**                                  |
| backend unreachable        | raises `ToolTransportError`              |
| TCaaS unreachable on reset | raises `TCaaSUnavailable`                |
| grader returns no score    | raises `GraderUnavailable`               |
| step budget exhausted      | episode graded, `metadata.truncated`     |

Everything that can be caught at load is caught at load: a world that cannot be
served or scored should never reach a rollout.

The two `501`s separate *policy error* from *operator error*. A bad argument is
a signal the policy should learn from. A world the template cannot serve or
score is a deployment bug, and returning `0.0` for it would bury that bug under
a plausible-looking training curve.

An outage is never scored `0.0`. Rewarding an outage trains on it. And a fallback
task on a TCaaS failure would train on the wrong data, so reset fails loudly.

A wrong answer scores `0.0` overall regardless of process rubrics, so a correct
solve always outranks a wrong one.

## Determinism

`pick_task(split, seed)` is a pure lookup — no RNG. GRPO reuses one seed across a
group of trajectories, so identical seeds must yield identical tasks by
construction, not by luck.

## Running it

```bash
pip install -e '.[dev]'
python -m pytest tests -q

# render the profile from the bundled example world
python -m ft_rle_template.profile --out harness-profile.json

# run the container app locally (mocks ride along under /mock)
SELF_BASE_URL=http://127.0.0.1:8000 \
  python -m uvicorn ft_rle_template.server.app:app --port 8000
```

Episodes are multi-turn, and `POST /reset` and `POST /step` support that: the
template serves them from one persistent environment (`server/sticky_http.py`),
so `/step` resumes the episode `/reset` started. This is a deliberate override.
OpenEnv's own handlers build and `close()` a fresh environment per request, which
makes a multi-turn episode impossible over REST — `/step` lands on an environment
that was never reset and raises. Its session pool is only consulted by `/ws` and
`/mcp`, and **Foundry RLE drives a leased sandbox over the plain REST routes**,
so without the override every rollout dies on its first tool call.

`/ws` and `/mcp` are untouched and keep their per-session environments, so they
stay concurrent. Set `FT_STATEFUL_HTTP=0` to restore OpenEnv's stock behaviour.
One leased sandbox runs one rollout at a time; `reset` may be called repeatedly
to run episodes back-to-back on the same sandbox.

```bash
docker build -f server/Dockerfile -t ft-rle-template .
docker run -p 8000:8000 -e SELF_BASE_URL=http://localhost:8000 ft-rle-template
```

## Configuration

| Variable                     | Default                        | Purpose                              |
| ---------------------------- | ------------------------------ | ------------------------------------ |
| `FT_TOOL_MODE`               | `training`                     | `training` contains writes           |
| `FT_VIRTUALIZATION`          | `local`                        | `local` or `delegated` containment    |
| `FT_VIRTUALIZATION_BASE_URL` | unset                          | FT session service, `delegated` only  |
| `FT_CATALOG_PATH`            | bundled `tcaas/catalog.json`   | point the mock at another world       |
| `FT_WORLD_TOOLS`             | unset                          | module exporting `TOOLS` *(mock only)* |
| `FT_WORLD_CHECKS`            | unset                          | module exporting `CHECKS` *(mock only)* |
| `TCAAS_BASE_URL`             | `$SELF_BASE_URL/mock/tcaas`    | content service                      |
| `GRADERS_BASE_URL`           | `$SELF_BASE_URL/mock/graders`  | grading service                      |
| `FT_TENANT_ID` / `FT_USER_ID` / `FT_WORLD_ID` | demo values *(mocks only)* | tenant scope, never a reset argument |
| `FT_MAX_STEPS_PER_EPISODE`   | `12`                           | step budget                          |
| `FT_SUCCESS_THRESHOLD`       | `0.5`                          | pass mark, mirrored into the profile |
| `MAX_CONCURRENT_ENVS`        | `8`                            | concurrent episodes per container    |
| `FT_STATEFUL_HTTP`           | `1`                            | keep one env across `/reset`+`/step` |

Tenancy is deployment configuration, never a `reset` parameter. A seed that
could cross tenants would be a data-boundary bug, not a task knob.

The demo tenant defaults apply **only while both `TCAAS_BASE_URL` and
`GRADERS_BASE_URL` still point at the bundled mocks**. Once either names a real
service, all three identity variables become mandatory and startup fails without
them. A container pointed at real TCaaS but never told its tenant would
otherwise read another tenant's world under `tenant-demo`, and every downstream
check would pass because the request is internally consistent.

## What is mocked

`tcaas/app.py` and `graders/app.py` are stand-ins so the container runs end to
end offline. They are mounted as sub-apps but still reached over HTTP, so the
service boundary stays honest — swapping in the real services is two environment
variables, not a refactor.

The example world is HR Onboarding: 2 skills, 4 tools, 4 rubrics, 8 train and 4
validation samples. `tests/fixtures/world_it.json` is a second, unrelated world
used to prove the template is not written around the first, and
`tests/fixtures/world_ext.json` is a third that exercises the two extension
points.

Every stand-in is marked `REAL SYSTEM:` in the source.

## Alignment with `m365_number_guess_v2`

Same decisions, for the same reasons:

- Content is served from TCaaS at reset, not baked into the image.
- Grading calls tc_graders with the trajectory; rubric ids come from the world.
- The skill is **pinned from the sample** (`sample.skill_id`), that skill's
  workflow goes in the prompt, and only that skill's rubrics grade the episode.
  No router is trained. Skill *selection* is an inference-time concern; this
  gym trains skill *execution*.

The difference: v2's stated gap is that TCaaS should render `harness-profile.json`
per deployment. Here `profile.py` does exactly that, and a test fails if the
checked-in file drifts from the world.

The v2 world itself is reproduced end to end in `../m365_dropin/` — same tasks,
same rubric prose, reward parity verified against its `grading.py`. Building it
needed no change to the gym, the profile renderer, the action envelope, the
containment strategies, or the grading contract; only the two offline-mock
extension points above, which are the parts that get deleted in production.

## Known gaps

- **Rubric scorers are crude.** The four declared checks are deterministic string
  and set operations. Production wants an LLM judge reading each rubric's
  `criteria`, with these as the cheap regression layer.
- **No content version pinning.** An RLE version should pin an immutable TCaaS
  content version so a training run is reproducible. Today the profile only
  records `content_version` in `notes`.
- **Endpoint shapes are approximations.** The real TCaaS, tc_graders, and FT
  virtualization-session contracts will differ. Each is behind one client class.
- **`local` containment guesses merge semantics.** `_matches` treats read
  arguments as a filter subset. This is why `delegated` exists; `local` is for
  offline development, not production training.
