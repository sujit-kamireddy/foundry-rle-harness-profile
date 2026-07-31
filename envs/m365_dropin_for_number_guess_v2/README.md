# `m365_dropin_for_number_guess_v2` — everything M365 supplies on top of the template

This folder is the **complete** input side of an FT world. Nothing here is part
of `ft_rle_template/`; nothing in `ft_rle_template/` was edited to make this
world work. That separation is the point: it shows, concretely, where the
template ends and where M365's responsibility begins.

The world implemented is the **number-guess** reference env from
`foundry-rle-harness-profile/envs/m365_number_guess_v2`, chosen because it is
the env Foundry itself ships as the RLE example — so "does the template cover
it?" has a checkable answer rather than a plausible one.

## The whole contract

| You supply | What it is | Required? |
| --- | --- | --- |
| `catalog.json` | The world: skills, rubrics, tool schemas, tasks per split | **Yes** |
| `world_tools.py` | Tools that *compute* instead of filtering | Only if `serves` can't express a tool |
| `world_checks.py` | Scorers for rubrics the four built-in checks miss | Only while running the offline mock |
| `examples/` | Captured wire traffic | No — docs only |

Three files. Two of them are optional, and **both optional ones disappear in
production**: `compare` becomes a registered MCP endpoint, and the rubrics get
read by a real LLM judge off their `criteria` prose. What survives contact with
the real system is `catalog.json` alone.

## Run it

```bash
cd ..                       # the parent of ft_rle_template/ and m365_dropin_for_number_guess_v2/
export FT_CATALOG_PATH=$PWD/m365_dropin_for_number_guess_v2/catalog.json
export FT_WORLD_TOOLS=m365_dropin_for_number_guess_v2.world_tools
export FT_WORLD_CHECKS=m365_dropin_for_number_guess_v2.world_checks
export FT_MAX_STEPS_PER_EPISODE=10
export FT_EPISODE_TIMEOUT_S=60
export FT_SUCCESS_THRESHOLD=1.0

python -m ft_rle_template.server.app          # gym on :8000
python -m ft_rle_template.profile --out -     # render harness-profile.json
python -m ft_rle_template.verify_world        # smoke-test the reward signal
```

`verify_world` reports `competent=0.194 blind=0.000 → PASS`: its deliberately
world-agnostic policy probes once and answers, and still beats a policy that
answers blind. That gap is the thing training needs; a world where it is zero
will not learn, no matter how good the rubric prose is.

## Deploying to Foundry RLE

Verified end-to-end against RLE TIP (`westus2`). The image is two layers: the
generic base, then this world as a thin layer over it.

```bash
cd ../ft_rle_template && docker build -t ft-rle-template:latest -f server/Dockerfile .
cd ../m365_dropin_for_number_guess_v2 && docker build --platform linux/amd64 --provenance=false --sbom=false \
  -t devrle.azurecr.io/<project>-<env>:latest .

az acr login --name devrle
docker push devrle.azurecr.io/<project>-<env>:latest
```

`--platform linux/amd64` is required, and `--provenance=false --sbom=false` keeps
buildx from pushing a multi-manifest index that the disk-image converter rejects.

Register the pushed image, then poll until the disk-image conversion is `Ready`
(this is the ACR→sandbox conversion, and a sandbox cannot be leased before it
finishes). Prefer the immutable `@sha256:` digest over `:latest` so a version
always names one image:

```http
POST {project}/fine_tuning/environments?api-version=2025-05-01
{"name": "m365-number-guess", "acrImagePath": "devrle.azurecr.io/...@sha256:..."}
```

Re-registering the same name publishes a new version (`1.0.0` → `2.0.0`).

Then lease a sandbox and drive an episode with `azure-ai-projects` (2.4.0+):

```python
with client.rle.get_openenv_client(name="m365-number-guess") as oe:
    inst = oe.get_instance()
    inst.reset(seed=0)
    inst.step({"tool_name": "compare", "arguments": {"number": 5}})
```

Two things bit us, both worth knowing:

- **Auth.** The RLE data plane rejects AAD bearer tokens with
  `401 Jwt issuer is not configured`, at every audience we tried. Use the
  resource's API key instead — pass `AzureKeyCredentialPolicy(key, "api-key")`
  as the client's `authentication_policy`. The control plane accepts both.
- **Statefulness.** RLE drives the sandbox over plain `POST /reset` and
  `POST /step`, which OpenEnv serves from a throwaway environment per request.
  The template overrides those routes (`server/sticky_http.py`) so an episode
  survives; without it every rollout fails on its first tool call.

Measured on the deployed environment, which is also the reward parity check
below, reproduced end-to-end through RLE:

| policy                                  | reward |
| --------------------------------------- | ------ |
| binary search, 4 probes then submit      | 1.0    |
| linear scan, 7 probes then submit        | 0.875  |
| blind but lucky, 0 probes                | 0.667  |
| maximally wrong, 0 probes                | 0.111  |

## What each file carries

### `catalog.json`

- **1 skill** — `number-guess`. The `workflow` text is the reference env's,
  extended with the two sentences that describe *this* action space (`compare`
  probes, `submit_answer` ends the episode). The reference env put that in a
  `prompt_template` in its profile; here it belongs to the skill, because in FT
  a skill's workflow is exactly this kind of instruction.
- **2 rubrics** — `efficient-solve` (weight 1.0, `outcome: true`) and
  `probe-before-commit` (weight 0.5). `criteria` is the reference env's prose,
  unchanged, so a real judge can score it without this folder's Python.
- **1 tool** — `compare`. Note it has **no `serves` block**: it is answered by
  `world_tools.py` instead.
- **18 tasks** — 12 train / 6 validation, ported verbatim, each carrying
  `data.target`. `data` never reaches the sandbox; it is grading context only.
- **`datasets: {}`** — this world has no collections to read. The template's
  declarative read/write machinery is simply unused here, which is the correct
  outcome for a world that has no records.

### `world_tools.py`

`serves` binds a tool to a dataset and filters it. `compare` filters nothing —
it computes from the episode's hidden `target`. So it ships as a function:

```python
def compare(task_data, arguments) -> str: ...
TOOLS = {"compare": compare}
```

The template already sends `task_id` on every tool call, so this needed no new
wire field — it is the same scope a real user-tool endpoint receives.

Bad arguments raise a plain `ValueError`, which the template converts to a 400
and the gym renders as feedback. So this file **imports nothing from the
template** and still gets the reject-vs-outage split right.

### `world_checks.py`

The template's four built-in checks all compare strings. This world scores *how
efficiently* the answer was found, which none of them reach. The signature is
builtin types only:

```python
fn(params, expected, answer, calls) -> float | (float, str)
```

so this file is unit-testable standalone and survives refactors inside the
template.

## Reward parity with the reference env

The reference env computes one scalar in `grading.py`. The template computes a
weighted mean over rubrics with an outcome gate. Both were run over identical
episodes:

| Episode | `efficient-solve` | `probe-before-commit` | Reward |
| --- | --- | --- | --- |
| Optimal binary search, 4 probes + commit | 1.000 | 1.00 | **1.000** |
| Wasteful linear scan, 7 probes + commit | 0.813 | 1.00 | **0.875** |
| Probed well, answered wrong (5 vs 7) | 0.389 | 1.00 | **0.593** |
| Blind lucky one-step guess | 1.000 | 0.00 | **0.667** |
| Maximally wrong, no probes (10 vs 1) | 0.000 | 0.00 | **0.000** |

Two things worth reading off that table:

**The lucky guess is punished.** In the reference env a blind correct guess
scores a perfect 1.0, because one step is under the optimal count and efficiency
clamps. Splitting the signal into two rubrics fixes that without special-casing:
process is scored separately from outcome, so luck earns 0.667 and skill earns
1.0. This is the same reason FT rubrics are per-criterion rather than one score.

**The outcome gate only fires at exactly zero.** `efficient-solve` pays partial
credit for closeness, so it reaches 0.0 only when the answer is maximally far
from the target. Everywhere else the gate is inert and the process rubric still
pays. That is deliberate — a dense gradient is what an unsolved-at-first policy
needs — but it does mean this world's gate is nearly decorative. A world that
wants a hard gate should use a strictly binary outcome rubric.

## Behaviours worth knowing

- **A rejected call still costs a step.** `examples/step-rejected.json` shows
  `compare({"number": "five"})` returning feedback with `done: false` — and the
  step counter at 2. It also counts toward `steps_used`, so sloppy arguments
  cost efficiency. That is correct: a real agent's malformed call costs a real
  turn.
- **The answer is parsed, not structured.** `submit_answer` takes free text
  (that is the template's universal terminal action), so `solve_efficiency`
  pulls the first integer out of it. An episode that ends without naming a
  number scores 0.0 rather than defaulting to a guess.
- **Determinism.** `pick_task(split, seed)` is `tasks[split][seed % len]` with
  no RNG, so every rollout in a GRPO group sees the same target — which is what
  makes the group's advantage estimate mean anything.

## What this exercise exposed as missing

Building this world required **two additions to the template**, both in the
offline-mock layer only, both now shipped as documented extension points
(`ft_rle_template/extensions.py`):

1. **Task-scoped tool dispatch** (`FT_WORLD_TOOLS`) — the mock could only filter
   static datasets. The wire contract already carried `task_id`; only the mock
   ignored it.
2. **World-supplied checks** (`FT_WORLD_CHECKS`) — `CHECKS` was a closed set of
   four.

Nothing in the **gym**, the **profile renderer**, the **action envelope**, the
**containment strategies**, or the **grading contract** needed to change. The
parts that face the real system were already sufficient; the parts that needed
extending were exactly the parts that get deleted in production.

Still not covered by the template, and not exercised by this world:

- **No LLM judge** — real FT rubrics are prose scored by a model.
- **M365 Capabilities** (SharePoint, Teams, Meetings, GraphConnectors) are not
  modelled; only MCP-style tools are.
- **No parallel tool calls** — one call per step.
- **Split naming** — the template says `train`/`validation`, FT says
  `Training`/`Evaluation`, with no translation layer.
