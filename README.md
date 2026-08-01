# Foundry RLE Harness Profile Examples

This repository contains small Foundry RLE/OpenEnv examples that can be used to
iterate on harness profiles, environment contracts, and collaboration patterns.

## Harness Profile Reference

[`harness-profile/`](harness-profile/README.md) holds the specification the
environments below conform to - the documented `rle.harness/v0.1` schema plus
five worked example profiles covering bounded games, one-shot answers,
multi-action tools, MCP tool discovery, and test-based code rewards.

Read it when you need the contract itself; read the examples below when you
need to see it implemented. Each runnable environment ships its own
`harness-profile.json`; the drop-in inherits the template's.

## Examples

The five examples build on each other. To understand the template and how a
world is dropped into it, read **FT RLE Template** first, then the
**Number Guess drop-in** — the template defines the contract, the drop-in is a
complete worked example of satisfying it.

- [M365 POC](envs/m365_poc_env/README.md): a tiny OpenEnv-style
	environment for M365 collaboration proof-of-concepts.
- [Number Guess](envs/m365_number_guess_env/README.md): a multi-turn OpenEnv
	environment that rewards efficient, strategy-driven problem solving.
- [Number Guess v2](envs/m365_number_guess_v2/README.md): the same task wired
	into the **envisioned M365 / Foundry RLE architecture** - task content from a
	mocked TCaaS service, tools split between the sandbox and TCaaS, and reward
	from a mocked tc_graders service. Built to validate the generic gym
	integration end to end.
- [FT RLE Template](envs/ft_rle_template/README.md): a **world-agnostic** gym
	generalised out of Number Guess v2. Nothing in it names a world; the world
	arrives as configuration, so M365 supplies content rather than forking code.
- [Number Guess drop-in](envs/m365_dropin_for_number_guess_v2/README.md):
	everything M365 must supply to reproduce Number Guess v2 on that template -
	a catalog, one world tool, and two rubric checks. Deployed and verified
	end to end on Foundry RLE.

## Run Tests

```bash
python -m unittest discover -t . -s tests
```

The template ships its own suite, which the command above does **not** pick up:

```bash
cd envs/ft_rle_template && python -m pytest tests -q      # 97 tests
```

Environment-level tests need `openenv`, which lives only in the container image,
so they skip on a bare checkout. To run everything:

```bash
cd envs/m365_number_guess_v2
docker build -f server/Dockerfile -t m365-number-guess-v2 .
cd ../..
docker run --rm -v "$PWD:/work" -w /work -e PYTHONPATH=/work/envs \
    m365-number-guess-v2 python -m unittest discover -t . -s tests
```