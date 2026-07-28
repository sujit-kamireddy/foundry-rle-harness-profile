# Foundry RLE Harness Profile Examples

This repository contains small Foundry RLE/OpenEnv examples that can be used to
iterate on harness profiles, environment contracts, and collaboration patterns.

## Examples

- [M365 POC](envs/m365_poc_env/README.md): a tiny OpenEnv-style
	environment for M365 collaboration proof-of-concepts.
- [Number Guess](envs/m365_number_guess_env/README.md): a multi-turn OpenEnv
	environment that rewards efficient, strategy-driven problem solving.
- [Number Guess v2](envs/m365_number_guess_v2/README.md): the same task wired
	into the **envisioned M365 / Foundry RLE architecture** - task content from a
	mocked TCaaS service, tools split between the sandbox and TCaaS, and reward
	from a mocked tc_graders service. Built to validate the generic gym
	integration end to end.

## Run Tests

```bash
python -m unittest discover -t . -s tests
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