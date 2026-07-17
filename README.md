# Foundry RLE Harness Profile Examples

This repository contains small Foundry RLE/OpenEnv examples that can be used to
iterate on harness profiles, environment contracts, and collaboration patterns.

## Examples

- [M365 POC](envs/m365_poc_env/README.md): a tiny OpenEnv-style
	environment for M365 collaboration proof-of-concepts.
- [Number Guess](envs/m365_number_guess_env/README.md): a multi-turn OpenEnv
	environment that rewards efficient, strategy-driven problem solving.

## Run Tests

```bash
python -m unittest discover -s tests
```