# Harness Profile Reference

This folder contains the Foundry RLE harness profile reference and examples.

The harness profile is the environment-owned adapter between a raw OpenEnv runtime
and Foundry eval/training clients. OpenEnv defines the runtime protocol
(`reset`, `step`, `state`, `schema`, and optional tools). The harness profile
defines how Foundry should render observations, expose actions, classify terminal
or grader actions, fold rewards, and choose safe eval/training defaults.

Files:

- `schema.yaml` - documented YAML reference for `rle.harness/v0.1`.
- `harness-profile-review-page.md` - SharePoint-friendly RFC page for cross-team review.
- `sharepoint-publishing-plan.md` - recommended SharePoint page structure and publishing steps.
- `examples/finite-episodic-sokoban.yaml` - bounded game/task environment.
- `examples/one-shot-answer.yaml` - one-shot answer submission environment.
- `examples/multi-action-git-tools.yaml` - profile-declared multi-action tool environment.
- `examples/mcp-tool-finqa.yaml` - MCP tool environment with discovery and terminal answer.
- `examples/code-rl.yaml` - code-generation environment with test-based reward.