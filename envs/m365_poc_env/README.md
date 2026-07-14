# M365 POC Environment

This is a minimal OpenEnv-style environment for collaborating with M365 partners
on Foundry RLE integration patterns. It mirrors the upstream OpenEnv environment
layout:

```text
envs/m365_poc_env/
  pyproject.toml
  models.py
  logic.py
  harness-profile.json
  server/
    app.py
    m365_poc_environment.py
    Dockerfile
```

The task is intentionally small: submit a greeting message that includes both
`hello` and `M365`. There are no Microsoft Graph or M365 service calls yet. That
keeps the first collaboration loop focused on the runtime contract, harness
profile, and container shape.

## Local Run

```bash
cd envs/m365_poc_env
python -m venv .venv
. .venv/bin/activate
pip install -e .
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl -s http://127.0.0.1:8000/reset -X POST -H 'content-type: application/json' -d '{}'
curl -s http://127.0.0.1:8000/step -X POST -H 'content-type: application/json' \
  -d '{"action":{"message":"Hello from M365 and Foundry RLE!"}}'
```

## Container Run

OpenEnv environments keep their Dockerfile under `server/` and build from the
environment root:

```bash
cd envs/m365_poc_env
docker build -f server/Dockerfile -t m365-poc-env .
docker run --rm -p 8000:8000 m365-poc-env
```

## Contract

The server is created through OpenEnv's `create_app` wrapper, so it exposes the
standard OpenEnv HTTP/WebSocket surface for typed environments. The environment
implements:

- `reset(seed=None, episode_id=None, **kwargs)`
- `step(action: M365HelloWorldAction, timeout_s=None, **kwargs)`
- `state`
- `get_metadata()`

The matching Foundry RLE harness profile is in
[harness-profile.json](harness-profile.json).

## Next M365 Integration Hooks

- Replace the in-memory grader with a real M365 task verifier.
- Add Graph-backed task setup and cleanup behind the same OpenEnv contract.
- Extend the action schema with M365-specific actions such as draft mail, search
  calendar, or summarize document.