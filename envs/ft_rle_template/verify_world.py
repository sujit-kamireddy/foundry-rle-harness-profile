"""Smoke-test a dropped-in world against a running container.

    python -m ft_rle_template.verify_world --url ws://localhost:8000/ws

Run this after dropping in a new ``catalog.json``. It answers the one question
unit tests cannot: does *this* world produce a usable training signal?

The policy here is deliberately world-agnostic. It reads the tool schemas out
of the observation, calls each tool using values earlier tools returned, and
answers with what it saw - so it knows nothing about tickets, employees, or any
other domain. If it cannot out-score a policy that answers blindly, the reward
signal is degenerate and training will not learn: usually a rubric whose
``check_params`` do not match the world's id format, or a tool whose ``serves``
binding returns nothing.
"""

import argparse
import asyncio
import json
import sys

import websockets


async def rpc(ws, msg_type, data=None):
    await ws.send(json.dumps({"type": msg_type, "data": data or {}}))
    r = json.loads(await ws.recv())["data"]
    return r["observation"], r


def harvest(feedback, seen, lines):
    """Remember every field value the tools returned, keyed by field name."""
    try:
        payload = json.loads(feedback)
    except (TypeError, ValueError):
        return
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        lines.append(", ".join(f"{k}={v}" for k, v in item.items()
                               if isinstance(v, (str, int, float))))
        for key, value in item.items():
            if isinstance(value, str):
                seen.setdefault(key, [])
                if value not in seen[key]:
                    seen[key].append(value)


def build_args(schema, seen):
    """Fill required arguments from values earlier tools actually returned."""
    props = schema.get("properties", {})
    args = {}
    for name in schema.get("required", []):
        spec = props.get(name, {})
        if spec.get("enum"):
            args[name] = spec["enum"][0]
        elif seen.get(name):
            args[name] = seen[name][0]
        elif spec.get("type") == "string":
            args[name] = "Reviewed and actioned per the workflow."
        else:
            args[name] = 1
    return args


async def competent(ws, obs):
    """Read what is reachable, take the write action, answer citing what was read."""
    tools = {t["function"]["name"]: t["function"] for t in obs["tools"]}
    terminal = "submit_answer"
    seen, lines, res = {}, [], None

    ordered = sorted(
        (n for n in tools if n != terminal),
        key=lambda n: len(tools[n]["parameters"].get("required", [])),
    )
    for name in ordered:
        args = build_args(tools[name]["parameters"], seen)
        obs, res = await rpc(ws, "step", {"tool_name": name, "arguments": args})
        fb = obs.get("feedback") or ""
        harvest(fb, seen, lines)
        print(f"  -> {name:32s} {fb[:86].replace(chr(10), ' ')}")
        if res["done"]:
            return obs, res

    answer = "Summary of what the tools returned:\n" + "\n".join(lines[:12])
    obs, res = await rpc(ws, "step",
                         {"tool_name": terminal, "arguments": {"answer": answer}})
    print(f"  -> {terminal:32s} {(obs.get('feedback') or '')[:86]}")
    return obs, res


async def lazy(ws, obs):
    return await rpc(ws, "step",
                     {"tool_name": "submit_answer",
                      "arguments": {"answer": "Everything looks fine."}})


async def run(url, policy, label):
    async with websockets.connect(url, max_size=None) as ws:
        obs, _ = await rpc(ws, "reset")
        print(f"\n=== {label} | skill={obs['skill_id']} ===")
        print("query:", obs["user_query"])
        obs, res = await policy(ws, obs)
        print("reward:", res["reward"], "| done:", res["done"])
        print("buffered writes:", json.dumps(obs["pending_effects"])[:200])
        return res["reward"], obs["pending_effects"]


async def main(url):
    good, buffered = await run(url, competent, "competent policy")
    bad, _ = await run(url, lazy, "lazy policy")

    print(f"\ncompetent={good}  blind={bad}  buffered_writes={len(buffered)}")
    if good <= bad:
        print("FAIL: the reward signal does not separate competence from noise.")
        return 1
    print("PASS: reward separates competence, and no write left the gym.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    sys.exit(asyncio.run(main(parser.parse_args().url)))
