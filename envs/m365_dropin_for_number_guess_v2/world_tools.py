"""World tools the template's declarative `serves` binding cannot express.

`serves` filters a collection. `compare` filters nothing - it *computes* an
answer from the episode's hidden target, which lives in the task's `data` and
which the sandbox must never see. So the world supplies the function.

Loaded via `FT_WORLD_TOOLS=m365_dropin_for_number_guess_v2.world_tools`.

Nothing here imports from the template. The contract is builtin types only:

    fn(task_data: dict, arguments: dict) -> str | dict

A bad argument is a plain `ValueError`. The template turns that into a 400,
which the policy reads as feedback rather than an outage - the same split the
template's own tools use, reached without a shared exception class.

REAL SYSTEM: this file disappears. `compare` becomes a registered MCP endpoint
and TCaaS routes to it; the gym-side proxy is unchanged because it only ever
forwards a name and arguments.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


def compare(task_data: Dict[str, Any], arguments: Dict[str, Any]) -> str:
    """Report where the hidden target sits relative to the caller's number."""
    number = arguments.get("number")
    if isinstance(number, bool) or not isinstance(number, int):
        raise ValueError("compare requires an integer 'number'.")

    target = task_data.get("target")
    if not isinstance(target, int):
        raise ValueError("This task carries no integer 'target' to compare against.")

    if target > number:
        return f"The target is higher than {number}."
    if target < number:
        return f"The target is lower than {number}."
    return f"The target is equal to {number}."


TOOLS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Any]] = {
    "compare": compare,
}
