"""Serves the world's tools from catalog data, driven entirely by declarations.

Nothing here knows what an employee or a ticket is. A tool is served by its
``serves`` binding in ``catalog.json``:

```json
{
  "tool_name": "get_employee_record",
  "effect": "read",
  "serves": {"dataset": "employees", "entity_type": "employee",
             "required": ["employee_id"]}
}
```

That is the whole contract for adding a tool: a schema, an effect, and a
binding. No Python changes, which is the property that makes this folder a
template rather than one worked example.

A tool that *computes* rather than filters - answering from the episode's own
task data - has nothing to bind to. Those worlds set ``FT_WORLD_TOOLS`` to a
module exporting ``TOOLS: {name: fn(task_data, arguments)}``, which is tried
before ``serves``.

REAL SYSTEM: these are the customer's registered MCP endpoints. Replace
``dispatch`` with MCP ``tools/call``; ``serves`` and ``TOOLS`` are both ignored
and the gym-side proxy is untouched.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List

from .catalog import load_catalog
from .models import ToolBinding, ToolSpec
from ..config import world_tools_module
from ..extensions import registry

_ids = itertools.count(1)

_COMMITTED: List[Dict[str, Any]] = []
"""Writes that actually landed.

In training the gym buffers writes and never reaches here, so this list staying
empty across a training rollout is the leak test.
"""


class ToolRejected(ValueError):
    """Bad arguments. A signal for the policy, never an infrastructure fault."""


class ToolUnservable(RuntimeError):
    """The world declared a tool the offline mock cannot serve.

    Raised rather than swallowed: a tool that silently answers "unknown" would
    let a broken world train to completion and learn nothing.
    """


def _spec(tool_name: str) -> ToolSpec:
    for spec in load_catalog().tools():
        if spec.tool_name == tool_name:
            return spec
    raise ToolRejected(f"Unknown tool {tool_name!r}.")


def _rows(dataset: str) -> List[Dict[str, Any]]:
    return list(load_catalog().datasets().get(dataset, []))


def _read(binding: ToolBinding, arguments: Dict[str, Any]) -> Dict[str, Any]:
    for field in binding.required:
        if not arguments.get(field):
            raise ToolRejected(f"{field} is required.")

    items = _rows(binding.dataset)
    for field, value in binding.where.items():
        items = [row for row in items if row.get(field) == value]

    for field in list(binding.required) + list(binding.filters):
        supplied = arguments.get(field)
        if supplied in (None, ""):
            continue
        items = [row for row in items if _eq(row.get(field), supplied)]

    if binding.not_found_is_error and not items:
        missing = ", ".join(f"{f}={arguments.get(f)!r}" for f in binding.required)
        raise ToolRejected(f"No {binding.entity_type or binding.dataset} for {missing}.")

    return {"entity_type": binding.entity_type or binding.dataset, "items": items}


def _write(binding: ToolBinding, arguments: Dict[str, Any]) -> Dict[str, Any]:
    for field in binding.required:
        if not arguments.get(field):
            raise ToolRejected(f"{field} is required.")

    for ref in binding.references:
        field, dataset, key = ref.get("field"), ref.get("dataset"), ref.get("key")
        value = arguments.get(str(field))
        if not any(_eq(row.get(str(key)), value) for row in _rows(str(dataset))):
            raise ToolRejected(f"No {dataset} record for {field}={value!r}.")

    record = dict(arguments)
    if binding.id_field:
        record[binding.id_field] = f"{binding.id_prefix}-{next(_ids):04d}"

    _COMMITTED.append({"dataset": binding.dataset, "record": record})
    load_catalog().datasets().setdefault(binding.dataset, []).append(record)
    return {"entity_type": binding.entity_type or binding.dataset, "items": [record]}


def _eq(left: Any, right: Any) -> bool:
    """Case-insensitive string compare, so ids are not a source of flakiness."""
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().lower() == right.strip().lower()
    return left == right


def _world_tools() -> Dict[str, Any]:
    return registry(world_tools_module(), "TOOLS")


def _task_data(task_id: str | None) -> Dict[str, Any]:
    """The episode's own task data, which a computational tool answers from.

    The gym already sends ``task_id`` on every call, so this needs no new wire
    field - it is the scope a real user-tool endpoint would receive anyway.
    """
    if not task_id:
        raise ToolRejected("task_id is required to serve this tool.")
    try:
        return dict(load_catalog().find_task(task_id).data)
    except KeyError as exc:
        raise ToolRejected(str(exc)) from exc


def dispatch(
    tool_name: str, arguments: Dict[str, Any], task_id: str | None = None
) -> Dict[str, Any]:
    spec = _spec(tool_name)
    arguments = arguments or {}

    world_tool = _world_tools().get(tool_name)
    if world_tool is not None:
        try:
            result = world_tool(_task_data(task_id), arguments)
        except ValueError as exc:
            # World modules import nothing from this package, so they reject with
            # a plain ValueError and the 400/500 split is decided here.
            raise ToolRejected(str(exc)) from exc
        if isinstance(result, str):
            result = {"entity_type": tool_name, "items": [{"result": result}]}
        return result

    binding = spec.serves
    if binding is None:
        raise ToolUnservable(
            f"Tool {tool_name!r} declares no `serves` binding and no world tool "
            f"implements it, so the offline mock cannot run it. Add a binding, set "
            f"FT_WORLD_TOOLS, or point TCAAS_BASE_URL at a real service."
        )
    return _write(binding, arguments) if binding.appends else _read(binding, arguments)


def committed_writes() -> List[Dict[str, Any]]:
    """Test hook: what actually landed. Must stay empty during training."""
    return list(_COMMITTED)


def reset_store() -> None:
    """Drop committed writes and the rows they appended to the in-memory world."""
    for entry in _COMMITTED:
        rows = load_catalog().datasets().get(entry["dataset"], [])
        if entry["record"] in rows:
            rows.remove(entry["record"])
    _COMMITTED.clear()
