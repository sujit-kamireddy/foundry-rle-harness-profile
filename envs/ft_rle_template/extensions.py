"""Optional world-supplied Python, loaded by module path.

Most FT worlds are data: a tool reads records, a rubric checks the answer against
them, and ``catalog.json`` says so declaratively. Some are not. A tool can be
*computational* - it answers from the episode's own task data rather than a
collection - and a rubric can score something no string comparison reaches, like
how few steps a solve took.

Those worlds get two hooks rather than a fork of this template:

``FT_WORLD_TOOLS``
    module exporting ``TOOLS: {tool_name: fn(task_data, arguments)}``

``FT_WORLD_CHECKS``
    module exporting ``CHECKS: {check_name: fn(params, expected, answer, calls)}``
    and optionally ``REQUIRED_PARAMS: {check_name: (param, ...)}``

Both are **offline-mock concerns only**, exactly like ``ToolSpec.serves``. In
production the tool is a real MCP endpoint and the rubric is read by an LLM
judge, so neither hook is loaded.

A world module imports nothing from this package. It exchanges plain dicts,
lists, and floats, so a world can never be broken by a refactor in here.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any, Dict


class WorldExtensionError(RuntimeError):
    """A world named an extension module that could not be loaded.

    Raised rather than ignored: silently continuing would serve a world whose
    tools are missing, which surfaces much later as a policy that cannot act.
    """


@lru_cache(maxsize=8)
def _module(dotted_path: str) -> Any:
    try:
        return import_module(dotted_path)
    except Exception as exc:  # noqa: BLE001 - any import failure is fatal here
        raise WorldExtensionError(
            f"could not import world extension module {dotted_path!r}: {exc}"
        ) from exc


def registry(dotted_path: str, attribute: str) -> Dict[str, Any]:
    """Read one mapping from a world module, or ``{}`` when none is configured."""
    if not dotted_path:
        return {}
    module = _module(dotted_path)
    found = getattr(module, attribute, None)
    if found is None:
        return {}
    if not isinstance(found, dict):
        raise WorldExtensionError(
            f"{dotted_path}.{attribute} must be a dict, got {type(found).__name__}"
        )
    return dict(found)
