"""Fixture world extensions, used only by ``test_extensions.py``.

Deliberately trivial and deliberately dependency-free: the point of the hook is
that a world module imports nothing from this package, so this file must not
either. If an import from ``ft_rle_template`` ever appears here, the hook has
stopped being a hook and become a plugin API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def echo_secret(task_data: Dict[str, Any], arguments: Dict[str, Any]) -> str:
    """Answers from task data, which no dataset binding could reach."""
    prefix = arguments.get("prefix")
    if not isinstance(prefix, str):
        raise ValueError("echo_secret requires a string 'prefix'.")
    return f"{prefix}:{task_data.get('secret')}"


TOOLS = {"echo_secret": echo_secret}


def call_count(
    params: Dict[str, Any],
    expected: Dict[str, Any],
    answer: str,
    calls: List[Tuple[str, Dict[str, Any]]],
) -> Tuple[float, str]:
    """Scores on trajectory length, which no built-in check looks at."""
    target = int(params["target_calls"])
    return min(1.0, len(calls) / target), f"{len(calls)} calls"


def out_of_range(
    params: Dict[str, Any],
    expected: Dict[str, Any],
    answer: str,
    calls: List[Tuple[str, Dict[str, Any]]],
) -> float:
    """Returns an illegal score, so the adapter's guard has something to catch."""
    return 7.5


CHECKS = {"call_count": call_count, "out_of_range": out_of_range}

REQUIRED_PARAMS = {"call_count": ("target_calls",)}
