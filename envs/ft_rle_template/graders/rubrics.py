"""Generic stand-in scorers, selected by declaration rather than by rubric id.

A rubric names a ``check`` and its ``check_params`` in ``catalog.json``:

```json
{"rubric_id": "cites-tickets", "check": "cites_grounded_entities",
 "check_params": {"pattern": "TKT-\\\\d+"}, "outcome": true}
```

Four checks cover both example worlds, which is the evidence they are actually
generic rather than one world's logic renamed. A world needing a fifth sets
``FT_WORLD_CHECKS`` rather than editing this file - see ``extensions.py``.

A rubric that declares no check, an unknown check, or a check without the params
it needs raises. The alternative - scoring it 0.0 with a polite note - is how a
dropped-in world silently trains on a flat reward signal, which is worse than a
crash because nothing tells you it happened.

REAL SYSTEM: replace this module with one LLM judge that reads each rubric's
``criteria``. The outcome gate below is the part that must survive.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import Item, RubricRef, RubricResult, Sample
from ..config import world_checks_module
from ..extensions import _module, registry


class RubricNotScorable(RuntimeError):
    """A rubric the offline mock cannot score. Never silently zero."""


def _required(rubric: RubricRef, key: str) -> Any:
    """Read a check param that has no safe default.

    Raising beats defaulting: a guessed id pattern that does not match the world
    scores zero on every episode, which reads as a bad policy rather than a bad
    config.
    """
    value = rubric.check_params.get(key)
    if not value:
        raise RubricNotScorable(
            f"Rubric {rubric.rubric_id!r} declares check {rubric.check!r} but no "
            f"{key!r} in check_params."
        )
    return value


def _expected(item: Item) -> Dict[str, Any]:
    try:
        return json.loads(item.expected) if item.expected else {}
    except json.JSONDecodeError:
        return {}


def _entities(text: str, pattern: str) -> Set[str]:
    return {m.lower() for m in re.findall(pattern, text or "")}


def _entities_in_tool_output(sample: Sample, pattern: str) -> Set[str]:
    seen: Set[str] = set()
    for message in sample.output_trajectory:
        if message.get("role") == "tool":
            seen |= _entities(str(message.get("content") or ""), pattern)
    return seen


def _tool_names(sample: Sample) -> List[str]:
    names: List[str] = []
    for message in sample.output_trajectory:
        for call in message.get("tool_calls") or []:
            names.append(call["function"]["name"])
    return names


def _calls(sample: Sample) -> List[Tuple[str, Dict[str, Any]]]:
    """Every tool call as ``(name, arguments)``, in order.

    The trajectory view a world check gets. Names alone are not enough for a
    check that cares about *what* was asked, only about *whether* it was.
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    for message in sample.output_trajectory:
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            out.append((fn.get("name", ""), args if isinstance(args, dict) else {}))
    return out


def cites_grounded_entities(
    rubric: RubricRef, item: Item, sample: Sample
) -> RubricResult:
    """Every id in the answer must have appeared in tool output.

    This is the anti-hallucination check, and it is world-agnostic because the
    id shape is a parameter.
    """
    pattern = _required(rubric, "pattern")
    claimed = _entities(sample.output_text, pattern)
    if not claimed:
        return RubricResult(
            rubric_id=rubric.rubric_id, score=0.0, reasoning="The answer cites no ids."
        )
    ungrounded = claimed - _entities_in_tool_output(sample, pattern)
    return RubricResult(
        rubric_id=rubric.rubric_id,
        score=0.0 if ungrounded else 1.0,
        reasoning=(
            f"Ungrounded ids: {sorted(ungrounded)}"
            if ungrounded
            else "Every cited id appeared in tool output."
        ),
    )


def matches_expected_entities(
    rubric: RubricRef, item: Item, sample: Sample
) -> RubricResult:
    """The cited ids must equal the reference set the caller supplied."""
    field = rubric.check_params.get("field", "expected_ids")
    pattern = _required(rubric, "pattern")
    expected = {str(v).lower() for v in (_expected(item).get(field) or [])}
    claimed = _entities(sample.output_text, pattern)
    return RubricResult(
        rubric_id=rubric.rubric_id,
        score=1.0 if expected and claimed == expected else 0.0,
        reasoning=f"expected={sorted(expected)} answered={sorted(claimed)}",
    )


def mentions_any(rubric: RubricRef, item: Item, sample: Sample) -> RubricResult:
    """A crude proxy for a presentation rubric: did the answer use the vocabulary."""
    words = [str(w).lower() for w in _required(rubric, "words")]
    text = (sample.output_text or "").lower()
    hit = [w for w in words if w in text]
    return RubricResult(
        rubric_id=rubric.rubric_id,
        score=1.0 if hit else 0.0,
        reasoning=f"matched={hit}",
    )


def called_tool(rubric: RubricRef, item: Item, sample: Sample) -> RubricResult:
    """A process rubric: was a required tool actually used.

    ``when`` makes it conditional on the task, so one rubric covers the samples
    that need the action and the ones that do not.
    """
    tool = str(_required(rubric, "tool"))
    when = rubric.check_params.get("when")
    if when and not _expected(item).get(str(when)):
        return RubricResult(
            rubric_id=rubric.rubric_id, score=1.0, reasoning="Not required."
        )
    called = tool in _tool_names(sample)
    return RubricResult(
        rubric_id=rubric.rubric_id,
        score=1.0 if called else 0.0,
        reasoning=f"{tool} called: {called}",
    )


CHECKS = {
    "cites_grounded_entities": cites_grounded_entities,
    "matches_expected_entities": matches_expected_entities,
    "mentions_any": mentions_any,
    "called_tool": called_tool,
}

REQUIRED_PARAMS: Dict[str, Tuple[str, ...]] = {
    "cites_grounded_entities": ("pattern",),
    "matches_expected_entities": ("pattern",),
    "mentions_any": ("words",),
    "called_tool": ("tool",),
}
"""Params a check cannot work without.

There is deliberately no default id pattern. A world whose ids are GUIDs, UPNs,
or SharePoint URLs would score zero against a guessed ``[A-Za-z]+-\\d+`` on every
episode, and a rubric that can never be earned is indistinguishable from a bad
policy in the training curve.
"""


def _world_check_adapter(name: str, fn: Any) -> Any:
    """Wrap a world check so it never has to import from this package.

    A world check is a plain function::

        fn(params: dict, expected: dict, answer: str,
           calls: list[tuple[str, dict]]) -> float | (float, str)

    Everything it receives is a builtin type, so ``world_checks.py`` is unit
    testable on its own and survives this template being refactored.
    """

    def scorer(rubric: RubricRef, item: Item, sample: Sample) -> RubricResult:
        result = fn(
            dict(rubric.check_params),
            _expected(item),
            sample.output_text or "",
            _calls(sample),
        )
        score, reasoning = result if isinstance(result, tuple) else (result, "")
        try:
            score = float(score)
        except (TypeError, ValueError) as exc:
            raise RubricNotScorable(
                f"World check {name!r} returned {result!r}, which is not a score."
            ) from exc
        if not 0.0 <= score <= 1.0:
            raise RubricNotScorable(
                f"World check {name!r} returned {score}, outside [0, 1]. Rubric "
                f"scores are normalised so weights mean what they say."
            )
        return RubricResult(
            rubric_id=rubric.rubric_id, score=score, reasoning=str(reasoning)
        )

    return scorer


def _world() -> Tuple[Dict[str, Any], Dict[str, Tuple[str, ...]]]:
    dotted = world_checks_module()
    raw = registry(dotted, "CHECKS")
    checks = {name: _world_check_adapter(name, fn) for name, fn in raw.items()}
    required: Dict[str, Tuple[str, ...]] = {}
    if dotted and raw:
        declared = getattr(_module(dotted), "REQUIRED_PARAMS", {}) or {}
        required = {k: tuple(v) for k, v in declared.items()}
    return checks, required


def all_checks() -> Dict[str, Any]:
    """Built-in checks plus the world's own, if it declared any."""
    merged = dict(CHECKS)
    world, _ = _world()
    merged.update(world)
    return merged


def unscorable_reason(
    check: Optional[str], check_params: Dict[str, Any]
) -> Optional[str]:
    """Why this rubric cannot be scored offline, or ``None`` if it can.

    Shared with catalog validation so a world that cannot be graded fails at
    load rather than part-way through a rollout.
    """
    known = all_checks()
    if check not in known:
        return (
            f"declares check {check!r}, which the offline mock does not implement; "
            f"known checks: {sorted(known)}"
        )
    world_checks, world_required = _world()
    required = (
        world_required.get(check, ())
        if check in world_checks
        else REQUIRED_PARAMS.get(check, ())
    )
    missing = [p for p in required if not check_params.get(p)]
    if missing:
        return f"declares check {check!r} without required check_params {missing}"
    return None


def score_all(
    rubrics: List[RubricRef], item: Item, sample: Sample
) -> List[RubricResult]:
    known = all_checks()
    results: List[RubricResult] = []
    for rubric in rubrics:
        reason = unscorable_reason(rubric.check, rubric.check_params)
        if reason is not None:
            raise RubricNotScorable(
                f"Rubric {rubric.rubric_id!r} {reason}. Point GRADERS_BASE_URL at a "
                f"real judge, or fix the world."
            )
        results.append(known[str(rubric.check)](rubric, item, sample))
    return results


def outcome_failed(rubrics: List[RubricRef], results: List[RubricResult]) -> bool:
    """True when every outcome rubric scored zero.

    Which rubrics gate is declared per rubric, so a new world sets ``outcome``
    in its catalog instead of editing a hardcoded id set here.
    """
    gating = {r.rubric_id for r in rubrics if r.outcome}
    scored = [r for r in results if r.rubric_id in gating]
    return bool(scored) and all(r.score <= 0.0 for r in scored)
