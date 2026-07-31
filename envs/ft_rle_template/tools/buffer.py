"""Per-episode write-ahead buffer: the reason training can touch real tools.

Training mode splits tool traffic by declared effect:

- ``read``  - goes to the real backend, then is *overlaid* with anything this
  episode has buffered, so the policy sees a coherent world.
- ``write`` - never reaches the backend. It is recorded here and answered with a
  synthesized result.

There is deliberately **no commit method**. Containment is structural, not a
convention someone can forget: nothing in this class can apply an effect, so no
code path in training mode can write to a customer system.

The read overlay is the subtle half. Without it, a policy that creates a case
note at step 2 and lists case notes at step 4 would not see its own write, and
would be trained against an incoherent world.
REAL SYSTEM: the argument-subset match below is a generic stand-in. Production
needs per-connector merge semantics, declared alongside the tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..tcaas.models import ToolSpec


@dataclass(frozen=True)
class PendingEffect:
    """One buffered write, in call order."""

    seq: int
    tool_name: str
    entity_type: str | None
    arguments: Dict[str, Any]
    payload: Dict[str, Any]

    def summary(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "tool_name": self.tool_name,
            "entity_type": self.entity_type,
            "arguments": self.arguments,
        }


@dataclass
class WriteAheadBuffer:
    """Buffered writes for exactly one episode. Never shared across instances."""

    _effects: List[PendingEffect] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._effects)

    def effects(self) -> List[PendingEffect]:
        return list(self._effects)

    def summary(self) -> List[Dict[str, Any]]:
        return [e.summary() for e in self._effects]

    def request_headers(self) -> Dict[str, str]:
        """Nothing to add: containment is local, so calls need no session tag."""
        return {}

    def record(self, spec: ToolSpec, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Capture a write and synthesize the result the caller would have got.

        The synthesized record carries the caller's arguments plus an id, so the
        overlay can surface it to later reads and the grader can see the write
        happened.
        """
        seq = len(self._effects) + 1
        entity = spec.produces_entity or f"{spec.tool_name}_result"
        record: Dict[str, Any] = dict(arguments or {})
        record[f"{entity}_id"] = f"{entity}-wal-{seq:04d}"
        payload = {"entity_type": entity, "items": [record]}
        self._effects.append(
            PendingEffect(
                seq=seq,
                tool_name=spec.tool_name,
                entity_type=entity,
                arguments=dict(arguments or {}),
                payload=payload,
            )
        )
        return payload

    def overlay(self, spec: ToolSpec, result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge buffered writes into a real read result.

        A pending effect is surfaced when its entity type is one the read tool
        lists *and* the read's arguments are a subset of the buffered record - so
        listing notes for one employee does not surface another's.
        """
        if not spec.overlay_entities or not self._effects:
            return result

        arguments = result.get("_request_arguments") or {}
        merged = dict(result)
        items = list(merged.get("items") or [])
        for effect in self._effects:
            if effect.entity_type not in spec.overlay_entities:
                continue
            for candidate in effect.payload.get("items", []):
                if _matches(arguments, candidate):
                    items.append(candidate)
        merged["items"] = items
        merged.pop("_request_arguments", None)
        return merged


def _matches(request_arguments: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    """True when every filter in the read request is satisfied by the record."""
    for key, value in (request_arguments or {}).items():
        if key in candidate and candidate[key] != value:
            return False
    return True


def render(payload: Dict[str, Any]) -> str:
    """Compact JSON is what the policy reads as tool output."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)
