"""The write-ahead buffer is the containment boundary, so it gets the most tests."""

from __future__ import annotations

from ft_rle_template.tcaas.models import ToolSpec
from ft_rle_template.tools.buffer import WriteAheadBuffer

WRITE = ToolSpec(
    tool_name="create_case_note",
    effect="write",
    produces_entity="case_note",
)
READ = ToolSpec(
    tool_name="list_case_notes",
    effect="read",
    overlay_entities=["case_note"],
)
UNRELATED_READ = ToolSpec(tool_name="get_employee_record", effect="read")


def test_buffer_has_no_commit_path():
    """Containment must be structural: no method can apply a buffered effect."""
    surface = {name for name in dir(WriteAheadBuffer) if not name.startswith("_")}
    assert not surface & {"commit", "flush", "apply", "drain", "send", "replay"}


def test_write_is_recorded_and_answered():
    buffer = WriteAheadBuffer()
    payload = buffer.record(WRITE, {"employee_id": "E-1", "note": "hello"})
    assert len(buffer) == 1
    assert payload["entity_type"] == "case_note"
    assert payload["items"][0]["case_note_id"].startswith("case_note-wal-")
    assert payload["items"][0]["note"] == "hello"


def test_read_after_write_sees_the_buffered_record():
    """Without this the policy trains against a world that forgot its own writes."""
    buffer = WriteAheadBuffer()
    buffer.record(WRITE, {"employee_id": "E-1", "note": "call back"})
    merged = buffer.overlay(
        READ, {"items": [], "_request_arguments": {"employee_id": "E-1"}}
    )
    assert [i["note"] for i in merged["items"]] == ["call back"]


def test_overlay_respects_read_filters():
    """One employee's buffered note must not surface under another's query."""
    buffer = WriteAheadBuffer()
    buffer.record(WRITE, {"employee_id": "E-1", "note": "mine"})
    merged = buffer.overlay(
        READ, {"items": [], "_request_arguments": {"employee_id": "E-2"}}
    )
    assert merged["items"] == []


def test_overlay_skips_unrelated_entities():
    buffer = WriteAheadBuffer()
    buffer.record(WRITE, {"employee_id": "E-1", "note": "mine"})
    result = {"items": [{"employee_id": "E-1"}], "_request_arguments": {}}
    assert buffer.overlay(UNRELATED_READ, result) == result


def test_overlay_strips_the_internal_request_key():
    """`_request_arguments` is plumbing; it must not reach the policy."""
    buffer = WriteAheadBuffer()
    buffer.record(WRITE, {"employee_id": "E-1", "note": "x"})
    merged = buffer.overlay(READ, {"items": [], "_request_arguments": {}})
    assert "_request_arguments" not in merged


def test_buffers_are_per_episode():
    first, second = WriteAheadBuffer(), WriteAheadBuffer()
    first.record(WRITE, {"employee_id": "E-1", "note": "x"})
    assert len(second) == 0
    assert second.overlay(READ, {"items": [], "_request_arguments": {}})["items"] == []


def test_effects_preserve_call_order():
    buffer = WriteAheadBuffer()
    buffer.record(WRITE, {"employee_id": "E-1", "note": "first"})
    buffer.record(WRITE, {"employee_id": "E-1", "note": "second"})
    assert [e.seq for e in buffer.effects()] == [1, 2]
    assert [e["arguments"]["note"] for e in buffer.summary()] == ["first", "second"]
