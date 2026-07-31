"""Dual-mode dispatch, and the rejection-versus-outage split."""

from __future__ import annotations

import pytest

from ft_rle_template.tcaas import tools as mock_tools
from ft_rle_template.tcaas.client import TCaaSClient
from ft_rle_template.tools.base import ToolTransportError
from ft_rle_template.tools.local import LocalToolExecutor
from ft_rle_template.tools.proxy import TCaaSToolExecutor
from ft_rle_template.tools.registry import ToolRegistry


def _executor(tcaas: TCaaSClient, training: bool) -> TCaaSToolExecutor:
    bundle = tcaas.pick_task("train", 0)
    return TCaaSToolExecutor(
        tcaas, bundle.task_id, bundle.tools, training=training
    )


def test_training_buffers_writes(tcaas):
    executor = _executor(tcaas, training=True)
    result = executor.execute(
        "create_case_note", {"employee_id": "e-1041", "note": "follow up"}
    )
    assert result.success and result.buffered
    assert len(executor.buffer) == 1
    assert mock_tools.committed_writes() == []


def test_inference_commits_writes(tcaas):
    """The same tool, the other mode: the write must actually land."""
    executor = _executor(tcaas, training=False)
    result = executor.execute(
        "create_case_note", {"employee_id": "e-1041", "note": "real"}
    )
    assert result.success and not result.buffered
    assert [w["record"]["note"] for w in mock_tools.committed_writes()] == ["real"]


def test_training_reads_hit_the_real_backend(tcaas):
    executor = _executor(tcaas, training=True)
    result = executor.execute("get_employee_record", {"employee_id": "e-1041"})
    assert result.success
    assert "e-1041" in result.output


def test_training_read_sees_its_own_buffered_write(tcaas):
    executor = _executor(tcaas, training=True)
    executor.execute("create_case_note", {"employee_id": "e-1041", "note": "buffered"})
    result = executor.execute("list_case_notes", {"employee_id": "e-1041"})
    assert "buffered" in result.output
    assert mock_tools.committed_writes() == []


def test_bad_arguments_are_policy_signal_not_an_outage(tcaas):
    """A rejected call must keep the episode alive so the policy can recover."""
    executor = _executor(tcaas, training=True)
    result = executor.execute("get_employee_record", {"employee_id": "nobody"})
    assert not result.success
    assert result.error == "invalid_argument"


def test_unreachable_backend_raises(tcaas):
    """An outage must never be scored; it has to surface as an infrastructure fault."""
    bundle = tcaas.pick_task("train", 0)
    dead = TCaaSClient(base_url="http://127.0.0.1:9")
    executor = TCaaSToolExecutor(dead, bundle.task_id, bundle.tools, training=True)
    with pytest.raises(ToolTransportError):
        executor.execute("get_employee_record", {"employee_id": "e-1041"})


def test_unknown_tool_is_rejected_not_raised(tcaas):
    executor = _executor(tcaas, training=True)
    result = executor.execute("delete_everything", {})
    assert not result.success and result.error == "unknown_tool"


def test_registry_exposes_user_tools_and_the_terminal_tool(tcaas):
    bundle = tcaas.pick_task("train", 0)
    registry = ToolRegistry(
        LocalToolExecutor(),
        TCaaSToolExecutor(tcaas, bundle.task_id, bundle.tools, training=True),
    )
    names = [s["function"]["name"] for s in registry.list_tool_schemas()]
    assert "submit_answer" in names
    assert {t.tool_name for t in bundle.tools} <= set(names)


def test_registry_routes_the_terminal_tool_locally(tcaas):
    """submit_answer must never leave the sandbox."""
    bundle = tcaas.pick_task("train", 0)
    registry = ToolRegistry(
        LocalToolExecutor(),
        TCaaSToolExecutor(tcaas, bundle.task_id, bundle.tools, training=True),
    )
    result = registry.execute("submit_answer", {"answer": "done"})
    assert result.success
    assert "submitted" in result.output.lower()


def test_terminal_tool_rejects_an_empty_answer(tcaas):
    registry = ToolRegistry(
        LocalToolExecutor(),
        TCaaSToolExecutor(tcaas, "t", [], training=True),
    )
    assert not registry.execute("submit_answer", {}).success
