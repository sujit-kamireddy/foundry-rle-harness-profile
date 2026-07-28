"""Tool layer: base tools ship with the sandbox, user tools come from TCaaS."""

import unittest

import httpx
from fastapi.testclient import TestClient

from m365_number_guess_v2.logic import MAX_NUMBER, MIN_NUMBER
from m365_number_guess_v2.tcaas.app import create_tcaas_app
from m365_number_guess_v2.tcaas.catalog import Catalog
from m365_number_guess_v2.tcaas.client import TCaaSClient
from m365_number_guess_v2.tools.base import ToolTransportError
from m365_number_guess_v2.tools.local import LocalToolExecutor
from m365_number_guess_v2.tools.proxy import TCaaSToolExecutor
from m365_number_guess_v2.tools.registry import ToolRegistry


class _StubClient:
    """Stands in for TCaaSClient so failures can be provoked without a server."""

    def __init__(self, schemas=None, error=None, output="ok"):
        self._schemas = schemas or []
        self._error = error
        self._output = output

    def list_tool_schemas(self):
        return self._schemas

    def call_tool(self, task_id, tool_name, arguments):
        if self._error is not None:
            raise self._error
        return self._output


class LocalToolTests(unittest.TestCase):
    def setUp(self):
        self.executor = LocalToolExecutor(MIN_NUMBER, MAX_NUMBER)

    def test_guess_records_the_answer_without_judging_it(self):
        result = self.executor.execute("guess", {"number": 3}, call_id="c1")
        self.assertTrue(result.success)
        self.assertEqual("c1", result.call_id)
        self.assertIn("3", result.output)

    def test_out_of_range_guess_is_rejected_not_raised(self):
        result = self.executor.execute("guess", {"number": MAX_NUMBER + 1})
        self.assertFalse(result.success)
        self.assertEqual("invalid_argument", result.error)

    def test_non_integer_guess_is_rejected(self):
        self.assertFalse(self.executor.execute("guess", {"number": "three"}).success)

    def test_sandbox_exposes_only_its_base_tools(self):
        names = {s["function"]["name"] for s in self.executor.list_tool_schemas()}
        self.assertEqual({"guess"}, names)


class ProxiedToolTests(unittest.TestCase):
    def test_output_is_relayed_verbatim(self):
        executor = TCaaSToolExecutor(
            _StubClient(output="The target is higher than 5."), "task-1"
        )
        result = executor.execute("compare", {"number": 5}, call_id="c9")
        self.assertTrue(result.success)
        self.assertEqual("c9", result.call_id)
        self.assertIn("higher", result.output)

    def test_rejected_call_is_feedback(self):
        rejection = httpx.HTTPStatusError(
            "bad",
            request=httpx.Request("POST", "http://x/tools/compare"),
            response=httpx.Response(
                400, json={"detail": "compare requires an integer 'number'."}
            ),
        )
        executor = TCaaSToolExecutor(_StubClient(error=rejection), "task-1")

        result = executor.execute("compare", {"number": "five"})

        self.assertFalse(result.success)
        self.assertEqual("invalid_argument", result.error)
        self.assertIn("integer", result.output)

    def test_transport_failure_raises_instead_of_becoming_a_signal(self):
        outage = httpx.ConnectError("connection refused")
        executor = TCaaSToolExecutor(_StubClient(error=outage), "task-1")
        with self.assertRaises(ToolTransportError):
            executor.execute("compare", {"number": 5})

    def test_server_error_also_raises(self):
        boom = httpx.HTTPStatusError(
            "boom",
            request=httpx.Request("POST", "http://x/tools/compare"),
            response=httpx.Response(500),
        )
        executor = TCaaSToolExecutor(_StubClient(error=boom), "task-1")
        with self.assertRaises(ToolTransportError):
            executor.execute("compare", {"number": 5})


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.base = LocalToolExecutor(MIN_NUMBER, MAX_NUMBER)
        schemas = [
            {"type": "function", "function": {"name": "compare", "parameters": {}}}
        ]
        self.user = TCaaSToolExecutor(
            _StubClient(schemas=schemas, output="higher"), "task-1"
        )

    def test_user_tools_extend_the_base_surface(self):
        registry = ToolRegistry(self.base, self.user)
        self.assertEqual({"guess", "compare"}, set(registry.tool_names))

    def test_dispatch_reaches_the_owning_backend(self):
        registry = ToolRegistry(self.base, self.user)
        self.assertIn(
            "Answer submitted", registry.execute("guess", {"number": 4}).output
        )
        self.assertEqual("higher", registry.execute("compare", {"number": 4}).output)

    def test_sandbox_works_with_no_user_tools_at_all(self):
        registry = ToolRegistry(self.base)
        self.assertEqual(["guess"], registry.tool_names)

    def test_unknown_tool_is_rejected_not_raised(self):
        result = ToolRegistry(self.base).execute("teleport", {})
        self.assertFalse(result.success)
        self.assertEqual("unknown_tool", result.error)


class ProxyAgainstMockServiceTests(unittest.TestCase):
    """End-to-end over HTTP: client -> mock TCaaS -> user tool implementation."""

    def setUp(self):
        catalog = Catalog.load()
        self.bundle = catalog.pick_task("train", 0)
        self.client = TCaaSClient(
            tenant=catalog.tenant,
            http=TestClient(create_tcaas_app()),
        )

    def test_compare_narrows_the_range_through_the_full_stack(self):
        executor = TCaaSToolExecutor(self.client, self.bundle.task_id)
        target = self.bundle.data["target"]

        result = executor.execute("compare", {"number": target})

        self.assertTrue(result.success)
        self.assertIn("equal", result.output)

    def test_schemas_come_from_the_service(self):
        executor = TCaaSToolExecutor(self.client, self.bundle.task_id)
        names = {s["function"]["name"] for s in executor.list_tool_schemas()}
        self.assertEqual({"compare"}, names)


if __name__ == "__main__":
    unittest.main()
