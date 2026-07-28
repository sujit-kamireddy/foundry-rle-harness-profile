"""Catalog, tenant scoping, and mock TCaaS endpoints. No OpenEnv needed."""

import unittest

from fastapi.testclient import TestClient

from m365_number_guess_v2.tcaas.app import create_tcaas_app
from m365_number_guess_v2.tcaas.catalog import Catalog
from m365_number_guess_v2.tcaas.identity import TenantContext


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog.load()

    def test_same_seed_always_picks_same_task(self):
        first = self.catalog.pick_task("train", 42)
        second = self.catalog.pick_task("train", 42)
        self.assertEqual(first.task_id, second.task_id)
        self.assertEqual(first.data, second.data)

    def test_seeds_wrap_around_the_split(self):
        size = self.catalog.split_size("validation")
        wrapped = self.catalog.pick_task("validation", size + 3)
        self.assertEqual(
            self.catalog.pick_task("validation", 3).task_id, wrapped.task_id
        )

    def test_seeds_zero_to_n_cover_the_split_exactly_once(self):
        size = self.catalog.split_size("validation")
        picked = {
            self.catalog.pick_task("validation", seed).task_id for seed in range(size)
        }
        self.assertEqual(size, len(picked))

    def test_bundle_carries_the_skill_workflow_and_its_rubrics(self):
        bundle = self.catalog.pick_task("train", 0)
        self.assertTrue(bundle.skill)
        self.assertTrue(bundle.rubrics)
        for rubric in bundle.rubrics:
            self.assertEqual(bundle.skill_id, rubric.skill_id)

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(KeyError):
            self.catalog.pick_task("nope", 0)

    def test_dangling_skill_reference_fails_at_load(self):
        raw = {
            "tenant": {"tenant_id": "t", "user_id": "u"},
            "skills": [{"skill_id": "a", "name": "A", "workflow": "w"}],
            "rubrics": [],
            "tasks": {
                "train": [
                    {"task_id": "t", "skill_id": "ghost", "user_query": "q", "data": {}}
                ]
            },
        }
        with self.assertRaises(ValueError):
            Catalog(raw)


class TCaaSServiceTests(unittest.TestCase):
    def setUp(self):
        self.tenant = Catalog.load().tenant
        self.client = TestClient(create_tcaas_app(), headers=self.tenant.headers())

    def test_tasks_endpoint_is_deterministic(self):
        first = self.client.get("/tasks", params={"split": "train", "seed": 7}).json()
        second = self.client.get("/tasks", params={"split": "train", "seed": 7}).json()
        self.assertEqual(first, second)

    def test_unknown_split_returns_404(self):
        response = self.client.get("/tasks", params={"split": "nope", "seed": 0})
        self.assertEqual(404, response.status_code)

    def test_another_tenant_is_refused(self):
        other = TenantContext(tenant_id="tenant-other", user_id="user-other")
        response = TestClient(create_tcaas_app(), headers=other.headers()).get(
            "/tasks", params={"split": "train", "seed": 0}
        )
        self.assertEqual(403, response.status_code)

    def test_missing_tenant_headers_are_refused(self):
        response = TestClient(create_tcaas_app()).get("/tools")
        self.assertEqual(422, response.status_code)

    def test_user_tool_runs_against_task_data(self):
        bundle = self.client.get("/tasks", params={"split": "train", "seed": 0}).json()
        target = bundle["data"]["target"]

        below = self._call_tool("compare", bundle["task_id"], {"number": target - 1})
        exact = self._call_tool("compare", bundle["task_id"], {"number": target})

        self.assertIn("higher", below.json()["output"])
        self.assertIn("equal", exact.json()["output"])

    def test_bad_tool_arguments_are_a_rejection_not_an_outage(self):
        response = self._call_tool("compare", "guess-train-000", {"number": "five"})
        self.assertEqual(400, response.status_code)

    def test_unknown_tool_returns_404(self):
        response = self._call_tool("teleport", "guess-train-000", {})
        self.assertEqual(404, response.status_code)

    def test_tool_schemas_are_openai_format(self):
        tools = self.client.get("/tools").json()
        self.assertEqual({"compare"}, {tool["function"]["name"] for tool in tools})

    def _call_tool(self, tool_name, task_id, arguments):
        return self.client.post(
            f"/tools/{tool_name}", json={"task_id": task_id, "arguments": arguments}
        )


if __name__ == "__main__":
    unittest.main()
