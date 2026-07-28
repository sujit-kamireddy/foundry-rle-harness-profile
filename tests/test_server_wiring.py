"""The gym server's route wiring: the TCaaS and grader mocks must stay reachable.

Both mocks are sub-apps of the gym's FastAPI app, since the image runs a single
uvicorn process. OpenEnv reserves ``/{env_name}/...`` routes at the root, which
silently shadowed the mocks at ``/tcaas`` and ``/graders`` - hence the ``/mock``
prefix. Skips without ``openenv``, which lives only in the container image.
"""

import unittest

try:
    from m365_number_guess_v2.server.app import app
except ImportError:
    app = None


@unittest.skipIf(app is None, "openenv package is not installed")
class MountTests(unittest.TestCase):
    def _paths(self):
        return [getattr(route, "path", "") for route in app.routes]

    def test_both_mocks_are_mounted(self):
        self.assertIn("/mock/tcaas", self._paths())
        self.assertIn("/mock/graders", self._paths())

    def test_mounts_are_not_shadowed_by_reserved_routes(self):
        reserved = {p for p in self._paths() if "{env_name}" in p}
        self.assertTrue(reserved, "expected OpenEnv to reserve /{env_name} routes")

        for mount in ("/mock/tcaas", "/mock/graders"):
            env_name = mount.strip("/").split("/")[0]
            for path in {p.replace("{env_name}", env_name) for p in reserved}:
                self.assertFalse(
                    path.startswith(f"{mount}/"),
                    f"mount {mount} is shadowed by reserved route {path}",
                )

    def test_gym_endpoints_still_exist(self):
        for path in ("/reset", "/step", "/health"):
            self.assertIn(path, self._paths())


if __name__ == "__main__":
    unittest.main()
