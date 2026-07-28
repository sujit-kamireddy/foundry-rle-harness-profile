"""Puts ``envs/`` on ``sys.path`` so tests can import envs as packages."""

from __future__ import annotations

import pathlib
import sys

ENVS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "envs"
if str(ENVS_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVS_ROOT))
