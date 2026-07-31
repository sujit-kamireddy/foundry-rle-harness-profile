"""Frontier Tuning RLE template: a generic FT world running as a Foundry RLE.

The package is a *template*. Nothing here is world-specific: skills, rubrics,
tasks, and user tools all arrive from TCaaS at runtime, and the model-facing
contract is published in ``harness-profile.json``, which the FT backend renders
per deployment (see ``profile.py``).

Aligned with ``envs/m365_number_guess_v2`` in the foundry-rle-harness-profile
repo, which is the reference for the TCaaS / tc_graders seams.
"""

__version__ = "0.1.0"
