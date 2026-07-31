# The world layer. Everything M365 adds sits in this one thin image on top of a
# generic, world-agnostic base.
#
#   docker build -t ft-rle-template:latest -f ../ft_rle_template/server/Dockerfile ../ft_rle_template
#   docker build -t m365-number-guess:latest .
#
# Two images rather than one because they have different lifecycles: the base is
# built once per template release and shared by every world, while this layer is
# rebuilt whenever content changes. It also keeps the claim honest - if the world
# needed anything from inside the base, this file could not be four COPY lines.

ARG BASE_IMAGE=ft-rle-template:latest
FROM ${BASE_IMAGE}

# PYTHONPATH is already /app in the base, so this lands `m365_dropin_for_number_guess_v2` as an
# importable package and the two FT_WORLD_* module paths below resolve.
COPY . /app/m365_dropin_for_number_guess_v2

ENV FT_CATALOG_PATH=/app/m365_dropin_for_number_guess_v2/catalog.json \
    FT_WORLD_TOOLS=m365_dropin_for_number_guess_v2.world_tools \
    FT_WORLD_CHECKS=m365_dropin_for_number_guess_v2.world_checks

# World-level limits. These are also rendered into harness-profile.json, so the
# two must be changed together or the profile will describe an env that does not
# exist - which is why `make profile` regenerates it from these same values.
ENV FT_MAX_STEPS_PER_EPISODE=10 \
    FT_EPISODE_TIMEOUT_S=60 \
    FT_SUCCESS_THRESHOLD=1.0

# Fail at build time, not at the first rollout: this validates the catalog,
# resolves every declared rubric check, and rejects a world tool that collides
# with a base tool name.
RUN python -c "\
from ft_rle_template.tcaas.catalog import load_catalog; \
from ft_rle_template.tcaas.tools import _world_tools; \
from ft_rle_template.graders.rubrics import all_checks; \
c = load_catalog(); \
assert _world_tools(), 'FT_WORLD_TOOLS resolved to nothing'; \
assert 'solve_efficiency' in all_checks(), 'FT_WORLD_CHECKS did not load'; \
print('world OK:', c.descriptor().world_id, c.descriptor().split_sizes)"
