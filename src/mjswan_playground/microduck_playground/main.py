"""Microduck Playground: every experiment it publishes a policy for, one or more scenes
each, on the same robot and servos as ``microduck``. See ``README.md``."""

from __future__ import annotations

import mjswan

from . import _common, running

#: In the order the scenes are listed.
EXPERIMENTS = (running,)


def setup_builder() -> mjswan.Builder:
    root = _common.resolve_root()
    builder = mjswan.Builder()
    project = builder.add_project(name="Microduck Playground", license=root / "LICENSE")
    project.set_notice(root / "NOTICE")
    for experiment in EXPERIMENTS:
        experiment.add_scenes(project, root)
    return builder
