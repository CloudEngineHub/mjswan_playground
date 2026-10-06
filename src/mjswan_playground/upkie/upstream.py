"""The mjlab_upkie checkout: pinned clone and task ids registered."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock

from mjlab.utils.spec_config import CollisionCfg

from mjswan_playground._deps import ensure_repo

REPO_URL = "https://github.com/MarcDcls/mjlab_upkie.git"
REPO_COMMIT = "d7895789a601943f78b3b9bcdc4def7b444f9913"

#: Importing it registers `Mjlab-Velocity-Upkie`. The checkout is never installed, so
#: upstream's `mjlab.tasks` entry point never runs.
TASK_PACKAGE = "mjlab_upkie.tasks"
POLICY_ONNX = "logs/rsl_rl/upkie_velocity/bests/default.onnx"


def resolve_root() -> Path:
    return ensure_repo(
        name="mjlab_upkie",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=POLICY_ONNX,
        root_env_var="MJSWAN_UPKIE_ROOT",
    )


def register_tasks(root: Path) -> None:
    """Import upstream's task package from ``src/`` so mjlab's registry knows it."""
    if TASK_PACKAGE in sys.modules:
        return
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    with mock.patch("mjlab.utils.spec_config.CollisionCfg", _collision_cfg):
        importlib.import_module(TASK_PACKAGE)


# ponytail: upstream is on mjlab 1.3.0, which defaulted the CollisionCfg fields 1.6 made
# required; drop this once REPO_COMMIT is on 1.6.
_COLLISION_DEFAULTS = {"contype": 1, "conaffinity": 1, "condim": 3, "priority": 0}


def _collision_cfg(**kwargs) -> CollisionCfg:
    """1.3.0 defaulted the structural fields, also for geoms a dict leaves unmatched.
    Patterns match first-wins, so the catch-all goes last."""
    for name, default in _COLLISION_DEFAULTS.items():
        value = kwargs.setdefault(name, default)
        if isinstance(value, dict):
            kwargs[name] = {**value, ".*": value.get(".*", default)}
    return CollisionCfg(**kwargs)
