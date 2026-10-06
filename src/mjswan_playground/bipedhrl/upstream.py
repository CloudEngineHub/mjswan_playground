"""The biped_hrl checkout: pinned clone and task ids registered."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock

from mjlab.utils.spec_config import CollisionCfg

from mjswan_playground._deps import ensure_repo

REPO_URL = "https://github.com/spaethli/biped_hrl.git"
REPO_COMMIT = "cd2c75a1f12a78fe5aeb49f78c94ba3faf3116d7"

TASK_PACKAGE = "src.tasks.velocity.config.h1_2"
POLICY_DIR = "deploy/robots/h1_2/config/policy/velocity/v0"
#: The checkpoint the author ran on the robot; the tree holds several others.
POLICY_ONNX = f"{POLICY_DIR}/exported/policy.onnx"


def resolve_root() -> Path:
    return ensure_repo(
        name="biped_hrl",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=POLICY_ONNX,
        root_env_var="MJSWAN_BIPEDHRL_ROOT",
    )


def register_tasks(root: Path) -> None:
    """Import upstream's task package so mjlab's registry knows its task ids.

    Upstream runs from source as top-level ``src``, so its root must come *first* on
    ``sys.path``: this repo has a ``src/`` of its own that would shadow it.
    """
    if TASK_PACKAGE in sys.modules:
        return
    imported = sys.modules.get("src")
    if imported is not None:
        paths = [str(path) for path in getattr(imported, "__path__", ()) or ()]
        if not any(path.startswith(str(root)) for path in paths):
            raise RuntimeError(
                f"A different top-level 'src' package is already imported ({paths}), "
                f"so upstream's own 'src.*' modules under {root} cannot be reached. "
                "Build this task in a fresh interpreter."
            )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    with mock.patch("mjlab.utils.spec_config.CollisionCfg", _collision_cfg):
        importlib.import_module(TASK_PACKAGE)


# ponytail: upstream's robot constants omit the CollisionCfg fields mjlab 1.6 made
# required; drop this once REPO_COMMIT sets them.
_COLLISION_DEFAULTS = {"contype": 1, "conaffinity": 1, "condim": 3, "priority": 0}


def _collision_cfg(**kwargs) -> CollisionCfg:
    """The pre-1.6 defaults, also for geoms a dict leaves unmatched. Patterns match
    first-wins, so the catch-all goes last."""
    for name, default in _COLLISION_DEFAULTS.items():
        value = kwargs.setdefault(name, default)
        if isinstance(value, dict):
            kwargs[name] = {**value, ".*": value.get(".*", default)}
    return CollisionCfg(**kwargs)
