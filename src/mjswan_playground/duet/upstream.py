"""The DUET checkout: pinned clone, task ids registered and the deployed contract."""

from __future__ import annotations

import importlib
import json
import posixpath
import sys
from pathlib import Path
from unittest import mock

from mjswan_playground._compat import collision_defaults, dropping_env_ids
from mjswan_playground._deps import ensure_repo

REPO_URL = "https://github.com/bae-air-lab/DUET.git"
REPO_COMMIT = "439c21bd2817feedd519475e8301195b95cb6fcb"

TASK_PACKAGE = "src.tasks.duet.config.g1_23dof"
#: model_15500, the checkpoint the README deploys; ``exported/`` holds two later ones.
POLICY_ONNX = "exported/policy.onnx"
CONTRACT = "exported/deploy_contract.json"


def resolve_root() -> Path:
    return ensure_repo(
        name="duet",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=POLICY_ONNX,
        root_env_var="MJSWAN_DUET_ROOT",
    )


def deployed_contract(root: Path) -> dict:
    return json.loads((root / CONTRACT).read_text())


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
    # ponytail: upstream is on mjlab 1.2.0; drop the shims once REPO_COMMIT is on 1.6.
    with (
        collision_defaults(),
        mock.patch("mjlab.utils.os.update_assets", _update_assets, create=True),
    ):
        importlib.import_module(TASK_PACKAGE)
    commands = importlib.import_module("src.tasks.common.mdp.commands")
    for name in ("_update_command", "compute"):
        method = vars(commands.BaseHeightCommand)[name]
        setattr(commands.BaseHeightCommand, name, dropping_env_ids(method))


def _update_assets(assets: dict[str, bytes], path: Path, meshdir: str | None) -> None:
    """mjlab 1.2's ``update_assets``, which 1.6 removed: every file in ``path``, keyed
    under ``meshdir``."""
    for file in Path(path).iterdir():
        if file.is_file():
            key = file.name if meshdir is None else posixpath.join(meshdir, file.name)
            assets[key] = file.read_bytes()
