"""The asimov-v1-policies checkout, the Asimov 1 model it loads, and the get-up policy."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from unittest import mock

import onnx
from huggingface_hub import hf_hub_download

from mjswan_playground._deps import ensure_repo

REPO_URL = "https://github.com/openhorizon-labs/asimov-v1-policies.git"
#: The commit the get-up policy was trained at (its spec's `git_commit`), and the last
#: one that ships the fall states its reset draws from.
REPO_COMMIT = "695ceba93f6a2fb99be03a5854d94bc55f3df1be"
FALL_STATES = "data/fall_states.v1-local/fall_states.npz"

#: Upstream's `third_party/asimov-1` submodule, at the commit the checkout pins.
MODEL_URL = "https://github.com/menloresearch/asimov-1.git"
MODEL_COMMIT = "35ae7b3581ce36762bfddb00b8fc5e7c957ee7fa"
MODEL_DIR = "sim-model"

POLICY_REPO_ID = "openhorizon-labs/asimov1-getup-safefall"
POLICY_REVISION = "f657abede4c0e8c74b96d7a2546568b63fed619f"
POLICY_ONNX = "policies/getup.onnx"
POLICY_SPEC = "policies/getup_policy_spec.json"

TASK_PACKAGE = "asimov_policies.tasks"


def resolve_root() -> Path:
    return ensure_repo(
        name="asimov-v1-policies",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=FALL_STATES,
        root_env_var="MJSWAN_ASIMOV_ROOT",
    )


def resolve_model() -> Path:
    # The repository keeps its full CAD assembly in Git LFS; the sim model is plain git.
    with mock.patch.dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"):
        return ensure_repo(
            name="asimov-1",
            url=MODEL_URL,
            commit=MODEL_COMMIT,
            marker=f"{MODEL_DIR}/xmls/asimov_1.xml",
            root_env_var="MJSWAN_ASIMOV_MODEL_ROOT",
        )


def register_tasks(root: Path, model: Path) -> None:
    """Import upstream's task package from ``src/`` so mjlab's registry knows it, with
    the robot read from ``model`` rather than the submodule the checkout leaves empty."""
    if TASK_PACKAGE in sys.modules:
        return
    # Read once, when upstream's `robot` module is imported.
    os.environ["ASIMOV_1_MODEL_DIR"] = str(model / MODEL_DIR)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    importlib.import_module(TASK_PACKAGE)


def flat_play_cfg():
    """Upstream's play config with its own ``rough`` switch off: a plane, and the
    reset's spawn height for flat ground."""
    from asimov_policies.common.env_parts import TerrainParams
    from asimov_policies.tasks.getup.env_cfg import GetUpParams, make_getup_env_cfg

    return make_getup_env_cfg(
        GetUpParams(terrain=TerrainParams(rough=False)), play=True
    )


def policy() -> tuple[onnx.ModelProto, dict]:
    """The get-up ONNX and its spec (joint orders, default pose)."""
    path = hf_hub_download(POLICY_REPO_ID, POLICY_ONNX, revision=POLICY_REVISION)
    spec = hf_hub_download(POLICY_REPO_ID, POLICY_SPEC, revision=POLICY_REVISION)
    return onnx.load(path), json.loads(Path(spec).read_text())
