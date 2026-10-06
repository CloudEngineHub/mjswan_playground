"""Upkie, a wheeled biped, balancing and driving to velocity commands. See ``README.md``."""

from __future__ import annotations

import re

import mjswan
import onnx
from mjlab.tasks.registry import load_env_cfg
from mjswan.mjlab.onnx_meta import read_mjlab_metadata

from . import upstream

TASK_ID = "Mjlab-Velocity-Upkie"


def _action_order(env_cfg, joint_names: list[str]) -> list[str]:
    """The joints in the policy's action layout: term by term, entity order within.

    The legs are one term and the wheels another, so the action vector is not the
    model's joint order that the checkpoint's metadata lists.
    """
    order = []
    for term in env_cfg.actions.values():
        patterns = [re.compile(p) for p in term.actuator_names]
        order += [n for n in joint_names if any(p.fullmatch(n) for p in patterns)]
    if sorted(order) != sorted(joint_names):
        raise ValueError(f"The action terms do not cover {joint_names} once each.")
    return order


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    policy = onnx.load(str(root / upstream.POLICY_ONNX))
    contract = read_mjlab_metadata(policy)
    env_cfg = load_env_cfg(TASK_ID, play=True)
    joint_names = _action_order(env_cfg, list(contract.joint_names))
    default_pos = dict(zip(contract.joint_names, contract.default_joint_pos))

    builder = mjswan.Builder()
    project = builder.add_project(name="Upkie Velocity", license=root / "LICENSE")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    scene.add_policy(
        name="Default",
        policy=policy,
        policy_joint_names=[f"robot/{name}" for name in joint_names],
        default_joint_pos=[default_pos[name] for name in joint_names],
    )
    return builder
