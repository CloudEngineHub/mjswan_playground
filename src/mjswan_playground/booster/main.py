"""Booster K1 walking to velocity commands. See ``README.md``."""

from __future__ import annotations

import mjswan
from mjlab.tasks.registry import load_env_cfg

from . import upstream

TASK_ID = "Mjlab-Velocity-Flat-Booster-K1"


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    policy, demo = upstream.demo_policy(root)
    env_cfg = load_env_cfg(TASK_ID, play=True)
    # Play's resets apply the command curriculum's first stage; mjswan runs none.
    twist = env_cfg.commands["twist"]
    stage = env_cfg.curriculum["command_vel"].params["velocity_stages"][0]
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        setattr(twist.ranges, axis, stage[axis])
    # mjlab's `pseudo_inertia` does not trace: no ONNX op for `diag_embed` or `eigh`.
    del env_cfg.events["trunk_inertia"], env_cfg.events["limb_inertia"]

    builder = mjswan.Builder()
    project = builder.add_project(name="Booster K1")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    scene.add_policy(
        name="walk",
        policy=policy,
        policy_joint_names=[f"robot/{joint['name']}" for joint in demo["joints"]],
        default_joint_pos=[joint["default"] for joint in demo["joints"]],
    )
    return builder
