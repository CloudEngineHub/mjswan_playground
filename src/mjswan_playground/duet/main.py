"""Unitree G1-23DOF walking and squatting to commands, legs only. See ``README.md``."""

from __future__ import annotations

import mjswan
import onnx
from mjlab.tasks.registry import load_env_cfg

from mjswan_playground._gait_clock import clock_phase

from . import terms, upstream

TASK_ID = "Unitree-G1-23Dof-Duet-Flat"


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    contract = upstream.deployed_contract(root)
    env_cfg = load_env_cfg(TASK_ID, play=True)
    clock_phase(env_cfg)
    # model_15500 was trained before upstream moved the range to (0.24, 0.78). Play
    # pins the squat floor at full depth.
    height = env_cfg.commands["base_height"]
    height.height_range = tuple(contract["height_command_range"])
    height.floor_curriculum_start = height.height_range[0]
    env_cfg.observations["actor"].terms["height_command"].func = terms.height_command
    # Play holds the arms at their default pose; mjswan has no counterpart. Only
    # training-side terms read it.
    env_cfg.actions.pop("upper_body_pose")
    env_cfg.observations["critic"].terms.pop("arm_traj_vel")
    for name in ("arm_traj_speed", "arm_traj_accel"):
        env_cfg.metrics.pop(name)

    builder = mjswan.Builder()
    project = builder.add_project(name="G1 DUET", license=root / "LICENCE")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    scene.add_policy(
        name="model_15500",
        policy=onnx.load(str(root / upstream.POLICY_ONNX)),
        # The 13 joints the policy drives, which sets its action count.
        policy_joint_names=[f"robot/{name}" for name in contract["action_joint_names"]],
        default_joint_pos=contract["action_offset"],
    )
    return builder
