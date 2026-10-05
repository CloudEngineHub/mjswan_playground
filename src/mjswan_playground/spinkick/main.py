"""Unitree G1 double spin kick demo. See ``README.md``."""

from __future__ import annotations

import mjswan
import onnx
import spinkick_example  # noqa: F401  (the spinkick extra; registers the task)
from mjlab.tasks.registry import load_env_cfg
from mjswan.mjlab import DEFAULT_OBS_GROUP_KEY
from mjswan.mjlab.onnx_meta import read_mjlab_metadata

from . import upstream

TASK_ID = "Mjlab-Spinkick-Unitree-G1"


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    policy = onnx.load(str(root / upstream.POLICY_ONNX))
    # Joint order and rest pose, as the robot's `motion_tracking_controller` reads them.
    contract = read_mjlab_metadata(policy)
    env_cfg = load_env_cfg(TASK_ID, play=True)
    clip = upstream.ensure_clip(policy, env_cfg)

    builder = mjswan.Builder()
    project = builder.add_project(name="G1 Spinkick")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    handle = scene.add_policy(
        name="spinkick_safe",
        policy=policy,
        # `time_step` picks a frame of the baked clip; later outputs are its reference.
        in_keys=[DEFAULT_OBS_GROUP_KEY, "time_step"],
        out_keys=["action", *(output.name for output in policy.graph.output[1:])],
        policy_joint_names=[f"robot/{name}" for name in contract.joint_names],
        default_joint_pos=contract.default_joint_pos,
    )
    motion = env_cfg.commands["motion"]
    handle.add_motion(
        name="Double spin kick",
        source=str(clip),
        fps=1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation),
        anchor_body_name=motion.anchor_body_name,
        body_names=motion.body_names,
        default=True,
    )
    return builder
