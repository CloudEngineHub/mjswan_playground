"""Asimov 1 getting up from a fall. See ``README.md``."""

from __future__ import annotations

import mjswan

from . import terms, upstream

TASK_ID = "Asimov1-GetUp"


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    model = upstream.resolve_model()
    upstream.register_tasks(root, model)
    policy, spec = upstream.policy()
    env_cfg = upstream.flat_play_cfg()
    reset = env_cfg.events["reset_fall_state"]
    reset.func = terms.reset_from_fall_states
    reset.params["path"] = str(root / upstream.FALL_STATES)
    env_cfg.terminations["nan"].func = terms.nan_detection
    # Training's helper force, which play already holds at zero.
    del env_cfg.events["assist_force"]
    joint_names = spec["action"]["joint_order"]

    builder = mjswan.Builder()
    project = builder.add_project(name="Asimov 1 Get-Up", license=root / "LICENSE")
    project.set_notice(root / "NOTICE")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    # asimov-1 licenses hardware and software apart without saying which covers the sim
    # model, so the scene carries both.
    scene.add_attribution("asimov-1-hardware", license=model / "HARDWARE-LICENSE.txt")
    scene.add_attribution("asimov-1-software", license=model / "SOFTWARE-LICENSE.txt")
    scene.add_policy(
        name="getup",
        policy=policy,
        policy_joint_names=[f"robot/{name}" for name in joint_names],
        default_joint_pos=[spec["default_joint_pos"][name] for name in joint_names],
    )
    return builder
