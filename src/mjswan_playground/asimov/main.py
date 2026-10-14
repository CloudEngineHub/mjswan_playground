"""Asimov 1 walking, and getting up from a fall. See ``README.md``."""

from __future__ import annotations

import mjswan

from . import terms, upstream, walk

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
    walk.raise_effort_limits(env_cfg)
    # Menlo's policy acts on the same joints, in the same order, about the same pose:
    # upstream's robot.py copies them from Menlo's training code.
    joint_names = spec["action"]["joint_order"]
    joint_kwargs = {
        "policy_joint_names": [f"robot/{name}" for name in joint_names],
        "default_joint_pos": [spec["default_joint_pos"][name] for name in joint_names],
    }

    builder = mjswan.Builder()
    project = builder.add_project(name="Asimov 1", license=root / "LICENSE")
    project.set_notice(root / "NOTICE")
    # mjswan traces every MDP of a scene against the scene's env, so the scene takes the
    # walking config, the one with the velocity command.
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=walk.env_cfg())
    # asimov-1 licenses hardware and software apart without saying which covers the sim
    # model, so the scene carries both.
    scene.add_attribution("asimov-1-hardware", license=model / "HARDWARE-LICENSE.txt")
    scene.add_attribution("asimov-1-software", license=model / "SOFTWARE-LICENSE.txt")
    # The model card says BSD-3-Clause and names no holder; Menlo's training code does.
    scene.add_attribution(
        "asimov1-locomotion",
        license="BSD-3-Clause",
        copyright="2026, Menlo Research Pte. Ltd.",
    )
    scene.add_policy(name="walk", policy=walk.policy(), default=True, **joint_kwargs)
    scene.add_policy(name="getup", policy=policy, env_cfg=env_cfg, **joint_kwargs)
    return builder
