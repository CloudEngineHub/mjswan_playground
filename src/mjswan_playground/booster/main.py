"""Booster K1 walking to velocity commands, dancing and fighting. See ``README.md``."""

from __future__ import annotations

from typing import Any

import mjswan
from mjlab.tasks.registry import load_env_cfg

from . import deploy, envs, upstream

TASK_ID = "Mjlab-Velocity-Flat-Booster-K1"
#: (scene, booster_deploy task, clip): the K1's BeyondMimic tasks, as booster_deploy's
#: `scripts/deploy.py --task` names them.
CLIPS = (
    ("K1 MJ Dance", "k1_mj2", "MJ dance, segment 1"),
    ("K1 Fight", "k1_fight", "Fight"),
)


def _add_mjlab_walk(project: Any) -> None:
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    policy, demo = upstream.demo_policy(root)
    env_cfg = load_env_cfg(TASK_ID, play=True)
    # Play's resets apply the command curriculum's first stage; mjswan runs none.
    twist = env_cfg.commands["twist"]
    stage = env_cfg.curriculum["command_vel"].params["velocity_stages"][0]
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        setattr(twist.ranges, axis, stage[axis])
    # mjlab's `pseudo_inertia` does not trace (`diag_embed`, `mT`, a branch on data).
    del env_cfg.events["trunk_inertia"], env_cfg.events["limb_inertia"]

    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    scene.add_policy(
        name="walk",
        policy=policy,
        policy_joint_names=[f"robot/{joint['name']}" for joint in demo["joints"]],
        default_joint_pos=[joint["default"] for joint in demo["joints"]],
    )


def _add_deploy(project: Any) -> None:
    root, assets = deploy.resolve_roots()

    def add_scene(scene_name: str, env_cfg: Any, task: Any, **policy: Any) -> Any:
        scene = project.add_scene_mjlab(scene_name, env_cfg=env_cfg)
        scene.add_attribution("booster_deploy", license=root / "LICENSE")
        scene.add_attribution("booster_assets", license=assets / "LICENSE")
        return scene.add_policy(
            policy_joint_names=[f"robot/{joint}" for joint in task.robot.joint_names],
            default_joint_pos=list(task.robot.default_joint_pos),
            **policy,
        )

    walk = deploy.task_cfg(root, "k1_walk")
    add_scene(
        "K1 Walk",
        envs.walk_cfg(walk, assets),
        walk,
        name="k1_walk",
        policy=deploy.walk_onnx(walk),
        in_keys=["actor", "is_init", "adapt_hx"],
        out_keys=["action", ["next", "adapt_hx"]],
    )
    for scene_name, task_name, clip_name in CLIPS:
        task = deploy.task_cfg(root, task_name)
        env_cfg = envs.tracking_cfg(task, assets)
        clip = deploy.clip_npz(task, env_cfg)
        motion = env_cfg.commands["motion"]
        motion.motion_file = str(clip)
        handle = add_scene(
            scene_name,
            env_cfg,
            task,
            name=task_name,
            policy=deploy.tracking_onnx(task),
        )
        handle.add_motion(
            name=clip_name,
            source=str(clip),
            fps=1.0 / task.policy_dt,
            anchor_body_name=motion.anchor_body_name,
            body_names=motion.body_names,
            default=True,
        )


#: Parts built alone: every scene but booster_mjlab's walk, which states no license.
PARTS = {"booster-open": (_add_deploy,)}


def setup_builder(part: str | None = None) -> mjswan.Builder:
    builder = mjswan.Builder()
    project = builder.add_project(name="Booster K1")
    for add_scenes in PARTS[part] if part else (_add_mjlab_walk, _add_deploy):
        add_scenes(project)
    return builder
