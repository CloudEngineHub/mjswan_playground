"""Booster K1 walking, dancing and fighting. See ``README.md``."""

from __future__ import annotations

from typing import Any

import mjswan

from . import envs, upstream

#: (scene, booster_deploy task, clip): the K1's BeyondMimic tasks, as booster_deploy's
#: `scripts/deploy.py --task` names them.
CLIPS = (
    ("K1 MJ Dance", "k1_mj2", "MJ dance, segment 1"),
    ("K1 Fight", "k1_fight", "Fight"),
)


def setup_builder() -> mjswan.Builder:
    root, assets = upstream.resolve_roots()
    builder = mjswan.Builder()
    project = builder.add_project(name="Booster K1", license=root / "LICENSE")

    def add_scene(scene_name: str, env_cfg: Any, task: Any, **policy: Any) -> Any:
        scene = project.add_scene_mjlab(scene_name, env_cfg=env_cfg)
        scene.add_attribution("booster_assets", license=assets / "LICENSE")
        return scene.add_policy(
            policy_joint_names=[f"robot/{joint}" for joint in task.robot.joint_names],
            default_joint_pos=list(task.robot.default_joint_pos),
            **policy,
        )

    walk = upstream.task_cfg(root, "k1_walk")
    add_scene(
        "K1 Walk",
        envs.walk_cfg(walk, assets),
        walk,
        name="k1_walk",
        policy=upstream.walk_onnx(walk),
        in_keys=["actor", "is_init", "adapt_hx"],
        out_keys=["action", ["next", "adapt_hx"]],
    )
    for scene_name, task_name, clip_name in CLIPS:
        task = upstream.task_cfg(root, task_name)
        env_cfg = envs.tracking_cfg(task, assets)
        clip = upstream.clip_npz(task, env_cfg)
        motion = env_cfg.commands["motion"]
        motion.motion_file = str(clip)
        handle = add_scene(
            scene_name,
            env_cfg,
            task,
            name=task_name,
            policy=upstream.tracking_onnx(task),
        )
        handle.add_motion(
            name=clip_name,
            source=str(clip),
            fps=1.0 / task.policy_dt,
            anchor_body_name=motion.anchor_body_name,
            body_names=motion.body_names,
            default=True,
        )
    return builder
