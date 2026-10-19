"""PHP's G1 parkour teachers, each tracking a clip over its boxes. See ``README.md``."""

from __future__ import annotations

import mjswan
import onnx
from mjswan.mjlab.onnx_meta import read_mjlab_metadata

from . import php, upstream

PHP_NOTICE = "PHP training motions: https://github.com/amazon-far/php_parkour, release \
training-motion-examples-v1. Copyright Amazon.com, Inc. or its affiliates. All Rights \
Reserved."


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)

    builder = mjswan.Builder()
    project = builder.add_project(name="Parkour MJLab")
    for skill in upstream.BUNDLES:
        # Upstream's play draws each bundle's clips in turn, from its first.
        clip = upstream.clips(root, skill)[0]
        env_cfg = php.env_cfg(upstream.teacher(skill), clip)
        policy = onnx.load(str(upstream.teacher_onnx(root, skill, env_cfg)))
        contract = read_mjlab_metadata(policy)
        scene = project.add_scene_mjlab(upstream.teacher(skill), env_cfg=env_cfg)
        scene.add_attribution(
            "unitree_g1", license=root / "src/assets/robots/unitree_g1/LICENSE"
        )
        scene.add_attribution("php_parkour", license="Apache-2.0", notice=PHP_NOTICE)
        motion = env_cfg.commands["motion"]
        scene.add_policy(
            name=skill,
            policy=policy,
            policy_joint_names=[f"robot/{name}" for name in contract.joint_names],
            default_joint_pos=contract.default_joint_pos,
        ).add_motion(
            name=clip.stem.removesuffix("_motion"),
            source=motion.motion_file,
            fps=1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation),
            anchor_body_name=motion.anchor_body_name,
            body_names=motion.body_names,
            default=True,
        )
    return builder
