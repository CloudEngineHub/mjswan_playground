"""Perceptive parkour: PIE's Go2 on stairs, PHP's G1 teachers over boxes and their depth
student. See ``README.md``."""

from __future__ import annotations

import json
import textwrap
from typing import Any

import mjswan
import onnx
from mjswan.mjlab.onnx_meta import read_mjlab_metadata

from . import php, pie, upstream

#: The skill the student plays, as upstream's README plays it.
STUDENT_SKILL = "High-Climb"
#: Upstream draws the student's depth 3 or 4 steps old each step; the browser keeps 3.
STUDENT_DEPTH_LAG = 3
#: Apache-2.0 asks a changed file to say what changed.
PHP_NOTICE = """\
PHP training motions: https://github.com/amazon-far/php_parkour, release
training-motion-examples-v1. Copyright Amazon.com, Inc. or its affiliates. All Rights
Reserved. Changed here: the scene's clip is rewritten in mjlab's motion layout, and its
boxes are built into the scene.
"""


def _policy_notice(checkpoint: str, change: str) -> str:
    """The notice a scene's policy carries: its checkpoint, which parkour_mjlab's
    logs/LICENSE puts under Apache-2.0, and what the port changes, as Apache-2.0 asks."""
    return (
        textwrap.fill(
            f"Policy: {checkpoint} from {upstream.REPO_URL} at "
            f"{upstream.REPO_COMMIT[:7]}, under Apache-2.0 by its logs/LICENSE. "
            f"Changed here: {change}",
            88,
        )
        + "\n"
    )


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)

    builder = mjswan.Builder()
    project = builder.add_project(name="Parkour MJLab", license=root / "LICENSE")
    project.set_notice(root / "NOTICE")

    env_cfg, history = pie.env_cfg()
    policy = onnx.load(str(root / upstream.PIE_POLICY))
    contract = read_mjlab_metadata(policy)
    scene = project.add_scene_mjlab(upstream.PIE, env_cfg=env_cfg)
    scene.add_attribution(
        "parkour_mjlab-policy",
        notice=_policy_notice(
            upstream.PIE_POLICY,
            "wrapped to carry its depth history and GRU state from step to step.",
        ),
    )
    scene.add_policy(
        name="PIE",
        policy=pie.policy(root, **history),
        commands={"twist": pie.twist(env_cfg)},
        in_keys=["proprio", "history", "depth", "is_init", "adapt_hx"],
        out_keys=["action", ["next", "adapt_hx"]],
        policy_joint_names=[f"robot/{name}" for name in contract.joint_names],
        default_joint_pos=contract.default_joint_pos,
    )

    def php_scene(
        task_id: str, env_cfg: Any, clip: str, policy_notice: str, **policy: Any
    ) -> None:
        scene = project.add_scene_mjlab(task_id, env_cfg=env_cfg)
        scene.add_attribution(
            "unitree_g1", license=root / "src/assets/robots/unitree_g1/LICENSE"
        )
        scene.add_attribution("php_parkour", license="Apache-2.0", notice=PHP_NOTICE)
        scene.add_attribution("parkour_mjlab-policy", notice=policy_notice)
        motion = env_cfg.commands["motion"]
        scene.add_policy(**policy).add_motion(
            name=clip,
            source=motion.motion_file,
            fps=1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation),
            anchor_body_name=motion.anchor_body_name,
            body_names=motion.body_names,
            default=True,
        )

    for skill in upstream.BUNDLES:
        # Upstream's play draws each bundle's clips in turn, from its first.
        clip = upstream.clips(root, skill)[0]
        env_cfg = php.env_cfg(upstream.teacher(skill), clip)
        policy = onnx.load(str(upstream.teacher_onnx(root, skill, env_cfg)))
        contract = read_mjlab_metadata(policy)
        php_scene(
            upstream.teacher(skill),
            env_cfg,
            clip.stem.removesuffix("_motion"),
            _policy_notice(
                upstream.teacher_checkpoint(skill),
                "exported to ONNX by mjlab's runner.",
            ),
            name=skill,
            policy=policy,
            policy_joint_names=[f"robot/{name}" for name in contract.joint_names],
            default_joint_pos=contract.default_joint_pos,
        )

    clip = upstream.clips(root, STUDENT_SKILL)[0]
    env_cfg = php.student_cfg(php.env_cfg(upstream.STUDENT, clip), clip)
    policy = php.student_policy(root, STUDENT_DEPTH_LAG)
    # Upstream's sim2sim contract, in the export's metadata.
    contract = json.loads(
        {p.key: p.value for p in policy.metadata_props}["php_sim2sim"]
    )
    php_scene(
        upstream.STUDENT,
        env_cfg,
        clip.stem.removesuffix("_motion"),
        _policy_notice(
            upstream.STUDENT_POLICY,
            f"wrapped to read its depth image {STUDENT_DEPTH_LAG} steps old, where "
            "upstream draws 3 or 4 each step.",
        ),
        name="Student",
        policy=policy,
        in_keys=["proprio", "depth", "is_init", "adapt_hx"],
        out_keys=["action", ["next", "adapt_hx"]],
        policy_joint_names=[f"robot/{name}" for name in contract["joint_names"]],
        default_joint_pos=contract["default_joint_pos"],
    )
    return builder
