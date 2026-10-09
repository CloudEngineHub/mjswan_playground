"""KingKong Robotics' jumper: walking with a commanded body posture, walking on five legs,
dancing, gesturing and jumping. See ``README.md``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mjswan
import onnx
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjswan.mjlab.observation import adapt_observations

from . import dance, five_foot, jump, servos, terms, upstream

TASK_ID = upstream.TASK_ID


def play_env_cfg() -> tuple[Any, dict[str, tuple[int, ...]]]:
    """Upstream's posture play config as the browser runs it, and each strided term's
    offsets."""
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    from tasks.jumper.posture.mdp.commands import PostureCommandCfg

    env_cfg = load_env_cfg(TASK_ID, play=True)
    servos.hold_claws(env_cfg, upstream.deployed_contract(root))
    # Play's operator terms read a keyboard or a pad; the browser's panel does instead.
    env_cfg.commands["twist"] = upstream.as_base(
        UniformVelocityCommandCfg, env_cfg.commands["twist"]
    )
    env_cfg.commands["posture"] = upstream.as_base(
        PostureCommandCfg, env_cfg.commands["posture"]
    )
    # Play adds the ToF sensor for its viewer alone; no term reads it.
    env_cfg.scene.sensors = tuple(s for s in env_cfg.scene.sensors if s.name != "tof")
    offsets = upstream.unstride(env_cfg)
    terms.clock_gait_phase(env_cfg)
    servos.observe_reset_force(env_cfg, "actuator_force")
    return env_cfg, offsets


def _add_policy(
    scene: Any,
    root: Path,
    task: str,
    name: str,
    env_cfg: Any,
    offsets: dict[str, tuple[int, ...]],
    **kwargs: Any,
) -> None:
    """``jumper.<task>``'s exported policy on ``scene``, against ``env_cfg``'s terms; a
    strided term carries its frames as ``history_steps``."""
    contract = upstream.contract(root, task)
    observations = adapt_observations(env_cfg.observations["actor"])
    for term, steps in offsets.items():
        observations["actor"].terms[term].history_steps = steps
    joints = contract["action_joint_order"]
    scene.add_policy(
        name=name,
        policy=onnx.load(str(upstream.export(root, task) / "actor.onnx")),
        env_cfg=env_cfg,
        observations=observations,
        policy_joint_names=[f"robot/{joint}" for joint in joints],
        default_joint_pos=[contract["default_joint_pos"][joint] for joint in joints],
        **kwargs,
    )


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    builder = mjswan.Builder()
    project = builder.add_project(name="jumper", license=root / "LICENSE")
    project.set_notice(root / "NOTICE")

    env_cfg, offsets = play_env_cfg()
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    _add_policy(scene, root, "posture", "model_74800", env_cfg, offsets)

    env_cfg, offsets = five_foot.play_env_cfg(root)
    scene = project.add_scene_mjlab(
        upstream.register(five_foot.SCENE_ID, env_cfg), env_cfg=env_cfg
    )
    _add_policy(scene, root, five_foot.TASK, "model_86600", env_cfg, offsets)

    # One robot, one rate: every tracking policy shares a scene, each on its own clip.
    scene = None
    for i, (name, task) in enumerate(dance.POLICIES.items()):
        env_cfg = dance.play_env_cfg(root, task)
        if scene is None:
            scene = project.add_scene_mjlab(
                upstream.register(dance.SCENE_ID, env_cfg), env_cfg=env_cfg
            )
        _add_policy(scene, root, task, name, env_cfg, {}, default=i == 0)

    # Both jumps at 200 Hz, on one scene.
    scene = None
    for i, (name, task) in enumerate(jump.POLICIES.items()):
        env_cfg, offsets = jump.play_env_cfg(root, task)
        if scene is None:
            scene = project.add_scene_mjlab(
                upstream.register(jump.SCENE_ID, env_cfg), env_cfg=env_cfg
            )
        actions = jump.actions(task, env_cfg)
        _add_policy(
            scene, root, task, name, env_cfg, offsets, default=i == 0, actions=actions
        )
    return builder
